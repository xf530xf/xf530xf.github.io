"""
Contrastive Learning Module for Time Series Anomaly Detection.

Implements dual-branch temporal contrastive learning to learn more
discriminative representations. Inspired by:
  - "TS2Vec: Towards Universal Representation of Time Series" (AAAI 2022)
  - "DCdetector: Dual Attention Contrastive Representation Learning for
    Time Series Anomaly Detection" (KDD 2023)

Key Ideas:
  1. Temporal Contrastive Loss: Encourages temporally close representations
     to be similar and distant ones to be different.
  2. Instance Contrastive Loss: Distinguishes between different instances
     within a batch using augmented views.
  3. Dual-Branch Architecture: Uses two augmentation branches (jitter and
     masking) to generate diverse views for contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesAugmentor(nn.Module):
    """Applies stochastic augmentations to time series data to create
    diverse views for contrastive learning."""

    def __init__(self, jitter_sigma=0.03, mask_ratio=0.15):
        super(TimeSeriesAugmentor, self).__init__()
        self.jitter_sigma = jitter_sigma
        self.mask_ratio = mask_ratio

    def jitter(self, x):
        """Add Gaussian noise to the time series."""
        if self.training:
            noise = torch.randn_like(x) * self.jitter_sigma
            return x + noise
        return x

    def random_mask(self, x):
        """Randomly mask portions of the time series."""
        if self.training:
            mask = torch.rand_like(x) > self.mask_ratio
            return x * mask.float()
        return x

    def forward(self, x, aug_type='jitter'):
        """Apply augmentation to create a view.

        Args:
            x: (batch, seq_len, n_feats)
            aug_type: 'jitter' or 'mask'
        Returns:
            Augmented time series
        """
        if aug_type == 'jitter':
            return self.jitter(x)
        elif aug_type == 'mask':
            return self.random_mask(x)
        return x


class TemporalContrastiveLoss(nn.Module):
    """Temporal contrastive loss that encourages temporally adjacent
    representations to be similar.

    For each timestamp, nearby timestamps within a window are treated as
    positive pairs, while distant timestamps are negative pairs.

    Args:
        temperature: Temperature for the softmax (default: 0.07)
        positive_window: Number of adjacent timestamps to treat as positives
    """

    def __init__(self, temperature=0.07, positive_window=5):
        super(TemporalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.positive_window = positive_window

    def forward(self, z1, z2):
        """
        Args:
            z1: (batch, seq_len, d_model) representations from branch 1
            z2: (batch, seq_len, d_model) representations from branch 2
        Returns:
            Temporal contrastive loss scalar
        """
        B, T, D = z1.shape

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        sim_matrix = torch.bmm(z1, z2.transpose(1, 2)) / self.temperature

        pos_mask = torch.zeros(T, T, device=z1.device)
        for i in range(T):
            start = max(0, i - self.positive_window)
            end = min(T, i + self.positive_window + 1)
            pos_mask[i, start:end] = 1.0
        pos_mask = pos_mask.unsqueeze(0).expand(B, -1, -1)

        neg_mask = 1.0 - pos_mask

        pos_sim = (sim_matrix * pos_mask).sum(dim=-1) / pos_mask.sum(dim=-1).clamp(min=1)

        neg_exp = (torch.exp(sim_matrix) * neg_mask).sum(dim=-1)
        pos_exp = torch.exp(pos_sim)

        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))
        return loss.mean()


class InstanceContrastiveLoss(nn.Module):
    """Instance-level contrastive loss (InfoNCE style).

    Treats augmented views of the same instance as positives and
    other instances in the batch as negatives.

    Args:
        temperature: Temperature for the softmax
    """

    def __init__(self, temperature=0.2):
        super(InstanceContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1: (batch, d_model) instance representations from branch 1
            z2: (batch, d_model) instance representations from branch 2
        Returns:
            Instance contrastive loss scalar
        """
        B = z1.shape[0]

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        sim_11 = torch.mm(z1, z1.t()) / self.temperature
        sim_22 = torch.mm(z2, z2.t()) / self.temperature
        sim_12 = torch.mm(z1, z2.t()) / self.temperature

        mask = torch.eye(B, device=z1.device).bool()
        sim_11 = sim_11.masked_fill(mask, float('-inf'))
        sim_22 = sim_22.masked_fill(mask, float('-inf'))

        logits_1 = torch.cat([sim_12, sim_11], dim=1)
        logits_2 = torch.cat([sim_22, sim_12.t()], dim=1)

        labels = torch.arange(B, device=z1.device)
        loss = (F.cross_entropy(logits_1, labels) + F.cross_entropy(logits_2, labels)) / 2
        return loss


class DualBranchContrastive(nn.Module):
    """Dual-branch contrastive learning framework.

    Creates two augmented views of the input and computes both temporal
    and instance-level contrastive losses to improve representation learning.

    Args:
        d_model: Feature dimension
        proj_dim: Projection head dimension (default: 64)
        temperature: Temperature for contrastive losses
        temporal_weight: Weight for temporal contrastive loss
        instance_weight: Weight for instance contrastive loss
    """

    def __init__(
        self,
        d_model,
        proj_dim=64,
        temperature=0.07,
        temporal_weight=0.5,
        instance_weight=0.5,
    ):
        super(DualBranchContrastive, self).__init__()

        self.augmentor = TimeSeriesAugmentor()

        self.temporal_proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        self.instance_proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        self.temporal_loss = TemporalContrastiveLoss(temperature)
        self.instance_loss = InstanceContrastiveLoss(temperature)

        self.temporal_weight = temporal_weight
        self.instance_weight = instance_weight

    def forward(self, x, encoder_fn):
        """
        Args:
            x: (batch, seq_len, n_feats) input time series
            encoder_fn: Callable that takes x and returns (batch, seq_len, d_model)
        Returns:
            contrastive_loss: Combined contrastive loss scalar
        """
        view1 = self.augmentor(x, 'jitter')
        view2 = self.augmentor(x, 'mask')

        z1 = encoder_fn(view1)
        z2 = encoder_fn(view2)

        z1_temporal = self.temporal_proj(z1)
        z2_temporal = self.temporal_proj(z2)
        loss_temporal = self.temporal_loss(z1_temporal, z2_temporal)

        z1_instance = self.instance_proj(z1.mean(dim=1))
        z2_instance = self.instance_proj(z2.mean(dim=1))
        loss_instance = self.instance_loss(z1_instance, z2_instance)

        total_loss = (
            self.temporal_weight * loss_temporal
            + self.instance_weight * loss_instance
        )
        return total_loss
