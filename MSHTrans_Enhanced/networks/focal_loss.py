"""
Focal Reconstruction Loss for Time Series Anomaly Detection.

Implements asymmetric loss functions that focus on harder-to-reconstruct
patterns, which are more likely to be anomalous. Inspired by:
  - "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)
  - "Anomaly Transformer: Time Series Anomaly Detection with Association
    Discrepancy" (ICLR 2022) - for the association discrepancy idea

Key Ideas:
  1. Focal MSE: Applies higher weight to samples with larger reconstruction
     errors, making the model focus on hard examples.
  2. Adaptive Weighting: Learns to weight different time steps and features
     based on their reconstruction difficulty.
  3. Association Discrepancy: Combines reconstruction loss with a
     prior-association discrepancy for better anomaly scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalMSELoss(nn.Module):
    """Focal MSE loss that emphasizes hard-to-reconstruct samples.

    For each sample, the loss weight increases with the reconstruction error,
    making the model pay more attention to difficult patterns.

    Args:
        gamma: Focusing parameter. Higher gamma means more focus on hard
               examples. (default: 2.0)
        reduction: Loss reduction method ('mean', 'sum', 'none')
    """

    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, seq_len, n_feats) reconstructed time series
            target: (batch, seq_len, n_feats) original time series
        Returns:
            Focal MSE loss
        """
        mse = (pred - target) ** 2

        mse_normalized = mse / (mse.max().detach() + 1e-8)

        focal_weight = (1 + mse_normalized) ** self.gamma

        focal_mse = focal_weight * mse

        if self.reduction == 'mean':
            return focal_mse.mean()
        elif self.reduction == 'sum':
            return focal_mse.sum()
        return focal_mse


class AdaptiveWeightedLoss(nn.Module):
    """Learns adaptive weights for different time steps and features.

    Uses attention-based weighting to automatically identify and emphasize
    the most informative time steps and features for reconstruction.

    Args:
        seq_len: Sequence length
        n_feats: Number of features
    """

    def __init__(self, seq_len, n_feats):
        super(AdaptiveWeightedLoss, self).__init__()
        self.temporal_weight = nn.Sequential(
            nn.Linear(n_feats, n_feats),
            nn.ReLU(),
            nn.Linear(n_feats, 1),
            nn.Softmax(dim=1),
        )
        self.feature_weight = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Linear(seq_len, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, seq_len, n_feats)
            target: (batch, seq_len, n_feats)
        Returns:
            Adaptively weighted reconstruction loss
        """
        error = (pred - target) ** 2

        t_weights = self.temporal_weight(error)
        t_weights = t_weights.expand_as(error)

        f_weights = self.feature_weight(error.permute(0, 2, 1))
        f_weights = f_weights.expand_as(error.permute(0, 2, 1)).permute(0, 2, 1)

        combined_weights = t_weights * f_weights
        combined_weights = combined_weights / (combined_weights.sum(dim=(1, 2), keepdim=True) + 1e-8)

        weighted_loss = (combined_weights * error).sum(dim=(1, 2))
        return weighted_loss.mean()


class AssociationDiscrepancy(nn.Module):
    """Computes association discrepancy between prior and series associations.

    The prior association represents expected temporal correlations, while
    the series association captures actual correlations. The discrepancy
    between them helps identify anomalous time points.

    Args:
        seq_len: Sequence length
        d_model: Feature dimension
    """

    def __init__(self, seq_len, d_model):
        super(AssociationDiscrepancy, self).__init__()
        self.prior_proj = nn.Linear(d_model, d_model)
        self.series_proj = nn.Linear(d_model, d_model)
        self.sigma = nn.Parameter(torch.ones(1) * 5.0)

    def _prior_association(self, L):
        """Compute Gaussian prior association based on temporal distance."""
        positions = torch.arange(L, dtype=torch.float32)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)) ** 2
        prior = torch.exp(-dist / (2 * self.sigma ** 2))
        prior = prior / prior.sum(dim=-1, keepdim=True)
        return prior

    def forward(self, z):
        """
        Args:
            z: (batch, seq_len, d_model) encoded representations
        Returns:
            discrepancy: (batch, seq_len) association discrepancy scores
        """
        B, L, D = z.shape

        prior_assoc = self._prior_association(L).to(z.device)
        prior_assoc = prior_assoc.unsqueeze(0).expand(B, -1, -1)

        z_proj = self.series_proj(z)
        z_norm = F.normalize(z_proj, dim=-1)
        series_assoc = torch.bmm(z_norm, z_norm.transpose(1, 2))
        series_assoc = F.softmax(series_assoc, dim=-1)

        discrepancy = F.kl_div(
            series_assoc.log(),
            prior_assoc,
            reduction='none',
            log_target=False,
        ).sum(dim=-1)

        return discrepancy


class EnhancedAnomalyLoss(nn.Module):
    """Combined loss function for enhanced anomaly detection.

    Combines focal MSE reconstruction loss with association discrepancy
    and optional contrastive loss weighting.

    Args:
        seq_len: Sequence length
        n_feats: Number of features
        d_model: Feature dimension
        gamma: Focal loss focusing parameter
        assoc_weight: Weight for association discrepancy loss
    """

    def __init__(self, seq_len, n_feats, d_model, gamma=2.0, assoc_weight=0.1):
        super(EnhancedAnomalyLoss, self).__init__()
        self.focal_mse = FocalMSELoss(gamma=gamma)
        self.assoc_disc = AssociationDiscrepancy(seq_len, d_model)
        self.assoc_weight = assoc_weight

    def forward(self, pred, target, encoded_repr=None):
        """
        Args:
            pred: (batch, seq_len, n_feats) reconstructed time series
            target: (batch, seq_len, n_feats) original time series
            encoded_repr: Optional (batch, seq_len, d_model) for association discrepancy
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        rec_loss = self.focal_mse(pred, target)

        loss_dict = {'rec_loss': rec_loss.item()}
        total_loss = rec_loss

        if encoded_repr is not None:
            discrepancy = self.assoc_disc(encoded_repr)
            assoc_loss = discrepancy.mean()
            total_loss = total_loss + self.assoc_weight * assoc_loss
            loss_dict['assoc_loss'] = assoc_loss.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
