"""
Multi-Scale Patch Embedding Module for Time Series Anomaly Detection.

Implements learnable multi-scale patching that captures local patterns
at different granularities. Inspired by:
  - "PatchTST: A Time Series is Worth 64 Words" (ICLR 2023)
  - "Pathformer: Multi-scale Transformers with Adaptive Pathways for
    Time Series Forecasting" (ICLR 2024)

Key Ideas:
  1. Multi-scale patches: Multiple patch sizes capture patterns at different
     temporal resolutions (fine-grained local + coarse global patterns).
  2. Learnable patch aggregation: Attention-based aggregation of multi-scale
     patch representations.
  3. Positional encoding: Learnable position embeddings for each scale.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleScalePatchEmbed(nn.Module):
    """Patch embedding for a single scale.

    Args:
        n_feats: Number of input features
        patch_size: Size of each patch
        d_model: Output embedding dimension
        stride: Stride for patch extraction (default: same as patch_size)
    """

    def __init__(self, n_feats, patch_size, d_model, stride=None):
        super(SingleScalePatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size

        self.proj = nn.Conv1d(
            in_channels=n_feats,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=self.stride,
            padding=0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_feats)
        Returns:
            patches: (batch, n_patches, d_model)
        """
        x = x.permute(0, 2, 1)
        patches = self.proj(x)
        patches = patches.permute(0, 2, 1)
        patches = self.norm(patches)
        return patches


class MultiScalePatchEmbed(nn.Module):
    """Multi-scale patch embedding that captures patterns at different
    temporal resolutions.

    Args:
        n_feats: Number of input features
        seq_len: Input sequence length
        d_model: Embedding dimension
        patch_sizes: List of patch sizes for different scales
        strides: List of strides for each scale (default: same as patch_sizes)
    """

    def __init__(self, n_feats, seq_len, d_model, patch_sizes=None, strides=None):
        super(MultiScalePatchEmbed, self).__init__()

        if patch_sizes is None:
            patch_sizes = [4, 8, 16]
        if strides is None:
            strides = [s // 2 for s in patch_sizes]

        self.n_scales = len(patch_sizes)
        self.patch_sizes = patch_sizes
        self.d_model = d_model

        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.n_patches_list = []

        for i, (ps, st) in enumerate(zip(patch_sizes, strides)):
            self.patch_embeds.append(
                SingleScalePatchEmbed(n_feats, ps, d_model, st)
            )
            n_patches = (seq_len - ps) // st + 1
            self.n_patches_list.append(n_patches)
            self.pos_embeds.append(
                nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
            )

        self.scale_attention = ScaleAggregation(d_model, self.n_scales)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_feats)
        Returns:
            multi_scale_features: List of (batch, n_patches_i, d_model) for each scale
            aggregated: (batch, seq_len, d_model) aggregated multi-scale features
        """
        B, L, _ = x.shape
        scale_features = []

        for i in range(self.n_scales):
            patches = self.patch_embeds[i](x)
            patches = patches + self.pos_embeds[i][:, :patches.size(1), :]
            scale_features.append(patches)

        aggregated = self.scale_attention(scale_features, L)

        return scale_features, aggregated


class ScaleAggregation(nn.Module):
    """Attention-based aggregation of multi-scale patch representations.

    Learns to combine representations from different scales using
    an attention mechanism that weights the contribution of each scale.

    Args:
        d_model: Feature dimension
        n_scales: Number of scales
    """

    def __init__(self, d_model, n_scales):
        super(ScaleAggregation, self).__init__()
        self.n_scales = n_scales
        self.d_model = d_model

        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        self.scale_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_scales)
        ])
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, scale_features, target_len):
        """
        Args:
            scale_features: List of (batch, n_patches_i, d_model)
            target_len: Target sequence length for output
        Returns:
            aggregated: (batch, target_len, d_model)
        """
        B = scale_features[0].shape[0]
        weights = F.softmax(self.scale_weights, dim=0)

        upsampled = []
        for i, features in enumerate(scale_features):
            proj = self.scale_projs[i](features)

            up = F.interpolate(
                proj.permute(0, 2, 1),
                size=target_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)

            upsampled.append(up * weights[i])

        aggregated = sum(upsampled)
        aggregated = self.output_proj(aggregated)
        aggregated = self.norm(aggregated)

        return aggregated
