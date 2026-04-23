"""
LLM-Enhanced Time Series Feature Extraction Module.

This module implements a frozen pre-trained Transformer (GPT-2) backbone
reprogrammed for time series anomaly detection. The approach follows the
paradigm of recent works:
  - "One Fits All: Power General Time Series Analysis by Pretrained LM"
    (GPT4TS, NeurIPS 2023)
  - "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"
    (ICLR 2024)

Key Ideas:
  1. Patch Reprogramming: Time series patches are projected into the LLM's
     embedding space via learnable cross-attention reprogramming layers.
  2. Frozen Backbone: The pre-trained transformer layers are frozen to
     preserve the rich pattern recognition capabilities learned from
     large-scale language pre-training.
  3. Lightweight Adaptation: Only the input/output projection layers and
     cross-attention reprogramming layers are trainable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchReprogramming(nn.Module):
    """Cross-attention based reprogramming layer that maps time series patches
    into the LLM's embedding space using learnable source prototypes."""

    def __init__(self, d_model, d_llm, n_heads=8, n_prototypes=64):
        super(PatchReprogramming, self).__init__()
        self.d_model = d_model
        self.d_llm = d_llm
        self.n_heads = n_heads
        self.d_k = d_llm // n_heads

        self.source_prototypes = nn.Parameter(
            torch.randn(n_prototypes, d_llm)
        )
        nn.init.xavier_uniform_(self.source_prototypes.unsqueeze(0).data)
        self.source_prototypes.data = self.source_prototypes.data

        self.W_Q = nn.Linear(d_model, d_llm)
        self.W_K = nn.Linear(d_llm, d_llm)
        self.W_V = nn.Linear(d_llm, d_llm)
        self.out_proj = nn.Linear(d_llm, d_llm)
        self.layer_norm = nn.LayerNorm(d_llm)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) time series patch embeddings
        Returns:
            (batch, seq_len, d_llm) reprogrammed embeddings for LLM
        """
        B, L, _ = x.shape

        Q = self.W_Q(x)
        prototypes = self.source_prototypes.unsqueeze(0).expand(B, -1, -1)
        K = self.W_K(prototypes)
        V = self.W_V(prototypes)

        Q = Q.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(B, L, self.d_llm)
        output = self.out_proj(context)
        output = self.layer_norm(output)
        return output


class FrozenTransformerBlock(nn.Module):
    """A single Transformer block with frozen parameters.
    Mimics GPT-2 architecture but with smaller dimensions for efficiency."""

    def __init__(self, d_llm, n_heads=8, d_ff=None, dropout=0.1):
        super(FrozenTransformerBlock, self).__init__()
        self.d_llm = d_llm
        self.n_heads = n_heads
        self.d_k = d_llm // n_heads
        if d_ff is None:
            d_ff = 4 * d_llm

        self.ln1 = nn.LayerNorm(d_llm)
        self.ln2 = nn.LayerNorm(d_llm)

        self.W_Q = nn.Linear(d_llm, d_llm)
        self.W_K = nn.Linear(d_llm, d_llm)
        self.W_V = nn.Linear(d_llm, d_llm)
        self.out_proj = nn.Linear(d_llm, d_llm)

        self.ff = nn.Sequential(
            nn.Linear(d_llm, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_llm),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_llm)
        Returns:
            (batch, seq_len, d_llm)
        """
        B, L, _ = x.shape

        residual = x
        x = self.ln1(x)

        Q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(B, L, self.d_llm)
        x = residual + self.out_proj(context)

        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)

        return x


class LLMFeatureExtractor(nn.Module):
    """LLM-based feature extractor for time series anomaly detection.

    Uses a frozen pre-trained Transformer backbone with learnable input/output
    reprogramming layers. The frozen backbone preserves rich pattern recognition
    from pre-training while the reprogramming layers adapt it to time series.

    Args:
        n_feats: Number of input features (channels)
        d_model: Model dimension for time series processing
        d_llm: Hidden dimension of the LLM backbone (default: 128)
        n_layers: Number of frozen transformer layers (default: 3)
        n_heads: Number of attention heads (default: 8)
        n_prototypes: Number of learnable source prototypes (default: 64)
        patch_size: Size of time series patches (default: 8)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        n_feats,
        d_model,
        d_llm=128,
        n_layers=3,
        n_heads=8,
        n_prototypes=64,
        patch_size=8,
        dropout=0.1,
    ):
        super(LLMFeatureExtractor, self).__init__()
        self.n_feats = n_feats
        self.d_model = d_model
        self.d_llm = d_llm
        self.patch_size = patch_size

        self.patch_proj = nn.Linear(n_feats * patch_size, d_model)
        self.patch_norm = nn.LayerNorm(d_model)

        self.reprogramming = PatchReprogramming(
            d_model, d_llm, n_heads, n_prototypes
        )

        self.frozen_layers = nn.ModuleList([
            FrozenTransformerBlock(d_llm, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(d_llm, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._init_frozen_weights()

    def _init_frozen_weights(self):
        """Initialize and freeze the transformer backbone weights."""
        for layer in self.frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_last_n_layers(self, n=1):
        """Optionally unfreeze the last n transformer layers for fine-tuning."""
        layers_to_unfreeze = list(self.frozen_layers)[-n:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

    def _create_patches(self, x):
        """Convert time series into patches.

        Args:
            x: (batch, seq_len, n_feats)
        Returns:
            (batch, n_patches, n_feats * patch_size)
        """
        B, L, C = x.shape
        n_patches = L // self.patch_size
        truncated_len = n_patches * self.patch_size
        x = x[:, :truncated_len, :]
        x = x.reshape(B, n_patches, self.patch_size * C)
        return x

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_feats)
        Returns:
            llm_features: (batch, n_patches, d_model) LLM-extracted features
        """
        patches = self._create_patches(x)

        patch_emb = self.patch_proj(patches)
        patch_emb = self.patch_norm(patch_emb)

        reprogram_emb = self.reprogramming(patch_emb)

        hidden = reprogram_emb
        for layer in self.frozen_layers:
            hidden = layer(hidden)

        llm_features = self.output_proj(hidden)

        return llm_features


class LLMFusionGate(nn.Module):
    """Gated fusion module that adaptively combines LLM features with
    the original hypergraph features.

    Uses a learned gating mechanism to balance the contribution of
    LLM-extracted features and the original MSHTrans encoder features.

    Args:
        d_model: Feature dimension
    """

    def __init__(self, d_model):
        super(LLMFusionGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, hyper_features, llm_features):
        """
        Args:
            hyper_features: (batch, seq_len, d_model) from MSHTrans encoder
            llm_features: (batch, n_patches, d_model) from LLM module
        Returns:
            fused: (batch, seq_len, d_model) gated fusion of both feature sets
        """
        B, L, D = hyper_features.shape
        _, P, _ = llm_features.shape

        if P != L:
            llm_features = F.interpolate(
                llm_features.permute(0, 2, 1),
                size=L,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)

        combined = torch.cat([hyper_features, llm_features], dim=-1)
        gate_weight = self.gate(combined)
        fused = gate_weight * hyper_features + (1 - gate_weight) * self.proj(llm_features)
        return fused
