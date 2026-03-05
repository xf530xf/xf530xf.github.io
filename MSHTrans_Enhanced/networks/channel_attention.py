"""
Channel-wise Attention Module for Time Series Anomaly Detection.

Implements cross-channel attention to capture inter-feature (inter-variate)
dependencies more effectively. Inspired by:
  - "iTransformer: Inverted Transformers Are Effective for Time Series
    Forecasting" (ICLR 2024)
  - "Crossformer: Transformer Utilizing Cross-Dimension Dependency for
    Multivariate Time Series Forecasting" (ICLR 2023)

Key Ideas:
  1. Channel tokens: Each feature channel is treated as a token, with its
     temporal sequence as the token embedding.
  2. Cross-channel attention: Multi-head self-attention across channels
     captures inter-variate correlations.
  3. Adaptive channel gating: A gating mechanism controls the information
     flow between channels.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelTokenizer(nn.Module):
    """Converts time series into channel tokens.

    Each channel's temporal data is projected into a d_model-dimensional
    token representation.

    Args:
        seq_len: Sequence length (temporal dimension)
        d_model: Token embedding dimension
    """

    def __init__(self, seq_len, d_model):
        super(ChannelTokenizer, self).__init__()
        self.proj = nn.Linear(seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_feats)
        Returns:
            tokens: (batch, n_feats, d_model) channel tokens
        """
        x = x.permute(0, 2, 1)
        tokens = self.proj(x)
        tokens = self.norm(tokens)
        return tokens


class CrossChannelAttention(nn.Module):
    """Multi-head self-attention across channel dimension.

    Captures correlations between different feature channels, enabling
    the model to learn which channels are related and should be
    considered together for anomaly detection.

    Args:
        d_model: Token dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super(CrossChannelAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, d_model) channel tokens
        Returns:
            (batch, n_channels, d_model) attended channel tokens
        """
        B, C, D = x.shape
        residual = x

        Q = self.W_Q(x).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, C, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, C, D)
        out = self.out_proj(context)

        x = self.ln(residual + out)

        residual = x
        x = self.ln2(residual + self.ff(x))

        return x


class ChannelGating(nn.Module):
    """Adaptive channel gating mechanism.

    Learns to weight the importance of each channel adaptively based on
    the input, allowing the model to focus on more informative channels.

    Args:
        n_channels: Number of feature channels
        reduction: Reduction ratio for the bottleneck
    """

    def __init__(self, n_channels, reduction=4):
        super(ChannelGating, self).__init__()
        reduced = max(n_channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.Linear(n_channels, reduced),
            nn.ReLU(),
            nn.Linear(reduced, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_channels)
        Returns:
            (batch, seq_len, n_channels) channel-gated output
        """
        channel_stats = x.mean(dim=1)
        weights = self.gate(channel_stats)
        return x * weights.unsqueeze(1)


class ChannelAttentionModule(nn.Module):
    """Complete channel attention module combining tokenization,
    cross-channel attention, and adaptive gating.

    Args:
        seq_len: Temporal sequence length
        n_feats: Number of input features
        d_model: Token dimension
        n_heads: Number of attention heads
        n_layers: Number of cross-channel attention layers
        dropout: Dropout rate
    """

    def __init__(self, seq_len, n_feats, d_model, n_heads=4, n_layers=2, dropout=0.1):
        super(ChannelAttentionModule, self).__init__()

        self.tokenizer = ChannelTokenizer(seq_len, d_model)

        self.attention_layers = nn.ModuleList([
            CrossChannelAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.detokenizer = nn.Linear(d_model, seq_len)

        self.channel_gate = ChannelGating(n_feats)

        self.output_proj = nn.Linear(n_feats, n_feats)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_feats)
        Returns:
            channel_features: (batch, seq_len, n_feats) enhanced features
        """
        tokens = self.tokenizer(x)

        for layer in self.attention_layers:
            tokens = layer(tokens)

        channel_features = self.detokenizer(tokens)
        channel_features = channel_features.permute(0, 2, 1)

        channel_features = self.channel_gate(channel_features)

        channel_features = self.output_proj(channel_features)

        return channel_features
