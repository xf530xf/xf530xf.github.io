"""
Unit tests for the new innovation modules in MSHTrans-LLM Enhanced.

Tests cover:
1. LLM Feature Extractor module
2. Contrastive Learning module
3. Channel Attention module
4. Multi-Scale Patch Embedding module
5. Focal Loss module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import unittest


class TestLLMModule(unittest.TestCase):
    """Tests for the LLM-Enhanced Feature Extraction module."""

    def setUp(self):
        from networks.llm_module import (
            LLMFeatureExtractor,
            LLMFusionGate,
            PatchReprogramming,
            FrozenTransformerBlock,
        )
        self.LLMFeatureExtractor = LLMFeatureExtractor
        self.LLMFusionGate = LLMFusionGate
        self.PatchReprogramming = PatchReprogramming
        self.FrozenTransformerBlock = FrozenTransformerBlock
        self.batch_size = 4
        self.seq_len = 100
        self.n_feats = 25

    def test_patch_reprogramming_output_shape(self):
        reprogram = self.PatchReprogramming(d_model=32, d_llm=64, n_heads=4, n_prototypes=16)
        x = torch.randn(self.batch_size, 12, 32)
        out = reprogram(x)
        self.assertEqual(out.shape, (self.batch_size, 12, 64))

    def test_frozen_transformer_block(self):
        block = self.FrozenTransformerBlock(d_llm=64, n_heads=4)
        x = torch.randn(self.batch_size, 12, 64)
        out = block(x)
        self.assertEqual(out.shape, (self.batch_size, 12, 64))

    def test_llm_feature_extractor_output(self):
        extractor = self.LLMFeatureExtractor(
            n_feats=self.n_feats, d_model=self.n_feats,
            d_llm=64, n_layers=2, n_heads=4,
            n_prototypes=16, patch_size=10
        )
        x = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        out = extractor(x)
        n_patches = self.seq_len // 10
        self.assertEqual(out.shape, (self.batch_size, n_patches, self.n_feats))

    def test_llm_frozen_parameters(self):
        extractor = self.LLMFeatureExtractor(
            n_feats=self.n_feats, d_model=self.n_feats,
            d_llm=64, n_layers=2, n_heads=4,
        )
        frozen_count = sum(1 for p in extractor.frozen_layers.parameters() if not p.requires_grad)
        total_frozen_params = sum(1 for p in extractor.frozen_layers.parameters())
        self.assertEqual(frozen_count, total_frozen_params)

    def test_llm_fusion_gate(self):
        gate = self.LLMFusionGate(d_model=self.n_feats)
        hyper_features = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        llm_features = torch.randn(self.batch_size, 10, self.n_feats)
        fused = gate(hyper_features, llm_features)
        self.assertEqual(fused.shape, (self.batch_size, self.seq_len, self.n_feats))

    def test_llm_fusion_gate_same_length(self):
        gate = self.LLMFusionGate(d_model=self.n_feats)
        hyper_features = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        llm_features = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        fused = gate(hyper_features, llm_features)
        self.assertEqual(fused.shape, (self.batch_size, self.seq_len, self.n_feats))


class TestContrastiveModule(unittest.TestCase):
    """Tests for the Contrastive Learning module."""

    def setUp(self):
        from networks.contrastive import (
            TimeSeriesAugmentor,
            TemporalContrastiveLoss,
            InstanceContrastiveLoss,
            DualBranchContrastive,
        )
        self.TimeSeriesAugmentor = TimeSeriesAugmentor
        self.TemporalContrastiveLoss = TemporalContrastiveLoss
        self.InstanceContrastiveLoss = InstanceContrastiveLoss
        self.DualBranchContrastive = DualBranchContrastive
        self.batch_size = 4
        self.seq_len = 50
        self.d_model = 32

    def test_augmentor_jitter(self):
        aug = self.TimeSeriesAugmentor(jitter_sigma=0.1)
        aug.train()
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        out = aug(x, 'jitter')
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.allclose(x, out))

    def test_augmentor_mask(self):
        aug = self.TimeSeriesAugmentor(mask_ratio=0.5)
        aug.train()
        x = torch.ones(self.batch_size, self.seq_len, self.d_model)
        out = aug(x, 'mask')
        self.assertEqual(out.shape, x.shape)
        self.assertTrue((out == 0).any())

    def test_temporal_contrastive_loss(self):
        loss_fn = self.TemporalContrastiveLoss(temperature=0.1, positive_window=3)
        z1 = torch.randn(self.batch_size, self.seq_len, self.d_model)
        z2 = torch.randn(self.batch_size, self.seq_len, self.d_model)
        loss = loss_fn(z1, z2)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

    def test_instance_contrastive_loss(self):
        loss_fn = self.InstanceContrastiveLoss(temperature=0.2)
        z1 = torch.randn(self.batch_size, self.d_model)
        z2 = torch.randn(self.batch_size, self.d_model)
        loss = loss_fn(z1, z2)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

    def test_dual_branch_contrastive(self):
        dbc = self.DualBranchContrastive(d_model=self.d_model, proj_dim=16)
        dbc.train()
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        encoder_fn = lambda inp: inp
        loss = dbc(x, encoder_fn)
        self.assertEqual(loss.dim(), 0)


class TestChannelAttention(unittest.TestCase):
    """Tests for the Channel Attention module."""

    def setUp(self):
        from networks.channel_attention import (
            ChannelTokenizer,
            CrossChannelAttention,
            ChannelGating,
            ChannelAttentionModule,
        )
        self.ChannelTokenizer = ChannelTokenizer
        self.CrossChannelAttention = CrossChannelAttention
        self.ChannelGating = ChannelGating
        self.ChannelAttentionModule = ChannelAttentionModule
        self.batch_size = 4
        self.seq_len = 100
        self.n_feats = 25

    def test_channel_tokenizer(self):
        tokenizer = self.ChannelTokenizer(seq_len=self.seq_len, d_model=64)
        x = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        tokens = tokenizer(x)
        self.assertEqual(tokens.shape, (self.batch_size, self.n_feats, 64))

    def test_cross_channel_attention(self):
        attn = self.CrossChannelAttention(d_model=64, n_heads=4)
        x = torch.randn(self.batch_size, self.n_feats, 64)
        out = attn(x)
        self.assertEqual(out.shape, (self.batch_size, self.n_feats, 64))

    def test_channel_gating(self):
        gate = self.ChannelGating(n_channels=self.n_feats)
        x = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        out = gate(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.n_feats))

    def test_channel_attention_module(self):
        module = self.ChannelAttentionModule(
            seq_len=self.seq_len, n_feats=self.n_feats,
            d_model=64, n_heads=4, n_layers=2
        )
        x = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        out = module(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.n_feats))


class TestPatchEmbedding(unittest.TestCase):
    """Tests for the Multi-Scale Patch Embedding module."""

    def setUp(self):
        from networks.patch_embedding import (
            SingleScalePatchEmbed,
            MultiScalePatchEmbed,
            ScaleAggregation,
        )
        self.SingleScalePatchEmbed = SingleScalePatchEmbed
        self.MultiScalePatchEmbed = MultiScalePatchEmbed
        self.ScaleAggregation = ScaleAggregation
        self.batch_size = 4
        self.seq_len = 100
        self.n_feats = 25

    def test_single_scale_patch(self):
        embed = self.SingleScalePatchEmbed(n_feats=self.n_feats, patch_size=10, d_model=32)
        x = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        out = embed(x)
        expected_patches = self.seq_len // 10
        self.assertEqual(out.shape, (self.batch_size, expected_patches, 32))

    def test_multi_scale_patch(self):
        embed = self.MultiScalePatchEmbed(
            n_feats=self.n_feats, seq_len=self.seq_len,
            d_model=32, patch_sizes=[4, 8, 16], strides=[2, 4, 8]
        )
        x = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        scale_features, aggregated = embed(x)
        self.assertEqual(len(scale_features), 3)
        self.assertEqual(aggregated.shape, (self.batch_size, self.seq_len, 32))

    def test_scale_aggregation(self):
        agg = self.ScaleAggregation(d_model=32, n_scales=3)
        features = [
            torch.randn(self.batch_size, 49, 32),
            torch.randn(self.batch_size, 12, 32),
            torch.randn(self.batch_size, 6, 32),
        ]
        out = agg(features, target_len=self.seq_len)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, 32))


class TestFocalLoss(unittest.TestCase):
    """Tests for the Focal Loss module."""

    def setUp(self):
        from networks.focal_loss import (
            FocalMSELoss,
            AdaptiveWeightedLoss,
            AssociationDiscrepancy,
            EnhancedAnomalyLoss,
        )
        self.FocalMSELoss = FocalMSELoss
        self.AdaptiveWeightedLoss = AdaptiveWeightedLoss
        self.AssociationDiscrepancy = AssociationDiscrepancy
        self.EnhancedAnomalyLoss = EnhancedAnomalyLoss
        self.batch_size = 4
        self.seq_len = 50
        self.n_feats = 25

    def test_focal_mse_loss(self):
        loss_fn = self.FocalMSELoss(gamma=2.0)
        pred = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        target = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

    def test_focal_vs_standard_mse(self):
        """Focal loss should emphasize hard examples more."""
        focal_loss_fn = self.FocalMSELoss(gamma=2.0)
        standard_loss_fn = nn.MSELoss()
        pred = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        target = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        focal = focal_loss_fn(pred, target)
        standard = standard_loss_fn(pred, target)
        self.assertGreater(focal.item(), standard.item())

    def test_adaptive_weighted_loss(self):
        loss_fn = self.AdaptiveWeightedLoss(seq_len=self.seq_len, n_feats=self.n_feats)
        pred = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        target = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

    def test_association_discrepancy(self):
        assoc = self.AssociationDiscrepancy(seq_len=self.seq_len, d_model=32)
        z = torch.randn(self.batch_size, self.seq_len, 32)
        discrepancy = assoc(z)
        self.assertEqual(discrepancy.shape, (self.batch_size, self.seq_len))

    def test_enhanced_anomaly_loss(self):
        loss_fn = self.EnhancedAnomalyLoss(
            seq_len=self.seq_len, n_feats=self.n_feats, d_model=32
        )
        pred = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        target = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        encoded = torch.randn(self.batch_size, self.seq_len, 32)
        total_loss, loss_dict = loss_fn(pred, target, encoded)
        self.assertEqual(total_loss.dim(), 0)
        self.assertIn('rec_loss', loss_dict)
        self.assertIn('assoc_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)

    def test_enhanced_anomaly_loss_no_encoding(self):
        loss_fn = self.EnhancedAnomalyLoss(
            seq_len=self.seq_len, n_feats=self.n_feats, d_model=32
        )
        pred = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        target = torch.randn(self.batch_size, self.seq_len, self.n_feats)
        total_loss, loss_dict = loss_fn(pred, target)
        self.assertNotIn('assoc_loss', loss_dict)


if __name__ == '__main__':
    unittest.main()
