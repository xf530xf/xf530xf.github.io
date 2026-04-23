"""
MSHTrans-LLM Enhanced: Multi-Scale Hypergraph Transformer with LLM Integration
for Temporal Anomaly Detection.

This enhanced version integrates the following innovations on top of the
original MSHTrans (KDD 2025):

1. LLM-Enhanced Feature Extraction: Frozen GPT-2-style transformer backbone
   with patch reprogramming for rich pattern recognition.
2. Dual-Branch Contrastive Learning: Temporal and instance-level contrastive
   losses for more discriminative anomaly representations.
3. Channel-wise Attention: Cross-channel attention module to capture
   inter-variate dependencies (inspired by iTransformer).
4. Multi-Scale Patch Embedding: Learnable multi-resolution patching for
   better local pattern capture (inspired by PatchTST).
5. Focal Reconstruction Loss: Asymmetric loss that focuses on hard-to-
   reconstruct patterns (likely anomalous).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from torch_geometric.data import data as D
from torch.nn import Linear
from networks.MAHLayer import Multi_Adaptive_Hypergraph
from networks.Layers import (
    Bottleneck_Construct,
    SeriesDecomposition,
    SeasonTrendFusion,
    MSFusion,
    PositionalEncoding,
)
from networks.HyperGraphConv import HypergraphConv
from networks.llm_module import LLMFeatureExtractor, LLMFusionGate
from networks.contrastive import DualBranchContrastive
from networks.channel_attention import ChannelAttentionModule
from networks.patch_embedding import MultiScalePatchEmbed
from networks.focal_loss import FocalMSELoss, EnhancedAnomalyLoss


class MSHTrans(nn.Module):
    """Enhanced MSHTrans with LLM integration and modern innovations.

    This model extends the original MSHTrans architecture with:
    - A frozen LLM backbone for extracting rich temporal features
    - Channel-wise attention for inter-variate correlation modeling
    - Multi-scale patch embeddings for multi-resolution local patterns
    - Dual-branch contrastive learning for discriminative representations
    - Focal reconstruction loss for anomaly-aware training

    Args:
        args: Dictionary of model hyperparameters
        device: torch.device
    """

    def __init__(self, args, device):
        super(MSHTrans, self).__init__()
        self.n_feats = args["n_feats"]
        self.window_size = args["window_size"]
        self.max_seq_length = self.window_size
        self.hyper_num = args["scale_num"]
        self.lr = args["lr"]
        self.model_root = args["model_root"]
        self.device = device
        self.kernel_size = args["pool_size_list"]

        # --- Original MSHTrans Components ---
        self.conv_layers = Bottleneck_Construct(
            self.n_feats, args["pool_size_list"], self.n_feats
        )
        self.seq_length = [self.max_seq_length]
        for i in range(self.hyper_num - 1):
            self.seq_length.append(self.seq_length[-1] // args["pool_size_list"][i])
        self.hyper_head = args["head_num"]

        self.multi_adpive_hypergraph = Multi_Adaptive_Hypergraph(args, self.device)

        self.pos_encoder = PositionalEncoding(
            self.n_feats * 2, 0.1, self.window_size
        )

        self.hyconv_list = nn.ModuleList()
        for i in range(self.hyper_num):
            self.hyconv = nn.ModuleList()
            for j in range(self.hyper_head):
                self.hyconv.append(
                    HypergraphConv(self.n_feats * 2, self.n_feats * 2)
                )
            self.hyconv_list.append(self.hyconv)

        self.series_decomposition = nn.ModuleList()
        self.season_trend_fusion = nn.ModuleList()
        for i in range(self.hyper_num):
            seq_len = self.seq_length[i]
            self.series_decomposition.append(
                SeriesDecomposition(seq_len, self.n_feats * 2)
            )
            self.season_trend_fusion.append(
                SeasonTrendFusion(self.n_feats * 2, self.n_feats * 2)
            )

        self.msfusion = MSFusion(
            self.n_feats * 2,
            args["pool_size_list"],
            self.n_feats * 2,
            self.seq_length,
        )

        self.series_decomposition_d1 = SeriesDecomposition(
            self.max_seq_length, self.n_feats
        )
        self.series_decomposition_d2 = SeriesDecomposition(
            self.max_seq_length, args["n_feats"]
        )
        self.hyconv_d1 = nn.ModuleList()
        for j in range(self.hyper_head):
            self.hyconv_d1.append(HypergraphConv(self.n_feats * 2, self.n_feats))
        self.season_trend_fusion_d1 = SeasonTrendFusion(self.n_feats, self.n_feats)
        self.sigmoid = nn.Sigmoid()

        self.node_embedding_proj = Linear(args["d_model"], self.n_feats)

        # --- Innovation 1: LLM Feature Extractor ---
        self.use_llm = args.get("use_llm", True)
        if self.use_llm:
            self.llm_extractor = LLMFeatureExtractor(
                n_feats=self.n_feats,
                d_model=self.n_feats,
                d_llm=args.get("d_llm", 128),
                n_layers=args.get("llm_layers", 3),
                n_heads=args.get("llm_heads", 8),
                n_prototypes=args.get("llm_prototypes", 64),
                patch_size=args.get("llm_patch_size", 8),
                dropout=0.1,
            )
            self.llm_fusion = LLMFusionGate(self.n_feats)

        # --- Innovation 3: Channel Attention ---
        self.use_channel_attn = args.get("use_channel_attn", True)
        if self.use_channel_attn:
            self.channel_attn = ChannelAttentionModule(
                seq_len=self.window_size,
                n_feats=self.n_feats,
                d_model=args.get("channel_d_model", 64),
                n_heads=args.get("channel_heads", 4),
                n_layers=args.get("channel_layers", 2),
                dropout=0.1,
            )
            self.channel_fusion_weight = nn.Parameter(torch.tensor(0.3))

        # --- Innovation 4: Multi-Scale Patch Embedding ---
        self.use_patch_embed = args.get("use_patch_embed", True)
        if self.use_patch_embed:
            self.patch_embed = MultiScalePatchEmbed(
                n_feats=self.n_feats,
                seq_len=self.window_size,
                d_model=self.n_feats,
                patch_sizes=args.get("patch_sizes", [4, 8, 16]),
                strides=args.get("patch_strides", [2, 4, 8]),
            )
            self.patch_proj = nn.Linear(self.n_feats, self.n_feats)

        # --- Innovation 2: Contrastive Learning ---
        self.use_contrastive = args.get("use_contrastive", True)
        if self.use_contrastive:
            self.contrastive = DualBranchContrastive(
                d_model=self.n_feats,
                proj_dim=args.get("contrastive_proj_dim", 64),
                temperature=args.get("contrastive_temp", 0.07),
                temporal_weight=args.get("contrastive_temporal_w", 0.5),
                instance_weight=args.get("contrastive_instance_w", 0.5),
            )

        # --- Innovation 5: Enhanced Loss ---
        self.use_focal_loss = args.get("use_focal_loss", True)
        if self.use_focal_loss:
            self.enhanced_loss = EnhancedAnomalyLoss(
                seq_len=self.window_size,
                n_feats=self.n_feats,
                d_model=self.n_feats,
                gamma=args.get("focal_gamma", 2.0),
                assoc_weight=args.get("assoc_weight", 0.1),
            )

        # Optimizer setup
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 5, 0.9
        )

    def _enhance_input(self, x):
        """Apply input enhancements (channel attention + patch embedding + LLM).

        Args:
            x: (batch, seq_len, n_feats)
        Returns:
            enhanced_x: (batch, seq_len, n_feats)
        """
        enhanced = x

        # Channel attention enhancement
        if self.use_channel_attn:
            channel_features = self.channel_attn(x)
            w = torch.sigmoid(self.channel_fusion_weight)
            enhanced = w * enhanced + (1 - w) * channel_features

        # Multi-scale patch embedding enhancement
        if self.use_patch_embed:
            _, patch_features = self.patch_embed(x)
            patch_features = self.patch_proj(patch_features)
            enhanced = enhanced + 0.1 * patch_features

        # LLM feature extraction and fusion
        if self.use_llm:
            llm_features = self.llm_extractor(x)
            enhanced = self.llm_fusion(enhanced, llm_features)

        return enhanced

    def _contrastive_encode(self, x):
        """Lightweight encoder for contrastive learning.

        Args:
            x: (batch, seq_len, n_feats)
        Returns:
            (batch, seq_len, n_feats)
        """
        return self._enhance_input(x)

    def extract_downsample(self, x):
        window_x_list = [x]
        for i in range(self.hyper_num - 1):
            idx = [
                j * pow(self.kernel_size[i], i + 1)
                for j in range(self.seq_length[i + 1])
            ]
            sequence = x[:, idx, :]
            window_x_list.append(sequence)

        return window_x_list

    def encoder(self, x, hyper_graph_indicies):
        window_ori_x = self.extract_downsample(x)

        seq_enc = self.conv_layers(x)

        for i in range(self.hyper_num):
            seq_enc[i] = torch.concat([seq_enc[i], window_ori_x[i]], dim=-1)
            seq_enc[i] = self.pos_encoder(seq_enc[i])

        st_fusion_list = []
        for i in range(self.hyper_num):
            hyperedge_indices = torch.tensor(hyper_graph_indicies[i]).to(
                self.device
            )

            node_value = seq_enc[i].permute(0, 2, 1).to(self.device)

            edge_indices, node_indices = hyperedge_indices[1], hyperedge_indices[0]

            num_edges = edge_indices.max().item() + 1
            num_nodes = node_value.size(2)

            indices = torch.stack([edge_indices, node_indices])
            values = torch.ones(edge_indices.size(0), device=node_value.device)
            adj_matrix = torch.sparse_coo_tensor(
                indices, values, (num_edges, num_nodes), device=node_value.device
            )

            node_value = node_value.permute(2, 0, 1).contiguous()

            edge_features = torch.sparse.mm(
                adj_matrix, node_value.view(num_nodes, -1)
            ).view(num_edges, node_value.size(1), node_value.size(2))

            multi_head_hyconv = self.hyconv_list[i]
            output_list = []
            for j in range(self.hyper_head):
                output = multi_head_hyconv[j](
                    seq_enc[i], hyperedge_indices, edge_features
                ).permute(1, 0, 2)
                output_list.append(output)
            multi_head_node_emb = torch.mean(
                torch.stack(output_list, dim=-1), dim=-1
            )

            multi_head_node_emb = multi_head_node_emb + seq_enc[i]

            seasonality, trend = self.series_decomposition[i](multi_head_node_emb)

            st_fusion = self.season_trend_fusion[i](
                seq_enc[i], seasonality, trend
            )
            st_fusion_list.append(st_fusion)
        fused_logits = self.msfusion(st_fusion_list)
        return fused_logits

    def decoder(self, x, fused_logits, hyperedge_index):
        edge_features = {}

        node_value = x.permute(0, 2, 1)
        hyperedge_index = hyperedge_index.to(self.device)

        edge_indices, node_indices = hyperedge_index[1], hyperedge_index[0]

        num_edges = edge_indices.max().item() + 1
        num_nodes = node_value.size(2)

        indices = torch.stack([edge_indices, node_indices])
        values = torch.ones(edge_indices.size(0), device=node_value.device)
        adj_matrix = torch.sparse_coo_tensor(
            indices, values, (num_edges, num_nodes), device=node_value.device
        )

        node_value = node_value.permute(2, 0, 1).contiguous()

        edge_features = torch.sparse.mm(
            adj_matrix, node_value.view(num_nodes, -1)
        ).view(num_edges, node_value.size(1), node_value.size(2))

        z_sea_1, z_trend_1 = self.series_decomposition_d1(x)

        input_hyconv = torch.concat([z_sea_1, fused_logits], dim=-1)
        input_hyconv = self.pos_encoder(input_hyconv)

        output_list = []
        for i in range(self.hyper_head):
            output = self.hyconv_d1[i](
                input_hyconv, hyperedge_index, edge_features
            ).permute(1, 0, 2)
            output_list.append(output)

        multi_head_node_emb = torch.mean(
            torch.stack(output_list, dim=-1), dim=-1
        )
        z_sea_2, z_trend_2 = self.series_decomposition_d2(multi_head_node_emb)

        z_trend_3 = z_trend_1 + z_trend_2
        results = self.season_trend_fusion_d1(x, z_sea_2, z_trend_3)

        results = self.sigmoid(results)

        return results

    def forward(self, x, hyper_graph_indicies, fused_hypergraph):
        # Apply input enhancements
        enhanced_x = self._enhance_input(x)

        fused_logits = self.encoder(enhanced_x, hyper_graph_indicies)
        predict_logits = self.decoder(enhanced_x, fused_logits, fused_hypergraph)

        return predict_logits

    def hyperedge_constraint(
        self,
        window_ori_x,
        hyper_graph_indicies,
        node_embedding_list,
        edge_embedding_list,
        edge_retain_list,
    ):
        loss_hyperedge_all_scale = 0.0
        loss_node_all_scale = 0.0
        for i in range(self.hyper_num):
            hyper_graph_index = torch.tensor(hyper_graph_indicies[i]).to(
                self.device
            )

            edge_embedding = edge_embedding_list[i][edge_retain_list[i], :]
            x = window_ori_x[i]
            node_value = x.permute(0, 2, 1)

            edge_indices, node_indices = hyper_graph_index[1], hyper_graph_index[0]

            num_edges = edge_indices.max().item() + 1
            num_nodes = node_value.size(2)

            indices = torch.stack([edge_indices, node_indices])
            values = torch.ones(edge_indices.size(0), device=node_value.device)
            adj_matrix = torch.sparse_coo_tensor(
                indices, values, (num_edges, num_nodes), device=node_value.device
            )

            node_value = node_value.permute(2, 0, 1).contiguous()
            edge_features = torch.sparse.mm(
                adj_matrix, node_value.view(num_nodes, -1)
            ).view(num_edges, node_value.size(1), node_value.size(2))

            edge_features = torch.mean(edge_features, dim=1)

            loss_hyper = 0.0
            for k in range(edge_features.size(0)):
                for m in range(edge_features.size(0)):
                    inner_product = torch.sum(
                        edge_features[k, :] * edge_features[m, :],
                        dim=-1,
                        keepdim=True,
                    )
                    norm_q_i = torch.norm(
                        edge_features[k, :], dim=-1, keepdim=True
                    )
                    norm_q_i = torch.clamp(norm_q_i, min=1e-4)
                    norm_q_j = torch.norm(
                        edge_features[m, :], dim=-1, keepdim=True
                    )
                    norm_q_j = torch.clamp(norm_q_j, min=1e-4)
                    alpha = inner_product / (norm_q_i * norm_q_j)

                    distan = torch.norm(
                        edge_embedding[k, :] - edge_embedding[m, :],
                        dim=0,
                        keepdim=True,
                    )

                    loss_item = alpha * distan + (1 - alpha) * (
                        torch.clamp(torch.tensor(4.2) - distan, min=0.0)
                    )
                    loss_hyper = loss_hyper + torch.abs(torch.mean(loss_item))

            loss_hyper = loss_hyper / ((edge_features.size(0) + 1) ** 2)
            loss_hyperedge_all_scale = loss_hyperedge_all_scale + loss_hyper

            node_embedding = self.node_embedding_proj(node_embedding_list[i])
            x_i = torch.index_select(
                node_embedding, dim=0, index=hyper_graph_index[0]
            )
            x_j = torch.index_select(
                edge_features, dim=0, index=hyper_graph_index[1]
            )
            loss_node = abs(torch.mean(x_i - x_j))
            loss_node_all_scale = loss_node_all_scale + loss_node

        loss_hyperedge_all_scale = 0.1 * loss_hyperedge_all_scale

        return loss_hyperedge_all_scale, loss_node_all_scale

    def Laplacian_constraint(self, H, Z):
        A = H @ H.t()
        D = torch.diag(torch.sum(A, dim=1))
        L = D - A

        L_expanded = L.unsqueeze(0).expand(Z.size(0), -1, -1)

        LZ = torch.bmm(L_expanded, Z)

        Z_T_LZ = torch.bmm(Z.transpose(1, 2), LZ)

        loss = torch.einsum("bii->b", Z_T_LZ)
        loss = torch.mean(loss)
        return loss

    def train(self, args, dataloader):
        if self.use_focal_loss:
            mse_func = FocalMSELoss(gamma=args.get("focal_gamma", 2.0))
        else:
            mse_func = nn.MSELoss(reduction="none")

        for epoch in range(1, args["nb_epoch"] + 1):
            logging.info("Training epoch: {}".format(epoch))

            loss_all_batch = 0.0

            (
                hyper_graph_indicies,
                H_list,
                edge_retain_list,
                fused_hypergraph,
                fused_retain_edge,
            ) = self.multi_adpive_hypergraph()

            for d in dataloader:
                ori_window_data = d[0].to(self.device)

                z = self(ori_window_data, hyper_graph_indicies, fused_hypergraph)

                tgt = ori_window_data

                # --- Innovation 5: Focal or standard reconstruction loss ---
                if self.use_focal_loss:
                    rec_loss = mse_func(z, tgt)
                else:
                    rec_loss = mse_func(z, tgt)
                    rec_loss = torch.mean(rec_loss)

                window_ori_x = self.extract_downsample(ori_window_data)
                node_embedding_list, edge_embedding_list = (
                    self.multi_adpive_hypergraph.get_embeddings()
                )
                loss_hyperedge_all_scale, loss_node_all_scale = (
                    self.hyperedge_constraint(
                        window_ori_x,
                        hyper_graph_indicies,
                        node_embedding_list,
                        edge_embedding_list,
                        edge_retain_list,
                    )
                )

                loss_laplacian = 0.0

                for i in range(self.hyper_num):
                    H = torch.mm(
                        node_embedding_list[i], edge_embedding_list[i].t()
                    )
                    H = F.softmax(
                        F.relu(self.multi_adpive_hypergraph.alpha * H)
                    )
                    loss_laplacian = loss_laplacian + self.Laplacian_constraint(
                        H, window_ori_x[i]
                    )

                # --- Innovation 2: Contrastive loss ---
                loss_contrastive = 0.0
                if self.use_contrastive:
                    loss_contrastive = self.contrastive(
                        ori_window_data, self._contrastive_encode
                    )
                    loss_contrastive = args.get("contrastive_weight", 0.1) * loss_contrastive

                loss_sum = (
                    rec_loss
                    + loss_hyperedge_all_scale
                    + loss_node_all_scale
                    + loss_laplacian
                    + loss_contrastive
                )
                loss_all_batch += loss_sum
                self.optimizer.zero_grad()
                loss_sum.backward()
                self.optimizer.step()

            self.scheduler.step()
            logging.info(
                "Epoch: {} finished, loss is {:.4f}".format(
                    epoch, loss_all_batch.cpu().detach().numpy()
                )
            )

    def predict_prob(self, dataloader):
        mse_func = nn.MSELoss(reduction="none")

        with torch.no_grad():
            loss_steps = []
            loss_all_var_steps = []
            z_steps = []
            (
                hyper_graph_indicies,
                H_list,
                edge_retain_list,
                fused_hypergraph,
                fused_retain_edge,
            ) = self.multi_adpive_hypergraph(train=False)

            for d in dataloader:
                ori_window_data = d[0].to(self.device)

                z = self(ori_window_data, hyper_graph_indicies, fused_hypergraph)
                tgt = ori_window_data
                loss = mse_func(z, tgt)

                loss_all_var = loss
                z_all_var = z

                loss = torch.mean(loss, dim=-1)

                loss = loss[:, -1]

                loss_steps.append(loss.detach().cpu().numpy())
                loss_all_var_steps.append(loss_all_var.detach().cpu().numpy())
                z_steps.append(z_all_var.detach().cpu().numpy())
            anomaly_score = np.concatenate(loss_steps)
            loss_all_var_steps = np.concatenate(loss_all_var_steps, axis=0)
            z_steps = np.concatenate(z_steps, axis=0)

        return anomaly_score, loss_all_var_steps, z_steps
