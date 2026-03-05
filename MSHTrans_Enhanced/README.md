# MSHTrans-LLM Enhanced

**Multi-Scale Hypergraph Transformer with LLM Integration for Temporal Anomaly Detection**

This is an enhanced version of [MSHTrans](https://github.com/chenzl23/MSHTrans) (KDD 2025) that integrates Large Language Model (LLM) capabilities with several state-of-the-art innovations to improve time series anomaly detection performance.

## Innovations

### 1. 🧠 LLM-Enhanced Feature Extraction (`networks/llm_module.py`)
- **Frozen GPT-2-style Transformer backbone** reprogrammed for time series via learnable cross-attention
- **Patch Reprogramming Layer**: Maps time series patches into the LLM's embedding space using learnable source prototypes
- **Gated Fusion**: Adaptively combines LLM features with original hypergraph features
- References: [GPT4TS (NeurIPS 2023)](https://arxiv.org/abs/2302.11939), [Time-LLM (ICLR 2024)](https://arxiv.org/abs/2310.01728)

### 2. 🔄 Dual-Branch Contrastive Learning (`networks/contrastive.py`)
- **Temporal Contrastive Loss**: Encourages temporally adjacent representations to be similar
- **Instance Contrastive Loss**: Distinguishes different samples using augmented views (jitter + mask)
- Improves representation discriminability for anomaly detection
- References: [TS2Vec (AAAI 2022)](https://arxiv.org/abs/2106.10466), [DCdetector (KDD 2023)](https://arxiv.org/abs/2306.10347)

### 3. 📊 Channel-wise Attention (`networks/channel_attention.py`)
- **Channel Tokenization**: Each feature channel becomes a token with temporal embedding
- **Cross-Channel Self-Attention**: Captures inter-variate correlations via multi-head attention
- **Adaptive Channel Gating**: Learns to weight channel importance dynamically
- References: [iTransformer (ICLR 2024)](https://arxiv.org/abs/2310.06625), [Crossformer (ICLR 2023)](https://arxiv.org/abs/2209.05249)

### 4. 🔍 Multi-Scale Patch Embedding (`networks/patch_embedding.py`)
- **Multi-resolution Patching**: Captures local patterns at different temporal granularities (4, 8, 16 time steps)
- **Learnable Scale Aggregation**: Attention-based combination of multi-scale representations
- **Complementary to Hypergraph**: Provides local pattern information that complements the global hypergraph structure
- References: [PatchTST (ICLR 2023)](https://arxiv.org/abs/2211.14730), [Pathformer (ICLR 2024)](https://arxiv.org/abs/2402.05956)

### 5. ⚡ Focal Reconstruction Loss (`networks/focal_loss.py`)
- **Focal MSE Loss**: Applies higher weight to samples with larger reconstruction errors (likely anomalies)
- **Association Discrepancy**: Measures deviation between expected and actual temporal correlations
- **Combined Loss**: Focal reconstruction + association discrepancy + contrastive learning
- References: [Focal Loss (ICCV 2017)](https://arxiv.org/abs/1708.02002), [Anomaly Transformer (ICLR 2022)](https://arxiv.org/abs/2110.02642)

## Architecture Overview

```
Input Time Series (B, T, C)
         │
         ├──────────────────────────────────────┐
         │                                      │
    ┌────▼────┐   ┌──────────────┐   ┌─────────▼─────────┐
    │ Channel │   │ Multi-Scale  │   │  LLM Feature      │
    │Attention│   │   Patch      │   │  Extractor         │
    │ Module  │   │  Embedding   │   │  (Frozen GPT-2)   │
    └────┬────┘   └──────┬───────┘   └─────────┬─────────┘
         │               │                     │
         └───────┬───────┘                     │
                 │                             │
         ┌───────▼───────┐              ┌──────▼──────┐
         │  Enhanced     │              │  LLM Fusion │
         │  Input        │◄─────────────│    Gate     │
         │  Features     │              └─────────────┘
         └───────┬───────┘
                 │
    ┌────────────▼────────────────┐
    │   MSHTrans Encoder          │
    │   (Multi-Scale Hypergraph   │
    │    Transformer)             │
    └────────────┬────────────────┘
                 │
    ┌────────────▼────────────────┐
    │   MSHTrans Decoder          │
    │   (Hypergraph Conv +        │
    │    Series Decomposition)    │
    └────────────┬────────────────┘
                 │
    ┌────────────▼────────────────┐
    │   Loss Functions            │
    │   • Focal MSE Loss          │
    │   • Hyperedge Constraint    │
    │   • Laplacian Constraint    │
    │   • Contrastive Loss        │
    │   • Association Discrepancy │
    └─────────────────────────────┘
```

## Code Structure

```
MSHTrans_Enhanced/
├── common/                    # Data loading and evaluation (from original MSHTrans)
│   ├── data_preprocess.py     # Data preprocessing and windowing
│   ├── dataloader.py          # PyTorch data loaders
│   ├── exp.py                 # Experiment utilities
│   ├── utils.py               # General utilities
│   └── evaluation/            # Evaluation metrics and thresholding
├── networks/                  # Neural network modules
│   ├── MSHTrans.py            # ★ Enhanced main model with all innovations
│   ├── MAHLayer.py            # Multi-Adaptive Hypergraph layer
│   ├── HyperGraphConv.py      # Hypergraph convolution
│   ├── HyperGraphUp.py        # Hypergraph upsampling/fusion
│   ├── Layers.py              # Core layers (decomposition, fusion, etc.)
│   ├── fft.py                 # Fourier analysis layers
│   ├── llm_module.py          # ★ NEW: LLM feature extraction
│   ├── contrastive.py         # ★ NEW: Contrastive learning
│   ├── channel_attention.py   # ★ NEW: Channel-wise attention
│   ├── patch_embedding.py     # ★ NEW: Multi-scale patch embedding
│   └── focal_loss.py          # ★ NEW: Focal reconstruction loss
├── scripts/
│   ├── scripts.sh             # Main training scripts
│   └── ablation.sh            # Ablation study scripts
├── tests/
│   └── test_modules.py        # Unit tests for new modules
├── main.py                    # ★ Enhanced main entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Requirements

```bash
pip install -r requirements.txt
```

## Data Download

- Download data from Google Drive: [Download Link](https://drive.google.com/file/d/1bnFMU0jhJYRnhFwuWkZgRAq_DxecZZBQ/view?usp=sharing)
- Unzip and move data to data folder (defined in parameter `--data-root`)

## Quick Start

### Full Enhanced Model
```bash
bash ./scripts/scripts.sh
```

### Single Dataset
```bash
python ./main.py --dataset-id SWaT --device 0
```

### Ablation Study
```bash
bash ./scripts/ablation.sh SWaT
```

### Disable Specific Innovations
```bash
# Run without LLM module
python ./main.py --dataset-id SWaT --use-llm False

# Run without contrastive learning
python ./main.py --dataset-id SWaT --use-contrastive False

# Run original MSHTrans baseline
python ./main.py --dataset-id SWaT \
    --use-llm False --use-contrastive False \
    --use-channel-attn False --use-patch-embed False \
    --use-focal-loss False
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--d-llm` | 128 | LLM backbone hidden dimension |
| `--llm-layers` | 3 | Number of frozen transformer layers |
| `--llm-prototypes` | 64 | Number of learnable source prototypes |
| `--contrastive-weight` | 0.1 | Weight for contrastive loss |
| `--contrastive-temp` | 0.07 | Contrastive temperature |
| `--channel-d-model` | 64 | Channel attention dimension |
| `--channel-layers` | 2 | Number of channel attention layers |
| `--patch-sizes` | [4,8,16] | Multi-scale patch sizes |
| `--focal-gamma` | 2.0 | Focal loss focusing parameter |
| `--assoc-weight` | 0.1 | Association discrepancy weight |

## Running Tests

```bash
cd MSHTrans_Enhanced
python -m pytest tests/ -v
```

## Expected Performance Improvements

The innovations are designed to improve anomaly detection metrics through:

1. **LLM Feature Extraction**: Leverages pre-trained pattern recognition for richer temporal features → improved F1 and recall
2. **Contrastive Learning**: More discriminative representations → better separation of normal and anomalous patterns
3. **Channel Attention**: Better inter-variate correlation modeling → improved detection of multi-channel anomalies
4. **Multi-Scale Patching**: Captures fine-grained local patterns → reduced detection delay
5. **Focal Loss**: Focuses on hard-to-reconstruct patterns → improved precision and reduced false alarms

## Citation

Original MSHTrans:
```bibtex
@inproceedings{chen2025mshtrans,
  title={MSHTrans: Multi-Scale Hypergraph Transformer with Time-Series Decomposition for Temporal Anomaly Detection},
  author={Chen, Z. and Wu, Z. and Cheung, W. K. and Dai, H-N. and Choi, B. and Liu, J.},
  booktitle={SIGKDD},
  year={2025}
}
```

## License

This project builds upon MSHTrans which is based on Ada-MSHyper (Apache License 2.0).
