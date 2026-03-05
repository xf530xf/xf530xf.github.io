#!/bin/bash
# Ablation study scripts - run each innovation separately
# to measure individual contribution

echo "============================================="
echo "MSHTrans-LLM Enhanced - Ablation Study"
echo "============================================="

DATASET=$1
if [ -z "$DATASET" ]; then
    DATASET="SWaT"
fi

echo "Running ablation study on dataset: $DATASET"

# Baseline: Original MSHTrans (all innovations disabled)
echo "[1/6] Baseline (original MSHTrans)..."
python ./main.py --dataset-id $DATASET --device 0 \
    --model-id MSHTrans_baseline \
    --use-llm False --use-contrastive False \
    --use-channel-attn False --use-patch-embed False \
    --use-focal-loss False

# +LLM only
echo "[2/6] +LLM Feature Extraction..."
python ./main.py --dataset-id $DATASET --device 0 \
    --model-id MSHTrans_LLM_only \
    --use-llm True --use-contrastive False \
    --use-channel-attn False --use-patch-embed False \
    --use-focal-loss False

# +Contrastive only
echo "[3/6] +Contrastive Learning..."
python ./main.py --dataset-id $DATASET --device 0 \
    --model-id MSHTrans_contrastive_only \
    --use-llm False --use-contrastive True \
    --use-channel-attn False --use-patch-embed False \
    --use-focal-loss False

# +Channel Attention only
echo "[4/6] +Channel Attention..."
python ./main.py --dataset-id $DATASET --device 0 \
    --model-id MSHTrans_channel_only \
    --use-llm False --use-contrastive False \
    --use-channel-attn True --use-patch-embed False \
    --use-focal-loss False

# +Patch Embedding only
echo "[5/6] +Patch Embedding..."
python ./main.py --dataset-id $DATASET --device 0 \
    --model-id MSHTrans_patch_only \
    --use-llm False --use-contrastive False \
    --use-channel-attn False --use-patch-embed True \
    --use-focal-loss False

# Full model (all innovations)
echo "[6/6] Full enhanced model..."
python ./main.py --dataset-id $DATASET --device 0 \
    --model-id MSHTrans_LLM_full \
    --use-llm True --use-contrastive True \
    --use-channel-attn True --use-patch-embed True \
    --use-focal-loss True

echo "============================================="
echo "Ablation study complete!"
echo "============================================="
