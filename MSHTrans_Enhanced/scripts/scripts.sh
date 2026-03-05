#!/bin/bash
# MSHTrans-LLM Enhanced - Training Scripts
# Run with all innovations enabled (default)

echo "============================================="
echo "MSHTrans-LLM Enhanced Training"
echo "============================================="

# SWaT Dataset - Full enhanced model
python ./main.py --dataset-id SWaT --device 0 \
    --use-llm True --use-contrastive True \
    --use-channel-attn True --use-patch-embed True \
    --use-focal-loss True

# WADI Dataset
python ./main.py --dataset-id WADI --device 0 \
    --use-llm True --use-contrastive True \
    --use-channel-attn True --use-patch-embed True \
    --use-focal-loss True

# SMAP Dataset
python ./main.py --dataset-id SMAP --device 0 \
    --use-llm True --use-contrastive True \
    --use-channel-attn True --use-patch-embed True \
    --use-focal-loss True

# SMD Dataset
python ./main.py --dataset-id SMD --device 0 \
    --use-llm True --use-contrastive True \
    --use-channel-attn True --use-patch-embed True \
    --use-focal-loss True

# MSL Dataset
python ./main.py --dataset-id MSL --device 0 --stride 1 \
    --use-llm True --use-contrastive True \
    --use-channel-attn True --use-patch-embed True \
    --use-focal-loss True

echo "============================================="
echo "All datasets processed successfully!"
echo "============================================="
