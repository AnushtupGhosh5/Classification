#!/bin/bash

# Test CEF (Competitive Expert Fusion) with shared_base architecture
python src/main.py \
    --output-dir outputs \
    --dataset isic19 \
    --model efficientnet_b1 \
    --scheduler cosine \
    --proj-dim 224 \
    --epochs 100 \
    --loss focal \
    --batch-size 16 \
    --lr 0.0001 \
    --label-smoothing 0.1 \
    --freeze-epochs 5

python src/main.py \
    --output-dir outputs \
    --dataset isic19 \
    --model resnet101 \
    --scheduler cosine \
    --proj-dim 224 \
    --epochs 100 \
    --loss focal \
    --batch-size 16 \
    --lr 0.0001 \
    --label-smoothing 0.1 \
    --freeze-epochs 5

# Uncomment to test other fusion methods:

# EDF (Expert Disagreement Fusion)
python src/main.py \
    --output-dir outputs \
    --dataset isic19 \
    --model edf \
    --scheduler cosine \
    --backbone1 efficientnet_b1 \
    --expert-mode multi_layer \
    --disagreement-type learnable \
    --proj-dim 224 \
    --epochs 100 \
    --loss focal \
    --batch-size 16 \
    --lr 0.0001 \
    --label-smoothing 0.1 \
    --freeze-epochs 5

    python src/main.py \
    --output-dir outputs \
    --dataset isic19 \
    --model edf \
    --scheduler cosine \
    --backbone1 resnet101 \
    --expert-mode multi_layer \
    --disagreement-type learnable \
    --proj-dim 224 \
    --epochs 100 \
    --loss focal \
    --batch-size 16 \
    --lr 0.0001 \
    --label-smoothing 0.1 \
    --freeze-epochs 5


# CAEF (Confidence Aware Expert Fusion)
# python src/main.py \
#     --output-dir outputs \
#     --dataset lymphoma \
#     --model caef \
#     --backbone1 efficientnet_b0 \
#     --expert-mode shared_base \
#     --confidence-type fuzzy \
#     --proj-dim 256 \
#     --epochs 100 \
#     --loss focal \
#     --batch-size 32 \
#     --lr 0.0001 \
#     --freeze-epochs 5

# # MIEF (Mutual Information Expert Fusion)
# python src/main.py \
#     --output-dir outputs \
#     --dataset isic16 \
#     --scheduler cosine \
#     --model mief \
#     --backbone1 efficientnet_b1 \
#     --expert-mode shared_base \
#     --proj-dim 256 \
#     --epochs 100 \
#     --loss focal \
#     --batch-size 16 \
#     --lr 0.0001 \
#     --label-smoothing 0.1 \
#     --freeze-epochs 5
