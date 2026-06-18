#!/bin/bash

# Test CEF (Competitive Expert Fusion) with shared_base architecture
python src/main.py \
    --output-dir outputs \
    --dataset lymphoma \
    --model cef \
    --backbone1 mobilenetv3_large \
    --expert-mode shared_base \
    --top-k 2 \
    --proj-dim 256 \
    --branch-depth 2 \
    --epochs 100 \
    --loss focal \
    --batch-size 32 \
    --lr 0.0001 \
    --freeze-epochs 5

# Uncomment to test other fusion methods:

# EDF (Expert Disagreement Fusion)
# python src/main.py \
#     --output-dir outputs \
#     --dataset milk10k \
#     --model edf \
#     --backbone1 resnet50 \
#     --expert-mode shared_base \
#     --disagreement-type abs \
#     --proj-dim 256 \
#     --epochs 100 \
#     --loss focal \
#     --batch-size 32 \
#     --lr 0.0001 \
#     --freeze-epochs 5

# CAEF (Confidence Aware Expert Fusion)
# python src/main.py \
#     --output-dir outputs \
#     --dataset milk10k \
#     --model caef \
#     --backbone1 resnet50 \
#     --expert-mode shared_base \
#     --confidence-type fuzzy \
#     --proj-dim 256 \
#     --epochs 100 \
#     --loss focal \
#     --batch-size 32 \
#     --lr 0.0001 \
#     --freeze-epochs 5

# MIEF (Mutual Information Expert Fusion)
# python src/main.py \
#     --output-dir outputs \
#     --dataset milk10k \
#     --model mief \
#     --backbone1 resnet50 \
#     --expert-mode shared_base \
#     --proj-dim 256 \
#     --epochs 100 \
#     --loss focal \
#     --batch-size 32 \
#     --lr 0.0001 \
#     --freeze-epochs 5
