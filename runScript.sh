#!/bin/bash
set -e

# ISIC16 single-backbone, attention-equipped five-expert lesion MoE.
#
# One ConvNeXt-Tiny supplies hierarchical feature maps to the five routed
# experts. Training matches the existing ISIC16 ConvNeXt-Tiny baseline recipe
# (batch 16, 60 epochs, 1e-4 LR, sqrt sampling, skin transforms, and TTA).
# The best checkpoint is selected strictly by minimum validation loss.
python src/main.py \
    --output-dir outputs \
    --run-name isic16_convnext_five_expert_attention_v1 \
    --dataset isic16 \
    --model five_expert_moe \
    --backbone1 convnext_tiny \
    --proj-dim 128 \
    --branch-depth 1 \
    --expert-attention hybrid \
    --routing-mode soft \
    --router-hidden 128 \
    --router-dropout 0.10 \
    --router-temperature 0.70 \
    --router-lr-scale 1.0 \
    --expert-aux-weight 0.15 \
    --expert-diversity-weight 0.001 \
    --router-balance-weight 0.001 \
    --router-gain-weight 0.20 \
    --router-gain-temperature 0.15 \
    --expert-dropout 0.05 \
    --classifier-dropout 0.20 \
    --epochs 60 \
    --batch-size 16 \
    --img-size 256 \
    --augment-style skin \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --scheduler cosine \
    --loss ce \
    --label-smoothing 0.05 \
    --freeze-epochs 0 \
    --weighted-sampler \
    --sampler-mode sqrt \
    --class-weight-power 0.0 \
    --ema \
    --ema-decay 0.995 \
    --early-stopping \
    --es-patience 12 \
    --tta \
    --no-validation-only \
    --amp
