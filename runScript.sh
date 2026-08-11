#!/bin/bash
set -e

# Research constraint: exactly one shared CNN backbone. The four lightweight
# lesion specialists consume early/intermediate features and the input image;
# they are not additional pretrained classifiers.
python src/main.py \
    --output-dir outputs \
    --run-name milk10k_single_backbone_experts_v4 \
    --dataset milk10k \
    --model lesion_moe \
    --backbone1 convnext_tiny \
    --backbone-init-checkpoint outputs/models/milk10k/convnext_tiny_milk10k_b0_safe_recipe_v1/convnext_tiny_milk10k_b0_safe_recipe_v1_best.pth \
    --protect-baseline \
    --lesion-fusion-space features \
    --epochs 50 \
    --batch-size 16 \
    --img-size 256 \
    --proj-dim 128 \
    --branch-depth 1 \
    --routing-mode soft \
    --router-hidden 96 \
    --router-dropout 0.10 \
    --router-temperature 1.0 \
    --expert-aux-weight 0.15 \
    --expert-diversity-weight 0.001 \
    --router-balance-weight 0.005 \
    --expert-warmup-epochs 5 \
    --correction-aux-weight 0.10 \
    --correction-gate-init 0.0 \
    --correction-max-scale 0.50 \
    --correction-ramp-epochs 10 \
    --lr 0.0002 \
    --weight-decay 0.0001 \
    --scheduler plateau \
    --loss ce \
    --label-smoothing 0.05 \
    --freeze-epochs 5 \
    --weighted-sampler \
    --sampler-mode sqrt \
    --class-weight-power 0.25 \
    --early-stopping \
    --es-patience 15 \
    --tta \
    --validation-only \
    --skip-train-evaluation \
    --amp
