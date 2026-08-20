#!/bin/bash
set -e

# Independent repeat of the final MILK10k paired-lesion hybrid-attention MoE.
#
# This keeps the original lesion-level split (split seed 42) while changing
# only training randomness (seed 43). It therefore measures repeatability on
# exactly the same train/validation/local-test lesions. The existing seed-42
# checkpoint is preserved under its original run name.

REFERENCE_CHECKPOINT="outputs/models/milk10k/paired_lesion_moe_convnext_tiny_soft_milk10k_paired_five_expert_v1/paired_lesion_moe_convnext_tiny_soft_milk10k_paired_five_expert_v1_best.pth"

if [[ ! -f "$REFERENCE_CHECKPOINT" ]]; then
    echo "Missing original MILK10k hybrid checkpoint: $REFERENCE_CHECKPOINT"
    exit 1
fi

echo
echo "============================================================"
echo "MILK10k final paired-lesion hybrid-attention MoE repeat"
echo "Split seed: 42 | Training seed: 43"
echo "Experts: texture morphology semantic color boundary"
echo "============================================================"

python src/main.py \
    --output-dir outputs \
    --run-name milk10k_paired_five_expert_hybrid_repeat_seed43_v1 \
    --dataset milk10k \
    --model paired_lesion_moe \
    --backbone1 convnext_tiny \
    --enabled-experts texture morphology semantic color boundary \
    --attention none \
    --expert-attention hybrid \
    --routing-mode soft \
    --proj-dim 128 \
    --branch-depth 1 \
    --router-hidden 128 \
    --router-dropout 0.10 \
    --router-temperature 0.70 \
    --router-lr-scale 1.0 \
    --expert-aux-weight 0.15 \
    --expert-diversity-weight 0.005 \
    --router-balance-weight 0.002 \
    --expert-dropout 0.0 \
    --epochs 50 \
    --batch-size 8 \
    --img-size 256 \
    --augment-style milk_pair \
    --milk-val-fraction 0.10 \
    --milk-local-test-fraction 0.20 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --scheduler cosine \
    --loss ce \
    --label-smoothing 0.05 \
    --classifier-dropout 0.25 \
    --freeze-epochs 0 \
    --weighted-sampler \
    --sampler-mode sqrt \
    --class-weight-power 0.25 \
    --ema \
    --ema-decay 0.999 \
    --early-stopping \
    --es-patience 15 \
    --no-tta \
    --no-validation-only \
    --milk-split-seed 42 \
    --seed 43 \
    --amp
