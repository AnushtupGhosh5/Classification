#!/bin/bash
set -e

# PAD-UFES-20 expert-removal ablation.
# The existing full five-expert run is the reference. Every entry below changes
# only the active expert set; split, initialization and training recipe remain
# identical. Checkpoints are always selected by minimum validation loss.
ABLATION_NAMES=(
    no_texture
    no_boundary
    no_early_pair
    no_morphology
    no_color
    no_semantic
)
ABLATION_EXPERTS=(
    "morphology semantic color boundary"
    "texture morphology semantic color"
    "morphology semantic color"
    "texture semantic color boundary"
    "texture morphology semantic boundary"
    "texture morphology color boundary"
)

for INDEX in "${!ABLATION_NAMES[@]}"; do
    NAME="${ABLATION_NAMES[$INDEX]}"
    read -r -a ACTIVE_EXPERTS <<< "${ABLATION_EXPERTS[$INDEX]}"

    echo
    echo "============================================================"
    echo "PAD-UFES-20 expert ablation: ${NAME}"
    echo "Active experts: ${ACTIVE_EXPERTS[*]}"
    echo "============================================================"

    python src/main.py \
        --output-dir outputs \
        --run-name "pad_ufes20_five_expert_ablation_${NAME}_v1" \
        --dataset pad_ufes20 \
        --model five_expert_moe \
        --backbone1 convnext_tiny \
        --backbone-init-checkpoint outputs/models/pad_ufes20/convnext_tiny_pad_ufes20_convnext_tiny_patient384_v2/convnext_tiny_pad_ufes20_convnext_tiny_patient384_v2_best.pth \
        --enabled-experts "${ACTIVE_EXPERTS[@]}" \
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
        --expert-diversity-weight 0.001 \
        --router-balance-weight 0.001 \
        --expert-dropout 0.05 \
        --epochs 70 \
        --batch-size 8 \
        --img-size 384 \
        --augment-style pad_clinical \
        --lr 0.0001 \
        --weight-decay 0.0001 \
        --scheduler cosine \
        --loss focal \
        --label-smoothing 0.05 \
        --classifier-dropout 0.30 \
        --freeze-epochs 3 \
        --no-weighted-sampler \
        --sampler-mode sqrt \
        --class-weight-power 0.25 \
        --mixup-alpha 0.20 \
        --mix-prob 0.25 \
        --ema \
        --ema-decay 0.999 \
        --early-stopping \
        --es-patience 15 \
        --tta \
        --no-validation-only \
        --seed 42 \
        --amp
done
