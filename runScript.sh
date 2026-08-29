#!/bin/bash
set -Eeuo pipefail

# Matched PAD-UFES-20 backbone baselines.
#
# DenseNet-121 and ResNet-101 use the same patient-disjoint split and training
# recipe as the existing ConvNeXt-Tiny, EfficientNet-B0/B1, ResNet-50, and
# MobileNetV2 comparisons. The best checkpoint is selected by minimum
# validation loss; the local test split is evaluated only after training.

MODELS=(
    densenet121
    resnet101
)

EPOCHS=70
BATCH_SIZE=8
IMAGE_SIZE=384
LEARNING_RATE=0.0001
SEED=42

for MODEL in "${MODELS[@]}"; do
    RUN_NAME="pad_ufes20_${MODEL}_patient384_baseline_v1"

    echo
    echo "============================================================"
    echo "PAD-UFES-20 matched baseline: ${MODEL}"
    echo "Run name:    ${RUN_NAME}"
    echo "Image size:  ${IMAGE_SIZE}"
    echo "Batch size:  ${BATCH_SIZE}"
    echo "Epochs:      ${EPOCHS}"
    echo "Checkpoint:  minimum validation loss"
    echo "Test:        evaluated after loading the best checkpoint"
    echo "============================================================"

    python src/main.py \
        --output-dir outputs \
        --run-name "${RUN_NAME}" \
        --dataset pad_ufes20 \
        --model "${MODEL}" \
        --attention none \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --img-size "${IMAGE_SIZE}" \
        --augment-style pad_clinical \
        --lr "${LEARNING_RATE}" \
        --weight-decay 0.0001 \
        --scheduler cosine \
        --loss ce \
        --label-smoothing 0.05 \
        --classifier-dropout 0.30 \
        --freeze-epochs 0 \
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
        --seed "${SEED}" \
        --amp
done

echo
echo "============================================================"
echo "PAD-UFES-20 DenseNet-121 and ResNet-101 baselines complete"
echo "Reports: outputs/results/pad_ufes20/<model>_<run-name>/results.md"
echo "============================================================"
