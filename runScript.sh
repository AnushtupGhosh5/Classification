#!/bin/bash

# MILK10K image-only control.  This validates the data recipe before an EDF
# comparison; it keeps the provided train/validation/test folders untouched
# and selects the checkpoint exclusively by validation loss.
# These are intentionally explicit: change loss/lr/img-size/etc. here when
# running an experiment. Command-line values take precedence over dataset
# defaults in src/data/dataset_config.py.
#
python src/main.py \
    --output-dir outputs \
    --run-name milk10k_b0_safe_recipe_v1 \
    --dataset milk10k \
    --model convnext_tiny \
    --epochs 50 \
    --batch-size 32 \
    --img-size 256 \
    --loss ce \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --scheduler cosine \
    --label-smoothing 0.05 \
    --freeze-epochs 0 \
    --augment-style skin_focus \
    --weighted-sampler \
    --sampler-mode sqrt \
    --class-weight-power 0.25 \
    --mixup-alpha 0.1 \
    --cutmix-alpha 0 \
    --mix-prob 0.15 \
    --classifier-dropout 0.35 \
    --tta \
    --amp

# Once the B0 control is complete, replace --model efficientnet_b0 above with
# --model lesion_moe and add: --backbone1 efficientnet_b0 --proj-dim 128
# --branch-depth 1 --routing-mode soft --router-hidden 128
# --router-dropout 0.15 --router-temperature 1.0 --expert-aux-weight 0.10
# --expert-diversity-weight 0 --router-balance-weight 0.01

# Lesion-specialized five-expert MoE (first controlled experiment).
# This is intentionally commented so the active experiment above is unchanged.
# The experts use hierarchical texture/morphology/semantic features plus
# chromatic and boundary cues. Diversity is diagnostic-only in this first run.
# python src/main.py \
#     --output-dir outputs \
#     --run-name lesion_moe_b0_soft_v1 \
#     --dataset isic18 \
#     --model lesion_moe \
#     --backbone1 efficientnet_b0 \
#     --backbone-init-checkpoint outputs/models/isic18/efficientnet_b0_effnetb0_focus256_mixup_v1/efficientnet_b0_effnetb0_focus256_mixup_v1_best.pth \
#     --proj-dim 128 \
#     --branch-depth 1 \
#     --routing-mode soft \
#     --router-hidden 128 \
#     --router-dropout 0.15 \
#     --router-temperature 1.0 \
#     --expert-aux-weight 0.10 \
#     --expert-diversity-weight 0 \
#     --router-balance-weight 0.01 \
#     --epochs 50 \
#     --batch-size 24 \
#     --img-size 256 \
#     --loss ce \
#     --lr 0.0003 \
#     --weight-decay 0.0001 \
#     --scheduler cosine \
#     --label-smoothing 0.05 \
#     --freeze-epochs 5 \
#     --mixup-alpha 0.2 \
#     --cutmix-alpha 0 \
#     --mix-prob 0.15 \
#     --classifier-dropout 0.35 \
#     --tta \
#     --amp

# python src/main.py \
#     --output-dir outputs \
#     --dataset isic19 \
#     --model resnet101 \
#     --scheduler cosine \
#     --proj-dim 224 \
#     --epochs 100 \
#     --loss focal \
#     --batch-size 8 \
#     --lr 0.0001 \
#     --label-smoothing 0.1 \
#     --freeze-epochs 5

# # Uncomment to test other fusion methods:

# EDF (Expert Disagreement Fusion)
# python src/main.py \
#     --output-dir outputs \
#     --dataset isic16 \
#     --model edf \
#     --scheduler cosine \
#     --backbone1 densenet121 \
#     --expert-mode multi_layer \
#     --disagreement-type learnable \
#     --proj-dim 224 \
#     --epochs 100 \
#     --loss ce_focal \
#     --batch-size 32 \
#     --lr 0.0001 \
#     --label-smoothing 0 \
#     --freeze-epochs 5

# python src/main.py \
#     --output-dir outputs \
#     --dataset isic19 \
#     --model edf \
#     --scheduler cosine \
#     --backbone1 resnet101 \
#     --expert-mode multi_layer \
#     --disagreement-type learnable \
#     --proj-dim 224 \
#     --epochs 100 \
#     --loss focal \
#     --batch-size 8 \
#     --lr 0.0001 \
#     --label-smoothing 0.1 \
#     --freeze-epochs 5



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
