#!/bin/bash
set -e

# MILK10k class-aware oracle-router-v3 validation experiment. The 73.41% test-accuracy
# ConvNeXt-Tiny run supplies the immutable semantic baseline. Phase 1 trains
# only the four correction experts; phase 2 freezes them and trains only the
# router. V3 preserves all 11 class probabilities and distils soft oracle gains
# instead of collapsing them into hard route labels. The official test is not
# evaluated while this configuration is being developed.
#
python src/main.py \
    --output-dir outputs \
    --run-name milk10k_oracle_router_v3_convnext_256_v1 \
    --dataset milk10k \
    --model oracle_moe \
    --backbone1 convnext_tiny \
    --backbone-init-checkpoint outputs/models/milk10k/convnext_tiny_milk10k_b0_safe_recipe_v1/convnext_tiny_milk10k_b0_safe_recipe_v1_best.pth \
    --proj-dim 128 \
    --branch-depth 1 \
    --routing-mode soft \
    --router-hidden 128 \
    --router-dropout 0.10 \
    --router-temperature 1.0 \
    --router-lr-scale 0.25 \
    --expert-aux-weight 0.20 \
    --expert-diversity-weight 0 \
    --router-balance-weight 0 \
    --expert-pretrain-epochs 10 \
    --oracle-router-version v3 \
    --expert-warmup-epochs 10 \
    --expert-dropout 0 \
    --correction-aux-weight 0.10 \
    --protect-baseline \
    --residual-distill-weight 0.05 \
    --router-gain-weight 0.25 \
    --router-gain-temperature 0.25 \
    --epochs 35 \
    --batch-size 16 \
    --img-size 256 \
    --loss ce \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --scheduler cosine \
    --label-smoothing 0.05 \
    --freeze-epochs 10 \
    --augment-style skin_focus \
    --weighted-sampler \
    --sampler-mode sqrt \
    --class-weight-power 0.25 \
    --mix-prob 0 \
    --classifier-dropout 0.25 \
    --ema \
    --ema-decay 0.999 \
    --tta \
    --validation-only \
    --amp

# Earlier EfficientNet-B0 command retained only as a parameter reference. The
# current lesion_moe implementation uses the semantic backbone plus four
# complementary correction experts regardless of this older run name.
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
