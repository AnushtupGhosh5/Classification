#!/bin/bash

# Baseline runs for the two priority datasets.
# These are intentionally explicit: change loss/lr/img-size/etc. here when
# running an experiment. Command-line values take precedence over dataset
# defaults in src/data/dataset_config.py.
python src/main.py \
    --output-dir outputs \
    --dataset isic16 \
    --model efficientnet_b0 \
    --epochs 80 \
    --batch-size 32 \
    --img-size 256 \
    --loss ce \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --scheduler cosine \
    --label-smoothing 0.05 \
    --freeze-epochs 0 \
    --amp

# python src/main.py \
#     --output-dir outputs \
#     --dataset isic18 \
#     --model efficientnet_b0 \
#     --epochs 80 \
#     --batch-size 32 \
#     --img-size 256 \
#     --loss ce \
#     --lr 0.0003 \
#     --weight-decay 0.0001 \
#     --scheduler cosine \
#     --label-smoothing 0.05 \
#     --freeze-epochs 0 \
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
