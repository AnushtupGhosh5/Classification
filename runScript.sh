#!/bin/bash

python src/main.py \
    --output-dir outputs \
    --dataset all4 \
    --model efficientnet_b0 \
    --attention none \
    --epochs 100 \
    --loss focal \
    --batch-size 32 \
    --lr 0.0001 \
    --freeze-epochs 5
