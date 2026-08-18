import argparse
import os
import csv
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import random
import re
from torch.utils.data import DataLoader

from src.data.dataset_config import DATASET_REGISTRY, get_dataset_config
from src.data.dataset import (
    FolderDataset,
    PairedLesionDataset,
    create_milk10k_lesion_splits,
    create_splits,
)
from src.data.preprocess import get_train_transforms, get_val_transforms
from src.models.mobilenetv2 import create_mobilenetv2
from src.models.mobilenetv3 import create_mobilenetv3_small, create_mobilenetv3_large
from src.models.resnet34 import create_resnet34
from src.models.resnet50 import create_resnet50
from src.models.resnet101 import create_resnet101
from src.models.densenet import create_densenet121
from src.models.efficientnet import (
    create_efficientnet_b0,
    create_efficientnet_b1,
    create_efficientnet_b2,
    create_efficientnet_v2_s,
)
from src.models.convnext import create_convnext_tiny
from src.models.squeezenet import create_squeezenet1_0, create_squeezenet1_1
from src.models.vgg import create_vgg16
from src.models.vit import create_vit_b16, create_vit_b32
from src.models.fusion_mobilenet_densenet import create_fusion_mobilenet_densenet
from src.models.dual_fusion import create_dual_fusion
from src.models.backbone_extractor import BACKBONE_CHOICES
from src.models.cef import create_cef
from src.models.edf import create_edf
from src.models.caef import create_caef
from src.models.mief import create_mief
from src.models.moe_edf import create_moe_edf
from src.models.lesion_moe import create_lesion_moe
from src.models.oracle_moe import create_oracle_moe
from src.models.paired_lesion_moe import create_paired_lesion_moe
from src.models.paired_backbone import create_paired_convnext_tiny
from src.models.five_expert_moe import create_five_expert_moe
from src.losses import (
    FocalLoss,
    SEEFNetCEFocalLoss,
    BiTemperedLogisticLoss,
    GeneralizedCrossEntropyLoss,
    SymmetricCrossEntropyLoss,
)
from src.train import train_model
from src.utils import (
    calibrate_router_temperature,
    calibrate_static_expert_fusion,
)
from src.evaluate import (
    evaluate_all_splits,
    run_expert_diagnostics,
    run_oracle_diagnostics,
    run_test_evaluation,
)
from src.visualize import plot_training_curves
from src.gradcam import visualize_gradcam_per_expert


MODEL_REGISTRY = {
    "mobilenetv2": create_mobilenetv2,
    "mobilenetv3_small": create_mobilenetv3_small,
    "mobilenetv3_large": create_mobilenetv3_large,
    "resnet34": create_resnet34,
    "resnet50": create_resnet50,
    "resnet101": create_resnet101,
    "densenet121": create_densenet121,
    "efficientnet_b0": create_efficientnet_b0,
    "efficientnet_b1": create_efficientnet_b1,
    "efficientnet_b2": create_efficientnet_b2,
    "efficientnet_v2_s": create_efficientnet_v2_s,
    "convnext_tiny": create_convnext_tiny,
    "squeezenet1_0": create_squeezenet1_0,
    "squeezenet1_1": create_squeezenet1_1,
    "vgg16": create_vgg16,
    "vit_b16": create_vit_b16,
    "vit_b32": create_vit_b32,
    "mobilenet_densenet_fusion": create_fusion_mobilenet_densenet,
    "cef": create_cef,
    "edf": create_edf,
    "caef": create_caef,
    "mief": create_mief,
    "moe_edf": create_moe_edf,
    "lesion_moe": create_lesion_moe,
    "oracle_moe": create_oracle_moe,
    "paired_lesion_moe": create_paired_lesion_moe,
    "paired_convnext_tiny": create_paired_convnext_tiny,
    "five_expert_moe": create_five_expert_moe,
}

ATTENTION_CHOICES = ["none", "se", "cbam", "eca"]

VIT_MODELS = {"vit_b16", "vit_b32"}


def get_data_loaders(
    data_dir,
    num_classes,
    class_names,
    has_predefined_splits,
    batch_size,
    img_size=224,
    num_workers=4,
    seed=42,
    split_dirs=None,
    train_sample_limit=None,
    train_sampling_strategy="balanced_random",
    validate_images=False,
    fallback_val_from_train=False,
    use_weighted_sampler=False,
    sampler_mode="equal",
    augment_style="balanced",
    split_strategy=None,
    metadata_csv=None,
    images_dir=None,
    val_fraction=0.15,
    test_fraction=0.15,
):
    train_samples, val_samples, test_samples = create_splits(
        data_dir, num_classes, class_names=class_names,
        has_predefined_splits=has_predefined_splits, seed=seed,
        split_dirs=split_dirs,
        train_sample_limit=train_sample_limit,
        train_sampling_strategy=train_sampling_strategy,
        validate_images=validate_images,
        fallback_val_from_train=fallback_val_from_train,
        split_strategy=split_strategy,
        metadata_csv=metadata_csv,
        images_dir=images_dir,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    train_dataset = FolderDataset(
        train_samples,
        transform=get_train_transforms(img_size, augment_style=augment_style),
    )
    train_eval_dataset = FolderDataset(
        train_samples,
        transform=get_val_transforms(img_size, augment_style=augment_style),
    )
    val_dataset = FolderDataset(
        val_samples,
        transform=get_val_transforms(img_size, augment_style=augment_style),
    )
    test_dataset = FolderDataset(
        test_samples,
        transform=get_val_transforms(img_size, augment_style=augment_style),
    )

    # WeightedRandomSampler: gives every class equal draw probability so that
    # the model sees a balanced view of all classes each epoch, replicating what
    # SEEFNet does via offline augmented datasets.
    #
    # sampler_mode='equal'  — weight = 1/count.  Every class is drawn ~max_count
    #   times per epoch. Best for small datasets (e.g. ISIC16 4:1 ratio).
    # sampler_mode='sqrt'   — weight = 1/sqrt(count).  Gentler rebalancing;
    #   epoch size stays close to the original.  Best for large, severely
    #   imbalanced datasets (e.g. HAM10K 58:1 ratio) where 'equal' would
    #   produce an epoch 4–5x larger than the real data.
    if use_weighted_sampler:
        import math
        from torch.utils.data import WeightedRandomSampler
        class_counts = train_dataset.get_class_counts()
        if sampler_mode == "sqrt":
            raw_weights = {c: 1.0 / math.sqrt(n) for c, n in class_counts.items()}
            total_weight = sum(raw_weights.values())
            norm_weights = {c: w / total_weight for c, w in raw_weights.items()}
            sample_weights = [norm_weights[label] for _, label in train_samples]
            # Keep epoch size = original dataset size (samples are redistributed, not added)
            num_samples = len(train_samples)
            mode_desc = "sqrt-weighted"
        else:  # 'equal'
            raw_weights = {c: 1.0 / n for c, n in class_counts.items()}
            sample_weights = [raw_weights[label] for _, label in train_samples]
            # Each class gets ~max_count draws per epoch
            max_count = max(class_counts.values())
            num_samples = num_classes * max_count
            mode_desc = "equal"

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
        )
        print(
            f"  WeightedRandomSampler ({mode_desc}): {num_samples} samples/epoch "
            f"(original: {len(train_samples)})"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

    train_eval_loader = DataLoader(
        train_eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, train_eval_loader, val_loader, test_loader


def get_milk10k_paired_loaders(
    config,
    class_names,
    batch_size,
    img_size,
    num_workers,
    seed,
    use_weighted_sampler,
    sampler_mode,
    augment_style,
    val_fraction,
    test_fraction,
):
    """Load MILK10k as 5,240 lesion pairs with leakage-free splits."""
    train_samples, val_samples, test_samples = create_milk10k_lesion_splits(
        config["paired_images_dir"],
        config["paired_ground_truth_csv"],
        config["paired_metadata_csv"],
        class_names,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    train_dataset = PairedLesionDataset(
        train_samples,
        transform=get_train_transforms(img_size, augment_style=augment_style),
    )
    evaluation_transform = get_val_transforms(
        img_size, augment_style=augment_style,
    )
    train_eval_dataset = PairedLesionDataset(train_samples, evaluation_transform)
    val_dataset = PairedLesionDataset(val_samples, evaluation_transform)
    test_dataset = PairedLesionDataset(test_samples, evaluation_transform)

    if use_weighted_sampler:
        import math
        from torch.utils.data import WeightedRandomSampler
        counts = train_dataset.get_class_counts()
        if sampler_mode == "sqrt":
            class_weights = {
                label: 1.0 / math.sqrt(count) for label, count in counts.items()
            }
            num_samples = len(train_samples)
            mode_description = "sqrt-weighted"
        else:
            class_weights = {label: 1.0 / count for label, count in counts.items()}
            num_samples = len(class_names) * max(counts.values())
            mode_description = "equal"
        sample_weights = [class_weights[sample[2]] for sample in train_samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
        print(
            f"  Paired WeightedRandomSampler ({mode_description}): "
            f"{num_samples} lesions/epoch (original: {len(train_samples)})"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
    common = dict(
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return (
        train_loader,
        DataLoader(train_eval_dataset, **common),
        DataLoader(val_dataset, **common),
        DataLoader(test_dataset, **common),
    )


CSV_FIELDS = [
    "model", "dataset", "attention", "batch_size", "epochs", "lr", "split",
    "accuracy", "precision", "recall", "balanced_accuracy", "malignant_recall",
    "f1", "specificity", "loss", "macro_auc",
]


def save_results_csv(results, filepath, model_name, dataset_name, attention, batch_size, epochs, lr):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    rows = []
    for split in ["train", "validation", "test"]:
        if split in results:
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "attention": attention,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "split": split,
            }
            row.update(results[split])
            rows.append(row)

    file_exists = os.path.exists(filepath)
    if file_exists:
        with open(filepath, newline="") as existing_file:
            reader = csv.DictReader(existing_file)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_fields != CSV_FIELDS:
            # Preserve prior experiments while evolving the metrics schema.
            temp_path = f"{filepath}.schema-update"
            with open(temp_path, "w", newline="") as migrated_file:
                writer = csv.DictWriter(migrated_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_rows)
            os.replace(temp_path, filepath)
            print(f"Updated results CSV schema while preserving {len(existing_rows)} existing rows")
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults appended to {filepath}")


def save_history_json(history, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Image Classification Pipeline")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Dataset to use")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()) + ["dual_fusion"],
                        help="Model architecture")
    parser.add_argument("--attention", type=str, default="none",
                        choices=ATTENTION_CHOICES,
                        help="Attention mechanism: none, se (SEBlock), cbam, eca")
    parser.add_argument("--backbone1", type=str, default="resnet50",
                        choices=BACKBONE_CHOICES,
                        help="Semantic expert backbone (expert fusion models)")
    parser.add_argument("--backbone2", type=str, default="mobilenetv2",
                        choices=BACKBONE_CHOICES,
                        help="Frequency expert backbone (expert fusion models)")
    parser.add_argument("--backbone3", type=str, default="densenet121",
                        choices=BACKBONE_CHOICES,
                        help="Geometry expert backbone (expert fusion models)")
    parser.add_argument("--fusion-mode", type=str, default="both",
                        choices=["pre_fusion", "post_fusion", "both"],
                        help="Where to apply attention in dual_fusion")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Number of experts to select in CEF")
    parser.add_argument("--disagreement-type", type=str, default="abs",
                        choices=["abs", "cosine", "learnable"],
                        help="Pairwise expert disagreement used by EDF and MoE-EDF")
    parser.add_argument("--confidence-type", type=str, default="scalar",
                        choices=["scalar", "channel", "uncertainty", "fuzzy"],
                        help="Confidence estimation type in CAEF")
    parser.add_argument("--proj-dim", type=int, default=256,
                        help="Common projection dimension for expert features")
    parser.add_argument("--branch-depth", type=int, default=2,
                        help="Number of residual blocks per expert branch (shared_base mode)")
    parser.add_argument("--router-hidden", type=int, default=128)
    parser.add_argument("--router-dropout", type=float, default=0.1)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument(
        "--enabled-experts", nargs="+",
        choices=["texture", "morphology", "semantic", "color", "boundary"],
        default=["texture", "morphology", "semantic", "color", "boundary"],
        help="Active branches for five_expert_moe ablation runs",
    )
    parser.add_argument(
        "--calibrate-router-temperature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Post-training soft-router temperature calibration selected "
            "strictly by minimum validation loss"
        ),
    )
    parser.add_argument(
        "--router-temperature-grid", type=float, nargs="+",
        default=(0.25, 0.4, 0.55, 0.7, 0.9, 1.2, 1.6),
        help="Validation-only temperature candidates for soft routing",
    )
    parser.add_argument("--router-lr-scale", type=float, default=1.0,
                        help="Router learning rate as a multiplier of --lr")
    parser.add_argument("--disagreement-scale", type=float, default=1.0)
    parser.add_argument("--load-balance-weight", type=float, default=0.01)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--expert-loss-weight", type=float, default=0.2)
    parser.add_argument("--expert-vote-weight", type=float, default=0.5)
    parser.add_argument("--routing-mode", type=str, default="soft",
                        choices=["soft", "top2", "top1"],
                        help="Sample-level routing mode for lesion_moe")
    parser.add_argument(
        "--lesion-fusion-space", type=str, default="logits",
        choices=["logits", "features"],
        help=(
            "Fuse lesion specialists as class-logit corrections or as "
            "residuals before the one shared semantic classifier"
        ),
    )
    parser.add_argument(
        "--expert-attention", type=str, default="none",
        choices=["none", "hybrid", "se", "cbam", "eca"],
        help=(
            "Attention inside lesion experts; hybrid uses ECA texture, "
            "CBAM morphology, SE color, and spatial boundary attention"
        ),
    )
    parser.add_argument(
        "--isolate-expert-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stop specialist/auxiliary gradients at shared backbone taps; "
            "the backbone remains trainable through the semantic final path"
        ),
    )
    parser.add_argument(
        "--paired-baseline-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For paired_lesion_moe, train/evaluate only its paired semantic "
            "expert as the matched single-backbone control"
        ),
    )
    parser.add_argument(
        "--milk-val-fraction", type=float, default=0.10,
        help="Lesion-level MILK10k validation fraction",
    )
    parser.add_argument(
        "--milk-local-test-fraction", type=float, default=0.20,
        help="Lesion-level public-data holdout fraction (not hidden benchmark)",
    )
    parser.add_argument("--expert-aux-weight", type=float, default=0.15,
                        help="Mean auxiliary expert-classifier loss weight")
    parser.add_argument("--expert-diversity-weight", type=float, default=0.01,
                        help="Pairwise expert decorrelation loss weight")
    parser.add_argument("--router-balance-weight", type=float, default=0.01,
                        help="Batch-level router load-balancing loss weight")
    parser.add_argument("--expert-warmup-epochs", type=int, default=0,
                        help="Epochs of uniform routing and zero MoE correction")
    parser.add_argument("--expert-pretrain-epochs", type=int, default=10,
                        help="Frozen-baseline expert-only epochs for oracle_moe")
    parser.add_argument("--oracle-router-version", type=str, default="v1",
                        choices=["v1", "v2", "v3"],
                        help="Oracle router: v1 feature-only, v2 summary/hard, or v3 class-aware/soft routing")
    parser.add_argument(
        "--static-fusion-calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace the learned sample router at evaluation with a global "
            "baseline-safe expert mixture selected by validation loss"
        ),
    )
    parser.add_argument(
        "--static-fusion-alpha-steps", type=int, default=41,
        help="Number of correction strengths in the validation-loss grid",
    )
    parser.add_argument(
        "--static-fusion-temperatures", type=float, nargs="+",
        default=(0.01, 0.025, 0.05, 0.1, 0.25, 1.0),
        help="Reliability-softmax temperatures tried during calibration",
    )
    parser.add_argument(
        "--static-fusion-optimize-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Jointly refine the four global convex expert coefficients",
    )
    parser.add_argument("--expert-dropout", type=float, default=0.0,
                        help="Probability of masking each expert route during training")
    parser.add_argument("--correction-aux-weight", type=float, default=0.1,
                        help="Auxiliary supervision for the MoE correction head")
    parser.add_argument("--correction-gate-init", type=float, default=0.0,
                        help="Initial residual MoE correction gate before tanh")
    parser.add_argument("--protect-baseline", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Keep a checkpoint-loaded semantic baseline frozen in stage 2")
    parser.add_argument("--correction-max-scale", type=float, default=1.0,
                        help="Maximum absolute multiplier applied to residual corrections")
    parser.add_argument("--correction-ramp-epochs", type=int, default=0,
                        help="Epochs after warm-up used to ramp residual correction strength")
    parser.add_argument("--residual-distill-weight", type=float, default=0.0,
                        help="KL penalty that keeps corrected predictions near the baseline")
    parser.add_argument("--router-gain-weight", type=float, default=0.0,
                        help="Router supervision from per-expert baseline-loss reduction")
    parser.add_argument("--router-gain-temperature", type=float, default=0.25,
                        help="Soft-target temperature for expert correction gains")
    parser.add_argument("--expert-mode", type=str, default="shared_base",
                        choices=["shared_base", "multi_layer"],
                        help="Expert mode: shared_base (1 backbone + lightweight branches) or multi_layer (1 backbone, 3 layers)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-train-evaluation", action="store_true",
        help="Skip the expensive full training-split evaluation pass",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 weight decay for optimizer (set 0 to disable)")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine"],
                        help="LR scheduler: plateau (ReduceLROnPlateau) or cosine (CosineAnnealingLR)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0 to disable, 0.1 typical)")
    parser.add_argument("--loss", type=str, default="focal",
                        choices=["focal", "ce_focal", "ce", "bi_tempered", "gce", "sce"],
                        help="Loss function: focal, ce_focal (SEEFNet CE+decaying focal), ce, bi_tempered, gce, sce")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Epochs to train with frozen backbone (stage 1)")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default="",
                        help="Optional artifact suffix for reproducible experiments")
    parser.add_argument("--init-checkpoint", type=str, default="",
                        help="Optional state-dict checkpoint used to initialize fine-tuning")
    parser.add_argument("--partial-init", action="store_true",
                        help="Allow missing/new model keys when loading --init-checkpoint")
    parser.add_argument("--backbone-init-checkpoint", type=str, default="",
                        help="Initialize backbone weights from a plain or nested fusion checkpoint")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision (FP16) training to reduce GPU memory")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="MixUp beta parameter (0 disables MixUp)")
    parser.add_argument("--cutmix-alpha", type=float, default=0.0,
                        help="CutMix beta parameter (0 disables CutMix)")
    parser.add_argument("--mix-prob", type=float, default=0.0,
                        help="Probability of applying MixUp or CutMix to a batch")
    parser.add_argument("--classifier-dropout", type=float, default=0.2,
                        help="Dropout probability in classifier heads that contain dropout")
    parser.add_argument("--train-scales", type=int, nargs="+", default=None,
                        help="Optional per-batch training sizes, e.g. 224 256 288")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=False,
                        help="Average identity and flip predictions for validation/test")
    parser.add_argument("--calibrate-binary", action=argparse.BooleanOptionalAction, default=False,
                        help="Select binary probability threshold on validation accuracy")
    parser.add_argument("--validation-only", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Defer all official-test evaluation and report validation only")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Load this run's existing minimum-validation-loss checkpoint and evaluate without retraining")
    parser.add_argument(
        "--expert-visualization-only", action="store_true",
        help="With --evaluate-only, load the locked checkpoint and generate only expert Grad-CAM artifacts",
    )
    parser.add_argument("--augment-style", type=str, default=None,
                        choices=["balanced", "seefnet", "skin", "skin_focus", "pad_clinical", "milk_pair"],
                        help="Override the dataset's image augmentation policy")
    parser.add_argument("--weighted-sampler", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Enable/disable the dataset's weighted training sampler")
    parser.add_argument("--sampler-mode", type=str, default=None,
                        choices=["equal", "sqrt"],
                        help="Class redistribution used with --weighted-sampler")
    parser.add_argument("--class-weight-power", type=float, default=None,
                        help="Exponent applied to inverse-frequency loss weights; 0 disables them")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable exponential moving-average evaluation")
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="EMA decay override")
    parser.add_argument(
        "--early-stopping", action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable early stopping on validation loss",
    )
    parser.add_argument(
        "--es-patience", type=int, default=15,
        help="Consecutive validation-loss non-improvements before stopping",
    )
    parser.add_argument(
        "--es-min-delta", type=float, default=0.0,
        help="Minimum validation-loss decrease counted as improvement",
    )
    args = parser.parse_args()

    config = get_dataset_config(args.dataset)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data_dir = config["data_dir"]
    num_classes = config["num_classes"]
    class_names = config["class_names"]
    has_predefined_splits = config["has_predefined_splits"]
    split_dirs = config.get("split_dirs")
    train_sample_limit = config.get("train_sample_limit")
    train_sampling_strategy = config.get("train_sampling_strategy", "balanced_random")
    validate_images = config.get("validate_images", False)
    fallback_val_from_train = config.get("fallback_val_from_train", False)
    split_strategy = config.get("split_strategy")
    metadata_csv = config.get("metadata_csv")
    images_dir = config.get("images_dir")
    val_fraction = config.get("val_fraction", 0.15)
    test_fraction = config.get("test_fraction", 0.15)

    # Dataset-specific training overrides (e.g. ISIC17 enables EMA / early
    # stopping / grad clipping to keep the validation loss well-behaved on its
    # small, noisy validation set). Empty for datasets without overrides, so
    # their behaviour is unchanged.
    overrides = config.get("training_overrides", {})

    supplied_options = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            supplied_options.add(token.split("=", 1)[0])

    def override_default(arg_name, override_key=None):
        override_key = override_key or arg_name
        option_name = f"--{arg_name.replace('_', '-')}"
        negative_option_name = f"--no-{arg_name.replace('_', '-')}"
        arg_value = getattr(args, arg_name)
        if option_name in supplied_options or negative_option_name in supplied_options:
            return arg_value
        if arg_value == parser.get_default(arg_name):
            return overrides.get(override_key, arg_value)
        return arg_value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nDataset: {args.dataset} ({data_dir})")
    print(f"Classes ({num_classes}): {class_names}")
    if train_sample_limit is not None:
        print(f"Train sampling: limit={train_sample_limit}, strategy={train_sampling_strategy}, seed={args.seed}")
    if validate_images:
        print("Image validation: enabled")
    use_weighted_sampler = (
        args.weighted_sampler
        if "--weighted-sampler" in supplied_options or "--no-weighted-sampler" in supplied_options
        else overrides.get("use_weighted_sampler", False)
    )
    sampler_mode = (
        args.sampler_mode
        if "--sampler-mode" in supplied_options
        else overrides.get("sampler_mode", "equal")
    )
    if use_weighted_sampler:
        print(f"Oversampling: WeightedRandomSampler enabled (mode={sampler_mode})")
    augment_style = (
        args.augment_style
        if "--augment-style" in supplied_options
        else overrides.get("augment_style", "balanced")
    )
    print(f"Train augmentation: {augment_style}")
    img_size = override_default("img_size")
    if img_size != args.img_size:
        print(f"Dataset input size override: {img_size}")
    paired_milk_protocol = args.model in (
        "paired_lesion_moe", "paired_convnext_tiny",
    )
    if paired_milk_protocol:
        if args.dataset != "milk10k":
            parser.error("paired MILK10k models are currently defined only for MILK10k")
        if augment_style != "milk_pair":
            print(
                "Paired MILK10k protocol requires one view per modality; "
                f"using milk_pair transforms instead of {augment_style}."
            )
            augment_style = "milk_pair"
        train_loader, train_eval_loader, val_loader, test_loader = (
            get_milk10k_paired_loaders(
                config, class_names, args.batch_size, img_size,
                args.num_workers, args.seed, use_weighted_sampler,
                sampler_mode, augment_style, args.milk_val_fraction,
                args.milk_local_test_fraction,
            )
        )
        print(
            "MILK10k protocol: paired clinical+dermoscopic lesions; "
            "local test is a grouped public-training holdout, not the hidden benchmark."
        )
    else:
        train_loader, train_eval_loader, val_loader, test_loader = get_data_loaders(
            data_dir, num_classes, class_names, has_predefined_splits,
            args.batch_size, img_size, args.num_workers, args.seed,
            split_dirs=split_dirs,
            train_sample_limit=train_sample_limit,
            train_sampling_strategy=train_sampling_strategy,
            validate_images=validate_images,
            fallback_val_from_train=fallback_val_from_train,
            use_weighted_sampler=use_weighted_sampler,
            sampler_mode=sampler_mode,
            augment_style=augment_style,
            split_strategy=split_strategy,
            metadata_csv=metadata_csv,
            images_dir=images_dir,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

    is_vit = args.model in VIT_MODELS
    is_dual_fusion = args.model == "dual_fusion"
    is_expert_fusion = args.model in (
        "cef", "edf", "caef", "mief", "moe_edf", "lesion_moe", "oracle_moe",
        "paired_lesion_moe", "five_expert_moe",
    )
    print(
        f"\nModel: {args.model} | Attention: {args.attention} | "
        f"Batch: {args.batch_size} | Epochs: {args.epochs} | "
        f"Image size: {img_size}"
    )
    if is_expert_fusion:
        if args.model == "five_expert_moe":
            print(
                f"Backbone: {args.backbone1} | Routed complete experts: "
                f"{'/'.join(args.enabled_experts)} | "
                f"Routing: {args.routing_mode} | "
                f"Expert attention: {args.expert_attention}"
            )
        elif args.model in (
            "lesion_moe", "oracle_moe", "paired_lesion_moe",
        ):
            print(
                f"Semantic baseline: {args.backbone1} | Complementary experts: "
                f"texture/morphology/color/boundary | Routing: {args.routing_mode} | "
                f"Fusion space: {args.lesion_fusion_space} | "
                f"Expert attention: {args.expert_attention}"
            )
        elif args.expert_mode == "multi_layer":
            print(f"Expert mode: multi_layer | Backbone: {args.backbone1}")
        else:
            print(f"Expert mode: shared_base | Backbone: {args.backbone1}")
        if args.model == "cef":
            print(f"Top-K: {args.top_k}")
        elif args.model == "edf":
            print(f"Disagreement type: {args.disagreement_type}")
        elif args.model == "caef":
            print(f"Confidence type: {args.confidence_type}")
        elif args.model == "moe_edf":
            print(
                f"Disagreement type: {args.disagreement_type} | "
                f"Router: hidden={args.router_hidden}, "
                f"dropout={args.router_dropout}, "
                f"temperature={args.router_temperature}"
            )
        elif args.model == "five_expert_moe":
            print(
                f"Five-expert routing: expert_dropout={args.expert_dropout} | "
                f"router_gain={args.router_gain_weight}@"
                f"{args.router_gain_temperature}"
            )
        elif args.model in (
            "lesion_moe", "oracle_moe", "paired_lesion_moe",
        ):
            print(
                f"Router: hidden={args.router_hidden}, "
                f"dropout={args.router_dropout}, "
                f"temperature={args.router_temperature}, "
                f"lr_scale={args.router_lr_scale} | "
                f"Aux/diversity/balance={args.expert_aux_weight}/"
                f"{args.expert_diversity_weight}/{args.router_balance_weight}"
            )
            print(
                f"Residual MoE: warmup={args.expert_warmup_epochs} | "
                f"expert_dropout={args.expert_dropout} | "
                f"correction_aux={args.correction_aux_weight} | "
                f"gate_init={args.correction_gate_init} | "
                f"protect_baseline={args.protect_baseline} | "
                f"max_scale={args.correction_max_scale} | "
                f"ramp={args.correction_ramp_epochs} | "
                f"distill={args.residual_distill_weight} | "
                f"router_gain={args.router_gain_weight}@"
                f"{args.router_gain_temperature} | "
                f"oracle_router={args.oracle_router_version}"
            )
        print(f"Projection dim: {args.proj_dim}")
        print(f"Branch depth: {args.branch_depth} residual blocks")
        print(f"Freeze epochs: {args.freeze_epochs} | Stage 2 epochs: {args.epochs - args.freeze_epochs}")
    elif is_dual_fusion:
        print(f"Backbones: {args.backbone1} + {args.backbone2} | Fusion mode: {args.fusion_mode}")
        print(f"Freeze epochs: {args.freeze_epochs} | Stage 2 epochs: {args.epochs - args.freeze_epochs}")
    elif is_vit:
        print(f"ViT model: full fine-tuning from epoch 1 (no freeze stage)")
    else:
        print(f"Freeze epochs: {args.freeze_epochs} | Stage 2 epochs: {args.epochs - args.freeze_epochs}")

    attention_arg = args.attention if args.attention != "none" else None
    use_imagenet_pretrained = not bool(
        args.init_checkpoint or args.backbone_init_checkpoint
    )
    if args.model in (
        "lesion_moe", "oracle_moe", "paired_lesion_moe",
        "five_expert_moe",
    ):
        residual_factory = (
            create_oracle_moe if args.model == "oracle_moe"
            else create_paired_lesion_moe if args.model == "paired_lesion_moe"
            else create_five_expert_moe if args.model == "five_expert_moe"
            else create_lesion_moe
        )
        residual_kwargs = dict(
            num_classes=num_classes,
            pretrained=use_imagenet_pretrained,
            attention=attention_arg,
            backbone1=args.backbone1,
            proj_dim=args.proj_dim,
            branch_depth=args.branch_depth,
            routing_mode=args.routing_mode,
            router_hidden=args.router_hidden,
            router_dropout=args.router_dropout,
            router_temperature=args.router_temperature,
            router_lr_scale=args.router_lr_scale,
            expert_aux_weight=args.expert_aux_weight,
            expert_diversity_weight=args.expert_diversity_weight,
            router_balance_weight=args.router_balance_weight,
            expert_warmup_epochs=args.expert_warmup_epochs,
            expert_dropout=args.expert_dropout,
            correction_aux_weight=args.correction_aux_weight,
            correction_gate_init=args.correction_gate_init,
            protect_baseline=args.protect_baseline,
            correction_max_scale=args.correction_max_scale,
            correction_ramp_epochs=args.correction_ramp_epochs,
            residual_distill_weight=args.residual_distill_weight,
            router_gain_weight=args.router_gain_weight,
            router_gain_temperature=args.router_gain_temperature,
        )
        if args.model == "oracle_moe":
            residual_kwargs["expert_pretrain_epochs"] = args.expert_pretrain_epochs
            residual_kwargs["oracle_router_version"] = args.oracle_router_version
        elif args.model == "lesion_moe":
            residual_kwargs["fusion_space"] = args.lesion_fusion_space
            residual_kwargs["expert_attention"] = args.expert_attention
            residual_kwargs["isolate_expert_backbone"] = (
                args.isolate_expert_backbone
            )
        else:
            residual_kwargs["expert_attention"] = args.expert_attention
            residual_kwargs["classifier_dropout"] = args.classifier_dropout
            if args.model == "paired_lesion_moe":
                residual_kwargs["paired_baseline_only"] = args.paired_baseline_only
            elif args.model == "five_expert_moe":
                residual_kwargs["enabled_experts"] = args.enabled_experts
        model, head_name = residual_factory(**residual_kwargs)
    elif is_expert_fusion:
        expert_kwargs = dict(
            num_classes=num_classes,
            pretrained=use_imagenet_pretrained,
            attention=attention_arg,
            backbone1=args.backbone1,
            backbone2=args.backbone2,
            backbone3=args.backbone3,
            proj_dim=args.proj_dim,
            expert_mode=args.expert_mode,
            branch_depth=args.branch_depth,
        )
        if args.model == "cef":
            expert_kwargs["top_k"] = args.top_k
        elif args.model == "edf":
            expert_kwargs["disagreement_type"] = args.disagreement_type
        elif args.model == "caef":
            expert_kwargs["confidence_type"] = args.confidence_type
        elif args.model == "moe_edf":
            expert_kwargs.update(
                disagreement_type=args.disagreement_type,
                router_hidden=args.router_hidden,
                router_dropout=args.router_dropout,
                router_temperature=args.router_temperature,
                disagreement_scale=args.disagreement_scale,
                load_balance_weight=args.load_balance_weight,
                diversity_weight=args.diversity_weight,
                expert_loss_weight=args.expert_loss_weight,
                expert_vote_weight=args.expert_vote_weight,
            )
        model, head_name = MODEL_REGISTRY[args.model](**expert_kwargs)
    elif is_dual_fusion:
        model, head_name = create_dual_fusion(
            num_classes=num_classes, pretrained=use_imagenet_pretrained, attention=attention_arg,
            backbone1=args.backbone1, backbone2=args.backbone2,
            fusion_mode=args.fusion_mode,
        )
    else:
        model, head_name = MODEL_REGISTRY[args.model](
            num_classes=num_classes, pretrained=use_imagenet_pretrained, attention=attention_arg,
        )
    model = model.to(device)

    if args.init_checkpoint:
        if not os.path.isfile(args.init_checkpoint):
            parser.error(f"--init-checkpoint does not exist: {args.init_checkpoint}")
        initial_state = torch.load(
            args.init_checkpoint,
            map_location=device,
            weights_only=True,
        )
        incompatible = model.load_state_dict(
            initial_state,
            strict=not args.partial_init,
        )
        if args.partial_init:
            print(
                "Partial checkpoint initialization: "
                f"missing={list(incompatible.missing_keys)}, "
                f"unexpected={list(incompatible.unexpected_keys)}"
            )
        print(f"Initialized model from checkpoint: {args.init_checkpoint}")

    if args.backbone_init_checkpoint:
        if not os.path.isfile(args.backbone_init_checkpoint):
            parser.error(
                f"--backbone-init-checkpoint does not exist: "
                f"{args.backbone_init_checkpoint}"
            )
        backbone_name = getattr(model, "_backbone_module_name", None)
        if backbone_name is None:
            # A regular classifier can be warm-started from the backbone
            # embedded in an earlier fusion checkpoint. Its newly created
            # classifier remains untouched and is trained normally.
            target = model
            excluded_prefix = f"{head_name}."
        else:
            backbone_container = getattr(model, backbone_name)
            target = getattr(backbone_container, "backbone", backbone_container)
            excluded_prefix = None
        source_state = torch.load(
            args.backbone_init_checkpoint,
            map_location=device,
            weights_only=True,
        )
        target_state = target.state_dict()
        source_prefixes = (
            "",
            "feature_pyramid.extractor.backbone.",
            "extractor.backbone.",
        )
        compatible = {}
        for source_key, value in source_state.items():
            for prefix in source_prefixes:
                if prefix and not source_key.startswith(prefix):
                    continue
                target_key = source_key[len(prefix):] if prefix else source_key
                if excluded_prefix and target_key.startswith(excluded_prefix):
                    continue
                if (
                    target_key in target_state
                    and target_state[target_key].shape == value.shape
                ):
                    compatible[target_key] = value
                    break
        if not compatible:
            parser.error(
                "--backbone-init-checkpoint contained no compatible backbone weights"
            )
        target.load_state_dict(compatible, strict=False)
        print(
            f"Initialized backbone from {args.backbone_init_checkpoint} "
            f"({len(compatible)}/{len(target_state)} tensors)"
        )
        if hasattr(model, "load_baseline_classifier"):
            classifier_tensors = model.load_baseline_classifier(source_state)
            if classifier_tensors:
                print(
                    "Initialized residual baseline classifier from checkpoint "
                    f"({classifier_tensors} tensors)"
                )
            else:
                print(
                    "Backbone checkpoint had no compatible baseline classifier; "
                    "the residual baseline head will be learned from scratch"
                )

    def restore_protected_baseline():
        """Reapply the exact source baseline after loading a fusion checkpoint.

        EMA and checkpoint serialization can otherwise introduce tiny drift in
        nominally frozen tensors. A protected residual model must retain the
        exact reference classifier for a valid baseline comparison.
        """
        if not (
            args.protect_baseline
            and args.backbone_init_checkpoint
            and hasattr(model, "load_baseline_classifier")
        ):
            return
        target.load_state_dict(compatible, strict=False)
        model.load_baseline_classifier(source_state)

    classifier_dropout = override_default("classifier_dropout")
    head = getattr(model, head_name)
    dropout_layers = [module for module in head.modules() if isinstance(module, nn.Dropout)]
    for module in dropout_layers:
        module.p = classifier_dropout
    if dropout_layers:
        print(f"Classifier dropout: {classifier_dropout}")

    class_counts = train_loader.dataset.get_class_counts()
    weight_counts = overrides.get("class_weight_counts", config.get("original_class_counts", class_counts))
    if isinstance(weight_counts, (list, tuple)):
        weight_counts = {i: count for i, count in enumerate(weight_counts)}
    total = sum(weight_counts.values())
    raw_class_weights = torch.tensor(
        [total / (num_classes * weight_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    )
    class_weight_power = (
        args.class_weight_power
        if "--class-weight-power" in supplied_options
        else overrides.get("class_weight_power", 1.0)
    )
    class_weights = raw_class_weights.pow(class_weight_power).to(device)

    # Label smoothing may be overridden per dataset (e.g. ISIC17 bumps it to
    # cap the overconfidence that drives validation-loss growth).  Similarly
    # the loss function can be overridden (e.g. ISIC17 uses bi_tempered for
    # its bounded-loss noise robustness).
    label_smoothing = override_default("label_smoothing")
    loss_type = override_default("loss")

    if loss_type == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=label_smoothing)
    elif loss_type == "ce_focal":
        criterion = SEEFNetCEFocalLoss(
            alpha=class_weights,
            gamma=2.0,
            initial_focal_weight=overrides.get("initial_focal_weight", 0.9),
            label_smoothing=label_smoothing,
        )
    elif loss_type == "bi_tempered":
        criterion = BiTemperedLogisticLoss(t1=0.8, t2=0.4, label_smoothing=label_smoothing, alpha=class_weights)
    elif loss_type == "gce":
        criterion = GeneralizedCrossEntropyLoss(q=0.7, label_smoothing=label_smoothing, alpha=class_weights)
    elif loss_type == "sce":
        criterion = SymmetricCrossEntropyLoss(alpha=1.0, beta=1.0, label_smoothing=label_smoothing, alpha_weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    counts_str = ", ".join(f"{class_names[i]}={class_counts.get(i, 0)}" for i in range(num_classes))
    weight_counts_str = ", ".join(f"{class_names[i]}={weight_counts.get(i, 0)}" for i in range(num_classes))
    print(f"Train counts: {counts_str}")
    print(
        f"Class weights: {class_weights.cpu().tolist()} "
        f"(from: {weight_counts_str}, power={class_weight_power})"
    )
    if overrides:
        active = [k for k, v in overrides.items() if v]
        print(f"Training overrides active: {', '.join(active)} (label_smoothing={label_smoothing})")

    attn_suffix = f"_{args.attention}" if args.attention != "none" else ""
    if args.model in (
        "lesion_moe", "oracle_moe", "paired_lesion_moe",
        "five_expert_moe",
    ):
        model_label = f"{args.model}_{args.backbone1}_{args.routing_mode}"
    elif is_expert_fusion:
        if args.expert_mode == "multi_layer":
            parts = [args.model, "ml", args.backbone1]
        else:
            parts = [args.model, "sb", args.backbone1]
        if args.model == "cef":
            parts.append(f"top{args.top_k}")
        elif args.model == "edf":
            parts.append(args.disagreement_type)
        elif args.model == "caef":
            parts.append(args.confidence_type)
        elif args.model == "moe_edf":
            parts.extend(["soft", args.disagreement_type])
        model_label = "_".join(parts)
    elif is_dual_fusion:
        model_label = f"dual_fusion_{args.backbone1}_{args.backbone2}{attn_suffix}_{args.fusion_mode}"
    else:
        model_label = f"{args.model}{attn_suffix}"
    if args.run_name:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
            parser.error("--run-name may contain only letters, numbers, dot, underscore, and hyphen")
        model_label = f"{model_label}_{args.run_name}"
    model_save_dir = os.path.join(args.output_dir, "models", args.dataset, model_label)
    results_dir = os.path.join(args.output_dir, "results", args.dataset)
    results_csv = os.path.join(results_dir, "results.csv")

    train_lr = override_default("lr")
    train_weight_decay = override_default("weight_decay")
    train_scheduler = override_default("scheduler")
    train_freeze_epochs = override_default("freeze_epochs")
    mixup_alpha = override_default("mixup_alpha")
    cutmix_alpha = override_default("cutmix_alpha")
    mix_prob = override_default("mix_prob")
    eval_tta = override_default("tta")
    calibrate_binary = override_default("calibrate_binary")
    early_stopping = override_default("early_stopping")
    es_patience = override_default("es_patience")
    es_min_delta = override_default("es_min_delta")
    if not 0.0 <= mix_prob <= 1.0:
        parser.error("--mix-prob must be between 0 and 1")
    if mix_prob > 0 and mixup_alpha <= 0 and cutmix_alpha <= 0:
        parser.error("--mix-prob requires --mixup-alpha or --cutmix-alpha to be positive")
    if not 0.0 <= classifier_dropout < 1.0:
        parser.error("--classifier-dropout must be in [0, 1)")
    train_scales = args.train_scales
    if train_scales is not None and any(scale <= 0 for scale in train_scales):
        parser.error("--train-scales values must be positive")
    if paired_milk_protocol and (mix_prob > 0 or train_scales):
        parser.error(
            "paired MILK10k models currently require --mix-prob 0 and no "
            "--train-scales so both modalities remain paired"
        )
    if not 0 < args.milk_val_fraction < 1:
        parser.error("--milk-val-fraction must be between 0 and 1")
    if not 0 < args.milk_local_test_fraction < 1:
        parser.error("--milk-local-test-fraction must be between 0 and 1")
    if args.milk_val_fraction + args.milk_local_test_fraction >= 1:
        parser.error("MILK10k validation + local-test fractions must be below 1")
    if args.router_hidden <= 0:
        parser.error("--router-hidden must be positive")
    if not 0.0 <= args.router_dropout < 1.0:
        parser.error("--router-dropout must be in [0, 1)")
    if args.router_temperature <= 0:
        parser.error("--router-temperature must be positive")
    if any(value <= 0 for value in args.router_temperature_grid):
        parser.error("--router-temperature-grid values must be positive")
    if args.router_lr_scale <= 0:
        parser.error("--router-lr-scale must be positive")
    if es_patience < 1:
        parser.error("--es-patience must be positive")
    if es_min_delta < 0:
        parser.error("--es-min-delta must be non-negative")
    if any(value < 0 for value in (
        args.load_balance_weight,
        args.diversity_weight,
        args.expert_loss_weight,
        args.expert_vote_weight,
        args.expert_aux_weight,
        args.expert_diversity_weight,
        args.router_balance_weight,
        args.correction_aux_weight,
        args.router_gain_weight,
        args.residual_distill_weight,
    )):
        parser.error("MoE auxiliary and vote weights must be non-negative")
    if args.expert_warmup_epochs < 0:
        parser.error("--expert-warmup-epochs must be non-negative")
    if args.expert_pretrain_epochs < 1:
        parser.error("--expert-pretrain-epochs must be positive")
    if not 0.0 <= args.expert_dropout < 1.0:
        parser.error("--expert-dropout must be in [0, 1)")
    if args.router_gain_temperature <= 0:
        parser.error("--router-gain-temperature must be positive")
    if args.correction_max_scale < 0:
        parser.error("--correction-max-scale must be non-negative")
    if args.correction_ramp_epochs < 0:
        parser.error("--correction-ramp-epochs must be non-negative")
    if (
        args.model == "oracle_moe"
        and train_freeze_epochs != args.expert_pretrain_epochs
    ):
        parser.error(
            "oracle_moe requires --freeze-epochs to equal "
            "--expert-pretrain-epochs"
        )
    ema_enabled = (
        args.ema if "--ema" in supplied_options or "--no-ema" in supplied_options
        else overrides.get("ema", False)
    )
    ema_decay = (
        args.ema_decay if "--ema-decay" in supplied_options
        else overrides.get("ema_decay", 0.999)
    )
    if not 0.0 <= ema_decay < 1.0:
        parser.error("--ema-decay must be in [0, 1)")
    # Research protocol invariant: checkpointing and early stopping are always
    # determined by minimum validation loss. Do not make this dataset-tunable.
    monitor_metric = "loss"
    monitor_mode = "min"
    scheduler_factor = overrides.get("scheduler_factor", 0.5)
    scheduler_patience = overrides.get("scheduler_patience", 15)
    print(
        f"Effective training: lr={train_lr} | weight_decay={train_weight_decay} | "
        f"scheduler={train_scheduler} | loss={loss_type} | "
        f"label_smoothing={label_smoothing} | image_size={img_size} | "
        f"monitor={monitor_metric}/{monitor_mode} | freeze_epochs={train_freeze_epochs}"
    )

    if args.evaluate_only:
        best_model_path = os.path.join(
            model_save_dir, f"{model_label}_best.pth",
        )
        if not os.path.isfile(best_model_path):
            parser.error(
                "--evaluate-only requires the existing minimum-validation-loss "
                f"checkpoint: {best_model_path}"
            )
        print(
            "\nEvaluation-only protocol active: training is skipped and the "
            "locked minimum-validation-loss checkpoint will be evaluated."
        )
        final_metrics = None
    else:
        final_metrics = train_model(
            model=model,
            head_name=head_name,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            lr=train_lr,
            device=device,
            num_classes=num_classes,
            num_epochs=args.epochs,
            freeze_epochs=train_freeze_epochs,
            model_name=model_label,
            save_dir=model_save_dir,
            skip_freeze=is_vit,
            use_amp=args.amp,
            weight_decay=train_weight_decay,
            scheduler_type=train_scheduler,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            early_stopping=early_stopping,
            es_patience=es_patience,
            es_min_delta=es_min_delta,
            ema=ema_enabled,
            ema_decay=ema_decay,
            grad_clip=overrides.get("grad_clip", None),
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mix_prob=mix_prob,
            eval_tta=eval_tta,
            train_scales=train_scales,
        )

        plot_training_curves(final_metrics["history"], results_dir, model_label)

    evaluation_test_loader = None if args.validation_only else test_loader
    if args.validation_only:
        print(
            "\nValidation-only protocol active: the official test split will "
            "not be evaluated or used for diagnostics."
        )

    router_best_path = os.path.join(
        model_save_dir, f"{model_label}_router_best.pth",
    )
    if (
        bool(getattr(model, "oracle_protocol", False))
        and os.path.exists(router_best_path)
    ):
        print(
            f"\nLoading best trained-router candidate for oracle diagnostics "
            f"from {router_best_path}"
        )
        model.load_state_dict(torch.load(
            router_best_path, map_location=device, weights_only=True,
        ))
        restore_protected_baseline()
        run_oracle_diagnostics(
            # The official test split is never used for oracle/expert
            # diagnostics. It is reserved for the locked learned model below.
            model, val_loader, None, criterion, device,
            num_classes, results_dir, model_label, tta=eval_tta,
        )

    best_model_path = os.path.join(model_save_dir, f"{model_label}_best.pth")
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        restore_protected_baseline()

    if args.expert_visualization_only:
        if not args.evaluate_only:
            parser.error("--expert-visualization-only requires --evaluate-only")
        if evaluation_test_loader is None:
            parser.error("Expert visualization requires an available test loader")
        if not hasattr(model, "expert_names"):
            parser.error("The selected model does not expose expert branches")
        visualize_gradcam_per_expert(
            model, evaluation_test_loader, device, results_dir, model_label,
            num_images=6, class_names=class_names,
        )
        print("\nExpert visualization complete; metrics were not re-evaluated.")
        return

    if args.calibrate_router_temperature:
        calibration = calibrate_router_temperature(
            model, val_loader, criterion, device, num_classes,
            temperatures=args.router_temperature_grid, tta=eval_tta,
        )
        os.makedirs(results_dir, exist_ok=True)
        calibration_path = os.path.join(
            results_dir, f"{model_label}_router_temperature_calibration.json",
        )
        with open(calibration_path, "w") as file:
            json.dump(calibration, file, indent=2)
        print("\nVALIDATION-LOSS ROUTER TEMPERATURE CALIBRATION:")
        for candidate in calibration["candidates"]:
            print(
                f"  T={candidate['temperature']:.3g} | "
                f"loss={candidate['loss']:.4f} | "
                f"Acc={candidate['accuracy']:.4f} | "
                f"F1={candidate['f1']:.4f} | "
                f"BAcc={candidate['balanced_accuracy']:.4f}"
            )
        print(
            "  Selected T="
            f"{calibration['selected_temperature']:.3g} by minimum "
            "validation loss"
        )
        print(f"Saved router-temperature calibration: {calibration_path}")

    if args.static_fusion_calibration:
        calibration = calibrate_static_expert_fusion(
            model, val_loader, criterion, device, num_classes, tta=eval_tta,
            alpha_steps=args.static_fusion_alpha_steps,
            temperatures=args.static_fusion_temperatures,
            optimize_weights=args.static_fusion_optimize_weights,
        )
        os.makedirs(results_dir, exist_ok=True)
        calibration_path = os.path.join(
            results_dir, f"{model_label}_static_fusion_calibration.json",
        )
        with open(calibration_path, "w") as file:
            json.dump(calibration, file, indent=2)
        baseline = calibration["baseline"]
        calibrated = calibration["calibrated_static_fusion"]
        weights = ", ".join(
            f"{name}={value:.3f}"
            for name, value in calibration["expert_weights"].items()
        )
        print("\nVALIDATION-LOSS STATIC FUSION CALIBRATION:")
        print(
            f"  baseline   loss={baseline['loss']:.4f} | "
            f"Acc={baseline['accuracy']:.4f} | F1={baseline['f1']:.4f}"
        )
        print(
            f"  calibrated loss={calibrated['loss']:.4f} | "
            f"Acc={calibrated['accuracy']:.4f} | F1={calibrated['f1']:.4f}"
        )
        print(
            f"  alpha={calibration['alpha']:.3f} | "
            f"temperature={calibration['temperature']:.3g} | "
            f"optimizer={calibration['fusion_optimizer']} | {weights}"
        )
        print(f"Saved static-fusion calibration: {calibration_path}")

    final_results = evaluate_all_splits(
        model,
        None if args.skip_train_evaluation else train_eval_loader,
        val_loader, evaluation_test_loader,
        criterion, device, num_classes, tta=eval_tta,
    )

    if not bool(getattr(model, "oracle_protocol", False)):
        run_expert_diagnostics(
            model, val_loader, evaluation_test_loader, device, num_classes,
            class_names, results_dir, model_label, tta=eval_tta,
        )

    if not args.validation_only:
        test_metrics = run_test_evaluation(
            model, test_loader, class_names, num_classes,
            device, results_dir, model_label, head_name,
            model_name=args.model, is_expert_fusion=is_expert_fusion,
            tta=eval_tta,
            val_loader=val_loader,
            calibrate_binary=calibrate_binary,
        )
        if "test" in final_results:
            final_results["test"].update(test_metrics)

    save_results_csv(
        final_results, results_csv,
        model_label if is_dual_fusion else args.model,
        args.dataset, args.attention,
        args.batch_size, args.epochs, train_lr,
    )

    if final_metrics is not None:
        history_path = os.path.join(
            args.output_dir, "results", args.dataset,
            f"{model_label}_bs{args.batch_size}_ep{args.epochs}_lr{train_lr}_history.json",
        )
        save_history_json(final_metrics["history"], history_path)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
