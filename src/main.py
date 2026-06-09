import argparse
import os
import csv
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset_config import get_dataset_config
from src.data.dataset import FolderDataset, create_splits
from src.data.preprocess import get_train_transforms, get_val_transforms
from src.models.mobilenetv2 import create_mobilenetv2
from src.models.mobilenetv3 import create_mobilenetv3_small, create_mobilenetv3_large
from src.models.resnet34 import create_resnet34
from src.models.resnet50 import create_resnet50
from src.models.resnet101 import create_resnet101
from src.models.densenet import create_densenet121
from src.models.efficientnet import create_efficientnet_b0, create_efficientnet_b1, create_efficientnet_b2
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
from src.losses import FocalLoss
from src.train import train_model
from src.evaluate import evaluate_all_splits, run_test_evaluation
from src.visualize import plot_training_curves


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
}

ATTENTION_CHOICES = ["none", "se", "cbam", "eca"]

VIT_MODELS = {"vit_b16", "vit_b32"}


def get_data_loaders(data_dir, num_classes, class_names, has_predefined_splits, batch_size, img_size=224, num_workers=4, seed=42):
    train_samples, val_samples, test_samples = create_splits(
        data_dir, num_classes, class_names=class_names,
        has_predefined_splits=has_predefined_splits, seed=seed,
    )

    train_dataset = FolderDataset(train_samples, transform=get_train_transforms(img_size))
    val_dataset = FolderDataset(val_samples, transform=get_val_transforms(img_size))
    test_dataset = FolderDataset(test_samples, transform=get_val_transforms(img_size))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


CSV_FIELDS = [
    "model", "dataset", "attention", "batch_size", "epochs", "lr", "split",
    "accuracy", "precision", "recall", "f1", "specificity", "loss", "macro_auc",
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
                        choices=["all4", "lymphoma", "pbc8", "raabin", "milk10k"],
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
                        help="Disagreement computation type in EDF")
    parser.add_argument("--confidence-type", type=str, default="scalar",
                        choices=["scalar", "channel", "uncertainty", "fuzzy"],
                        help="Confidence estimation type in CAEF")
    parser.add_argument("--proj-dim", type=int, default=256,
                        help="Common projection dimension for expert features")
    parser.add_argument("--expert-mode", type=str, default="multi_backbone",
                        choices=["multi_backbone", "multi_layer"],
                        help="Expert mode: multi_backbone (3 separate backbones) or multi_layer (1 backbone, 3 layers)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="focal",
                        choices=["focal", "ce"],
                        help="Loss function: focal or ce (cross-entropy)")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Epochs to train with frozen backbone (stage 1)")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = get_dataset_config(args.dataset)
    data_dir = config["data_dir"]
    num_classes = config["num_classes"]
    class_names = config["class_names"]
    has_predefined_splits = config["has_predefined_splits"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nDataset: {args.dataset} ({data_dir})")
    print(f"Classes ({num_classes}): {class_names}")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir, num_classes, class_names, has_predefined_splits,
        args.batch_size, args.img_size, args.num_workers, args.seed,
    )

    is_vit = args.model in VIT_MODELS
    is_dual_fusion = args.model == "dual_fusion"
    is_expert_fusion = args.model in ("cef", "edf", "caef", "mief")
    print(f"\nModel: {args.model} | Attention: {args.attention} | Loss: {args.loss} | Batch: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    if is_expert_fusion:
        if args.expert_mode == "multi_layer":
            print(f"Expert mode: multi_layer | Backbone: {args.backbone1}")
        else:
            print(f"Expert mode: multi_backbone | {args.backbone1}(semantic) + {args.backbone2}(frequency) + {args.backbone3}(geometry)")
        if args.model == "cef":
            print(f"Top-K: {args.top_k}")
        elif args.model == "edf":
            print(f"Disagreement type: {args.disagreement_type}")
        elif args.model == "caef":
            print(f"Confidence type: {args.confidence_type}")
        print(f"Projection dim: {args.proj_dim}")
        print(f"Freeze epochs: {args.freeze_epochs} | Stage 2 epochs: {args.epochs - args.freeze_epochs}")
    elif is_dual_fusion:
        print(f"Backbones: {args.backbone1} + {args.backbone2} | Fusion mode: {args.fusion_mode}")
        print(f"Freeze epochs: {args.freeze_epochs} | Stage 2 epochs: {args.epochs - args.freeze_epochs}")
    elif is_vit:
        print(f"ViT model: full fine-tuning from epoch 1 (no freeze stage)")
    else:
        print(f"Freeze epochs: {args.freeze_epochs} | Stage 2 epochs: {args.epochs - args.freeze_epochs}")

    attention_arg = args.attention if args.attention != "none" else None
    if is_expert_fusion:
        expert_kwargs = dict(
            num_classes=num_classes,
            pretrained=True,
            attention=attention_arg,
            backbone1=args.backbone1,
            backbone2=args.backbone2,
            backbone3=args.backbone3,
            proj_dim=args.proj_dim,
            expert_mode=args.expert_mode,
        )
        if args.model == "cef":
            expert_kwargs["top_k"] = args.top_k
        elif args.model == "edf":
            expert_kwargs["disagreement_type"] = args.disagreement_type
        elif args.model == "caef":
            expert_kwargs["confidence_type"] = args.confidence_type
        model, head_name = MODEL_REGISTRY[args.model](**expert_kwargs)
    elif is_dual_fusion:
        model, head_name = create_dual_fusion(
            num_classes=num_classes, pretrained=True, attention=attention_arg,
            backbone1=args.backbone1, backbone2=args.backbone2,
            fusion_mode=args.fusion_mode,
        )
    else:
        model, head_name = MODEL_REGISTRY[args.model](
            num_classes=num_classes, pretrained=True, attention=attention_arg,
        )
    model = model.to(device)

    class_counts = train_loader.dataset.get_class_counts()
    total = sum(class_counts.values())
    class_weights = torch.tensor(
        [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)

    if args.loss == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    counts_str = ", ".join(f"{class_names[i]}={class_counts.get(i, 0)}" for i in range(num_classes))
    print(f"Class weights: {class_weights.cpu().tolist()} ({counts_str})")

    attn_suffix = f"_{args.attention}" if args.attention != "none" else ""
    if is_expert_fusion:
        if args.expert_mode == "multi_layer":
            parts = [args.model, "ml", args.backbone1]
        else:
            parts = [args.model, args.backbone1, args.backbone2, args.backbone3]
        if args.model == "cef":
            parts.append(f"top{args.top_k}")
        elif args.model == "edf":
            parts.append(args.disagreement_type)
        elif args.model == "caef":
            parts.append(args.confidence_type)
        model_label = "_".join(parts)
    elif is_dual_fusion:
        model_label = f"dual_fusion_{args.backbone1}_{args.backbone2}{attn_suffix}_{args.fusion_mode}"
    else:
        model_label = f"{args.model}{attn_suffix}"
    model_save_dir = os.path.join(args.output_dir, "models", args.dataset, model_label)
    results_dir = os.path.join(args.output_dir, "results", args.dataset)
    results_csv = os.path.join(results_dir, "results.csv")

    final_metrics = train_model(
        model=model,
        head_name=head_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        lr=args.lr,
        device=device,
        num_classes=num_classes,
        num_epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        model_name=model_label,
        save_dir=model_save_dir,
        skip_freeze=is_vit,
    )

    plot_training_curves(final_metrics["history"], results_dir, model_label)

    best_model_path = os.path.join(model_save_dir, f"{model_label}_best.pth")
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    final_results = evaluate_all_splits(
        model, train_loader, val_loader, test_loader,
        criterion, device, num_classes,
    )

    test_metrics = run_test_evaluation(
        model, test_loader, class_names, num_classes,
        device, results_dir, model_label, head_name,
    )

    for split in ["train", "validation", "test"]:
        if split in final_results:
            final_results[split]["macro_auc"] = test_metrics.get("macro_auc", 0.0)

    save_results_csv(
        final_results, results_csv,
        model_label if is_dual_fusion else args.model,
        args.dataset, args.attention,
        args.batch_size, args.epochs, args.lr,
    )

    history_path = os.path.join(
        args.output_dir, "results", args.dataset,
        f"{model_label}_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_history.json",
    )
    save_history_json(final_metrics["history"], history_path)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
