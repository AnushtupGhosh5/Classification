import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from src.utils import evaluate_model, compute_metrics


def freeze_backbone(model, head_name):
    for param in model.parameters():
        param.requires_grad = False
    head = getattr(model, head_name)
    for param in head.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="  Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (outputs.squeeze(dim=1) > 0).long()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
    metrics = compute_metrics(all_labels, all_preds, num_classes)
    metrics["loss"] = round(avg_loss, 4)

    return metrics


def train_model(
    model,
    head_name,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    lr,
    device,
    num_classes,
    num_epochs,
    freeze_epochs,
    model_name,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)

    best_val_f1 = 0.0
    history = {
        "train": [],
        "val": [],
        "test": [],
    }

    print(f"\n{'='*60}")
    print(f"Training: {model_name} | Epochs: {num_epochs} (freeze: {freeze_epochs})")
    print(f"{'='*60}")

    if freeze_epochs > 0:
        print(f"\n--- Stage 1: Frozen backbone ({freeze_epochs} epochs) ---")
        freeze_backbone(model, head_name)
        head = getattr(model, head_name)
        stage1_optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        stage1_scheduler = ReduceLROnPlateau(stage1_optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7)

        for epoch in range(1, freeze_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs} [Stage 1 - Frozen]")

            train_metrics = train_one_epoch(model, train_loader, criterion, stage1_optimizer, device, num_classes)
            val_metrics = evaluate_model(model, val_loader, criterion, device, num_classes)
            test_metrics = evaluate_model(model, test_loader, criterion, device, num_classes)

            stage1_scheduler.step(val_metrics["f1"])

            history["train"].append({"epoch": epoch, **train_metrics})
            history["val"].append({"epoch": epoch, **val_metrics})
            history["test"].append({"epoch": epoch, **test_metrics})

            print(
                f"  Train - Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}"
            )
            print(
                f"  Val   - Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}"
            )
            print(
                f"  Test  - Loss: {test_metrics['loss']:.4f} | "
                f"Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}"
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                save_path = os.path.join(save_dir, f"{model_name}_best.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  -> Best model saved (val F1: {best_val_f1:.4f})")

    stage2_epochs = num_epochs - freeze_epochs
    if stage2_epochs > 0:
        print(f"\n--- Stage 2: Fine-tuning all layers ({stage2_epochs} epochs) ---")
        unfreeze_all(model)

        head = getattr(model, head_name)
        head_params = set(id(p) for p in head.parameters())
        backbone_params = [p for p in model.parameters() if id(p) not in head_params]
        head_params_list = list(head.parameters())

        stage2_optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": lr / 10},
            {"params": head_params_list, "lr": lr},
        ])
        scheduler = ReduceLROnPlateau(stage2_optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7)

        for epoch in range(freeze_epochs + 1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs} [Stage 2 - Fine-tune]")

            train_metrics = train_one_epoch(model, train_loader, criterion, stage2_optimizer, device, num_classes)
            val_metrics = evaluate_model(model, val_loader, criterion, device, num_classes)
            test_metrics = evaluate_model(model, test_loader, criterion, device, num_classes)

            scheduler.step(val_metrics["f1"])

            history["train"].append({"epoch": epoch, **train_metrics})
            history["val"].append({"epoch": epoch, **val_metrics})
            history["test"].append({"epoch": epoch, **test_metrics})

            print(
                f"  Train - Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}"
            )
            print(
                f"  Val   - Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}"
            )
            print(
                f"  Test  - Loss: {test_metrics['loss']:.4f} | "
                f"Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}"
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                save_path = os.path.join(save_dir, f"{model_name}_best.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  -> Best model saved (val F1: {best_val_f1:.4f})")

    final_metrics = {
        "train": history["train"][-1] if history["train"] else {},
        "val": history["val"][-1] if history["val"] else {},
        "test": history["test"][-1] if history["test"] else {},
        "best_val_f1": best_val_f1,
        "history": history,
    }

    return final_metrics
