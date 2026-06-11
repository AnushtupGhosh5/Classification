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


def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes, scaler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="  Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                result = model(images)
                if isinstance(result, dict):
                    outputs = result["logits"]
                    loss = criterion(outputs, labels) + result.get("aux_loss", 0.0)
                else:
                    outputs = result
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            result = model(images)
            if isinstance(result, dict):
                outputs = result["logits"]
                loss = criterion(outputs, labels) + result.get("aux_loss", 0.0)
            else:
                outputs = result
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
    metrics = compute_metrics(all_labels, all_preds, num_classes)
    metrics["loss"] = round(avg_loss, 4)

    return metrics


def _save_best(model, save_dir, model_name, val_loss, best_val_loss):
    save_path = os.path.join(save_dir, f"{model_name}_best.pth")
    torch.save(model.state_dict(), save_path)
    return val_loss


def _log_epoch(prefix, train_m, val_m, test_m):
    print(
        f"  Train - Loss: {train_m['loss']:.4f} | "
        f"Acc: {train_m['accuracy']:.4f} | F1: {train_m['f1']:.4f}"
    )
    print(
        f"  Val   - Loss: {val_m['loss']:.4f} | "
        f"Acc: {val_m['accuracy']:.4f} | F1: {val_m['f1']:.4f}"
    )
    print(
        f"  Test  - Loss: {test_m['loss']:.4f} | "
        f"Acc: {test_m['accuracy']:.4f} | F1: {test_m['f1']:.4f}"
    )


def _run_epoch(epoch, num_epochs, stage_label, model, train_loader, val_loader,
               test_loader, criterion, optimizer, device, num_classes, scheduler,
               history, best_val_loss, save_dir, model_name, scaler=None):
    print(f"\nEpoch {epoch}/{num_epochs} [{stage_label}]")

    train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes, scaler)
    val_metrics, _ = evaluate_model(model, val_loader, criterion, device, num_classes)
    test_metrics, _ = evaluate_model(model, test_loader, criterion, device, num_classes)

    scheduler.step(val_metrics["loss"])

    history["train"].append({"epoch": epoch, **train_metrics})
    history["val"].append({"epoch": epoch, **val_metrics})
    history["test"].append({"epoch": epoch, **test_metrics})

    _log_epoch(stage_label, train_metrics, val_metrics, test_metrics)

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = _save_best(model, save_dir, model_name, val_metrics["loss"], best_val_loss)
        print(f"  -> Best model saved (val loss: {best_val_loss:.4f})")

    return best_val_loss


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
    skip_freeze=False,
    use_amp=False,
):
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    history = {
        "train": [],
        "val": [],
        "test": [],
    }

    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    print(f"\n{'='*60}")
    if skip_freeze:
        print(f"Training: {model_name} | Epochs: {num_epochs} (full fine-tune)")
    else:
        print(f"Training: {model_name} | Epochs: {num_epochs} (freeze: {freeze_epochs})")
    if scaler is not None:
        print(f"Mixed precision (AMP): enabled")
    print(f"{'='*60}")

    if skip_freeze:
        unfreeze_all(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-7)

        for epoch in range(1, num_epochs + 1):
            best_val_loss = _run_epoch(
                epoch, num_epochs, "Full Fine-tune", model, train_loader,
                val_loader, test_loader, criterion, optimizer, device,
                num_classes, scheduler, history, best_val_loss, save_dir,
                model_name, scaler,
            )
    else:
        if freeze_epochs > 0:
            print(f"\n--- Stage 1: Frozen backbone ({freeze_epochs} epochs) ---")
            freeze_backbone(model, head_name)
            head = getattr(model, head_name)
            stage1_optimizer = torch.optim.Adam(head.parameters(), lr=lr)
            stage1_scheduler = ReduceLROnPlateau(stage1_optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-7)

            for epoch in range(1, freeze_epochs + 1):
                best_val_loss = _run_epoch(
                    epoch, num_epochs, "Stage 1 - Frozen", model, train_loader,
                    val_loader, test_loader, criterion, stage1_optimizer, device,
                    num_classes, stage1_scheduler, history, best_val_loss,
                    save_dir, model_name, scaler,
                )

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
            scheduler = ReduceLROnPlateau(stage2_optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-7)

            for epoch in range(freeze_epochs + 1, num_epochs + 1):
                best_val_loss = _run_epoch(
                    epoch, num_epochs, "Stage 2 - Fine-tune", model, train_loader,
                    val_loader, test_loader, criterion, stage2_optimizer, device,
                    num_classes, scheduler, history, best_val_loss, save_dir,
                    model_name, scaler,
                )

    final_metrics = {
        "train": history["train"][-1] if history["train"] else {},
        "val": history["val"][-1] if history["val"] else {},
        "test": history["test"][-1] if history["test"] else {},
        "best_val_loss": best_val_loss,
        "history": history,
    }

    return final_metrics
