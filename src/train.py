import os
import sys
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from src.utils import evaluate_model, compute_metrics


class EMA:
    """Exponential moving average of model parameters.

    Shadow weights are updated every optimizer step as
    ``shadow = decay * shadow + (1 - decay) * current``. For
    evaluation/validation call :meth:`apply_shadow` to swap the averaged
    weights into the model, then :meth:`restore` to put the live weights back.
    Buffers (e.g. BatchNorm running stats) are intentionally not averaged --
    standard EMA behaviour.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Deep-copy so the shadow is independent of the live parameters.
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
        self._backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self._backup:
                param.copy_(self._backup[name])
        self._backup = {}


class EarlyStopping:
    """Stop training once a monitored quantity stops improving."""

    def __init__(self, patience=15, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _make_scheduler(
    optimizer,
    scheduler_type,
    T_max,
    monitor_mode="min",
    scheduler_factor=0.5,
    scheduler_patience=15,
):
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-7)
    return ReduceLROnPlateau(
        optimizer,
        mode=monitor_mode,
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=1e-7,
    )


def freeze_backbone(model, head_name):
    backbone_name = getattr(model, "_backbone_module_name", None)
    if backbone_name is not None:
        # Expert fusion: freeze ONLY the shared backbone
        # Expert branches + fusion modules + head remain trainable
        backbone = getattr(model, backbone_name)
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        # Regular model: freeze all, unfreeze head only
        for param in model.parameters():
            param.requires_grad = False
        head = getattr(model, head_name)
        for param in head.parameters():
            param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes, scaler=None,
                    ema=None, grad_clip=None, epoch=None, num_epochs=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    if hasattr(criterion, "set_epoch") and epoch is not None and num_epochs is not None:
        criterion.set_epoch(epoch, num_epochs)

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
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(all_labels) if all_labels else 0.0
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
               history, best_monitor_score, save_dir, model_name, monitor_metric="loss",
               monitor_mode="min", scaler=None, ema=None, early_stopping=None,
               grad_clip=None):
    print(f"\nEpoch {epoch}/{num_epochs} [{stage_label}]")

    train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, device, num_classes,
        scaler, ema=ema, grad_clip=grad_clip, epoch=epoch, num_epochs=num_epochs,
    )

    # Evaluate with EMA weights swapped in (if enabled) so the reported
    # metrics and best-checkpoint selection reflect the averaged model. The
    # live weights are restored immediately afterwards.
    if ema is not None:
        ema.apply_shadow(model)
    try:
        val_metrics, _ = evaluate_model(model, val_loader, criterion, device, num_classes)
        test_metrics, _ = evaluate_model(model, test_loader, criterion, device, num_classes)
    finally:
        if ema is not None:
            ema.restore(model)

    monitor_value = val_metrics.get(monitor_metric, val_metrics["loss"])

    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(monitor_value)
    else:
        scheduler.step()

    history["train"].append({"epoch": epoch, **train_metrics})
    history["val"].append({"epoch": epoch, **val_metrics})
    history["test"].append({"epoch": epoch, **test_metrics})

    _log_epoch(stage_label, train_metrics, val_metrics, test_metrics)

    improved = (
        monitor_value > best_monitor_score if monitor_mode == "max"
        else monitor_value < best_monitor_score
    )
    if improved:
        # When EMA is active the averaged weights are currently loaded into
        # the model (restored above), so _save_best writes the EMA snapshot.
        # Without EMA this is just the live model as before.
        if ema is not None:
            ema.apply_shadow(model)
        try:
            best_monitor_score = _save_best(
                model,
                save_dir,
                model_name,
                monitor_value,
                best_monitor_score,
            )
        finally:
            if ema is not None:
                ema.restore(model)
        print(f"  -> Best model saved ({monitor_metric}: {best_monitor_score:.4f})")

    should_stop = False
    if early_stopping is not None and early_stopping.step(monitor_value):
        should_stop = True
        print(
            f"  -> Early stopping triggered (no {monitor_metric} improvement for "
            f"{early_stopping.patience} epochs)"
        )

    return best_monitor_score, should_stop


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
    weight_decay=0.0,
    scheduler_type="plateau",
    monitor_metric="loss",
    monitor_mode="min",
    scheduler_factor=0.5,
    scheduler_patience=15,
    early_stopping=False,
    es_patience=15,
    es_min_delta=0.0,
    ema=False,
    ema_decay=0.999,
    grad_clip=None,
):
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("-inf") if monitor_mode == "max" else float("inf")
    history = {
        "train": [],
        "val": [],
        "test": [],
    }

    # EMA and EarlyStopping are created once so they persist across the freeze
    # and fine-tune stages. Frozen parameters are simply tracked unchanged by
    # the shadow until stage 2.
    ema_state = EMA(model, decay=ema_decay) if ema else None
    stopper = EarlyStopping(patience=es_patience, min_delta=es_min_delta, mode=monitor_mode) if early_stopping else None

    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    print(f"\n{'='*60}")
    if skip_freeze:
        print(f"Training: {model_name} | Epochs: {num_epochs} (full fine-tune)")
    else:
        print(f"Training: {model_name} | Epochs: {num_epochs} (freeze: {freeze_epochs})")
    if scaler is not None:
        print(f"Mixed precision (AMP): enabled")
    if weight_decay > 0:
        print(f"Weight decay: {weight_decay}")
    if ema_state is not None:
        print(f"EMA: enabled (decay={ema_decay})")
    if stopper is not None:
        print(
            f"Early stopping: enabled (monitor={monitor_metric}, mode={monitor_mode}, "
            f"patience={es_patience}, min_delta={es_min_delta})"
        )
    if grad_clip is not None:
        print(f"Gradient clipping: max_norm={grad_clip}")
    print(f"{'='*60}")

    if skip_freeze or freeze_epochs <= 0:
        unfreeze_all(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = _make_scheduler(
            optimizer,
            scheduler_type,
            T_max=num_epochs,
            monitor_mode=monitor_mode,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
        )

        for epoch in range(1, num_epochs + 1):
            best_val_loss, should_stop = _run_epoch(
                epoch, num_epochs, "Full Fine-tune", model, train_loader,
                val_loader, test_loader, criterion, optimizer, device,
                num_classes, scheduler, history, best_val_loss, save_dir,
                model_name, monitor_metric=monitor_metric, monitor_mode=monitor_mode,
                scaler=scaler, ema=ema_state, early_stopping=stopper,
                grad_clip=grad_clip,
            )
            if should_stop:
                break
    else:
        if freeze_epochs > 0:
            print(f"\n--- Stage 1: Frozen backbone ({freeze_epochs} epochs) ---")
            freeze_backbone(model, head_name)
            # For expert fusion: backbone frozen but branches/fusion/head trainable
            # For regular models: only head is trainable
            trainable = [p for p in model.parameters() if p.requires_grad]
            stage1_optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
            stage1_scheduler = _make_scheduler(
                stage1_optimizer,
                scheduler_type,
                T_max=freeze_epochs,
                monitor_mode=monitor_mode,
                scheduler_factor=scheduler_factor,
                scheduler_patience=scheduler_patience,
            )

            for epoch in range(1, freeze_epochs + 1):
                best_val_loss, should_stop = _run_epoch(
                    epoch, num_epochs, "Stage 1 - Frozen", model, train_loader,
                    val_loader, test_loader, criterion, stage1_optimizer, device,
                    num_classes, stage1_scheduler, history, best_val_loss,
                    save_dir, model_name, monitor_metric=monitor_metric,
                    monitor_mode=monitor_mode, scaler=scaler, ema=ema_state,
                    early_stopping=stopper, grad_clip=grad_clip,
                )
                if should_stop:
                    break

        stage2_epochs = num_epochs - freeze_epochs
        if stage2_epochs > 0 and not (stopper is not None and stopper.should_stop):
            print(f"\n--- Stage 2: Fine-tuning all layers ({stage2_epochs} epochs) ---")
            unfreeze_all(model)

            head = getattr(model, head_name)
            head_params_list = list(head.parameters())
            head_param_ids = set(id(p) for p in head_params_list)

            backbone_name = getattr(model, "_backbone_module_name", None)

            if backbone_name is not None:
                # Expert fusion: 3-tier LR (backbone lr/10, branches+fusion lr/5, head lr)
                backbone = getattr(model, backbone_name)
                backbone_params = list(backbone.parameters())
                backbone_param_ids = set(id(p) for p in backbone_params)
                middle_params = [
                    p for p in model.parameters()
                    if id(p) not in backbone_param_ids and id(p) not in head_param_ids
                ]
                stage2_optimizer = torch.optim.AdamW([
                    {"params": backbone_params, "lr": lr / 10, "weight_decay": weight_decay},
                    {"params": middle_params, "lr": lr / 5, "weight_decay": weight_decay},
                    {"params": head_params_list, "lr": lr, "weight_decay": weight_decay},
                ])
            else:
                # Regular model: 2-tier LR (backbone lr/10, head lr)
                backbone_params = [
                    p for p in model.parameters() if id(p) not in head_param_ids
                ]
                stage2_optimizer = torch.optim.AdamW([
                    {"params": backbone_params, "lr": lr / 10, "weight_decay": weight_decay},
                    {"params": head_params_list, "lr": lr, "weight_decay": weight_decay},
                ])
            scheduler = _make_scheduler(
                stage2_optimizer,
                scheduler_type,
                T_max=stage2_epochs,
                monitor_mode=monitor_mode,
                scheduler_factor=scheduler_factor,
                scheduler_patience=scheduler_patience,
            )

            for epoch in range(freeze_epochs + 1, num_epochs + 1):
                best_val_loss, should_stop = _run_epoch(
                    epoch, num_epochs, "Stage 2 - Fine-tune", model, train_loader,
                    val_loader, test_loader, criterion, stage2_optimizer, device,
                    num_classes, scheduler, history, best_val_loss, save_dir,
                    model_name, monitor_metric=monitor_metric, monitor_mode=monitor_mode,
                    scaler=scaler, ema=ema_state, early_stopping=stopper,
                    grad_clip=grad_clip,
                )
                if should_stop:
                    break

    final_metrics = {
        "train": history["train"][-1] if history["train"] else {},
        "val": history["val"][-1] if history["val"] else {},
        "test": history["test"][-1] if history["test"] else {},
        "best_val_loss": best_val_loss,
        "history": history,
    }

    return final_metrics
