import os
import sys
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if hasattr(model, "freeze_loaded_baseline"):
            model.freeze_loaded_baseline()
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
                    ema=None, grad_clip=None, epoch=None, num_epochs=None,
                    mixup_alpha=0.0, cutmix_alpha=0.0, mix_prob=0.0,
                    train_scales=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    router_weight_sum = None
    router_entropy_sum = 0.0
    router_sample_count = 0
    correction_scale_sum = 0.0
    correction_scale_count = 0

    if hasattr(criterion, "set_epoch") and epoch is not None and num_epochs is not None:
        criterion.set_epoch(epoch, num_epochs)
    if hasattr(model, "set_training_epoch") and epoch is not None:
        model.set_training_epoch(epoch)

    for images, labels in tqdm(dataloader, desc="  Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        if train_scales:
            scale_index = torch.randint(len(train_scales), (1,)).item()
            target_size = train_scales[scale_index]
            if images.shape[-2:] != (target_size, target_size):
                images = F.interpolate(
                    images,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )

        labels_a, labels_b, lam = labels, labels, 1.0
        if mix_prob > 0 and torch.rand(1).item() < mix_prob and images.size(0) > 1:
            permutation = torch.randperm(images.size(0), device=device)
            labels_b = labels[permutation]
            use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or torch.rand(1).item() < 0.5)
            alpha = cutmix_alpha if use_cutmix else mixup_alpha
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
            if use_cutmix:
                height, width = images.shape[-2:]
                cut_ratio = (1.0 - lam) ** 0.5
                cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
                center_x = torch.randint(width, (1,)).item()
                center_y = torch.randint(height, (1,)).item()
                x1, x2 = max(center_x - cut_w // 2, 0), min(center_x + cut_w // 2, width)
                y1, y2 = max(center_y - cut_h // 2, 0), min(center_y + cut_h // 2, height)
                images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
                lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
            else:
                images = lam * images + (1.0 - lam) * images[permutation]

        def mixed_loss(outputs):
            return lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)

        def expert_auxiliary_loss(result):
            expert_logits = result.get("expert_logits")
            weight = getattr(model, "expert_loss_weight", 0.0)
            if expert_logits is None or weight <= 0:
                return 0.0
            losses = [
                mixed_loss(expert_logits[:, index])
                for index in range(expert_logits.size(1))
            ]
            return weight * torch.stack(losses).mean()

        def correction_auxiliary_loss(result):
            correction_logits = result.get("correction_logits")
            weight = getattr(model, "correction_loss_weight", 0.0)
            if correction_logits is None or weight <= 0:
                return 0.0
            return weight * mixed_loss(correction_logits)

        def router_gain_auxiliary_loss(result):
            baseline_logits = result.get("baseline_logits")
            expert_logits = result.get("expert_logits")
            probabilities = result.get("router_probabilities")
            weight = getattr(model, "router_gain_loss_weight", 0.0)
            if (
                baseline_logits is None or expert_logits is None
                or probabilities is None or weight <= 0
            ):
                return 0.0

            def per_sample_ce(logits, targets):
                if isinstance(criterion, nn.CrossEntropyLoss):
                    return F.cross_entropy(
                        logits,
                        targets,
                        weight=criterion.weight,
                        reduction="none",
                        label_smoothing=criterion.label_smoothing,
                    )
                return F.cross_entropy(logits, targets, reduction="none")

            baseline_loss = (
                lam * per_sample_ce(baseline_logits, labels_a)
                + (1.0 - lam) * per_sample_ce(baseline_logits, labels_b)
            )
            expert_losses = torch.stack([
                lam * per_sample_ce(expert_logits[:, index], labels_a)
                + (1.0 - lam) * per_sample_ce(expert_logits[:, index], labels_b)
                for index in range(expert_logits.size(1))
            ], dim=1)
            gains = baseline_loss.unsqueeze(1) - expert_losses
            temperature = max(
                float(getattr(model, "router_gain_temperature", 0.25)), 1e-4,
            )
            targets = F.softmax(gains.detach() / temperature, dim=1)
            routing_loss = -(
                targets * probabilities.clamp_min(1e-8).log()
            ).sum(dim=1).mean()
            return weight * routing_loss

        def residual_distillation_loss(result):
            baseline_logits = result.get("baseline_logits")
            final_logits = result.get("logits")
            weight = getattr(model, "residual_distill_weight", 0.0)
            if baseline_logits is None or final_logits is None or weight <= 0:
                return 0.0
            temperature = 2.0
            baseline_probabilities = F.softmax(
                baseline_logits.detach() / temperature, dim=1,
            )
            distillation = F.kl_div(
                F.log_softmax(final_logits / temperature, dim=1),
                baseline_probabilities,
                reduction="batchmean",
            ) * (temperature ** 2)
            return weight * distillation

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                result = model(images)
                if isinstance(result, dict):
                    outputs = result["logits"]
                    loss = (
                        mixed_loss(outputs)
                        + expert_auxiliary_loss(result)
                        + correction_auxiliary_loss(result)
                        + router_gain_auxiliary_loss(result)
                        + residual_distillation_loss(result)
                        + result.get("aux_loss", 0.0)
                    )
                else:
                    outputs = result
                    loss = mixed_loss(outputs)
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
                loss = (
                    mixed_loss(outputs)
                    + expert_auxiliary_loss(result)
                    + correction_auxiliary_loss(result)
                    + router_gain_auxiliary_loss(result)
                    + residual_distillation_loss(result)
                    + result.get("aux_loss", 0.0)
                )
            else:
                outputs = result
                loss = mixed_loss(outputs)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        if isinstance(result, dict) and result.get("router_weights") is not None:
            router_weights = result["router_weights"].detach().float()
            batch_weight_sum = router_weights.sum(dim=0).cpu()
            router_weight_sum = (
                batch_weight_sum if router_weight_sum is None
                else router_weight_sum + batch_weight_sum
            )
            router_entropy_sum += float(
                (-(router_weights * router_weights.clamp_min(1e-8).log()).sum(dim=1))
                .sum().cpu()
            )
            router_sample_count += router_weights.size(0)
        if isinstance(result, dict) and result.get("correction_scale") is not None:
            scales = result["correction_scale"].detach().float()
            correction_scale_sum += float(scales.sum().cpu())
            correction_scale_count += scales.numel()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(all_labels) if all_labels else 0.0
    metrics = compute_metrics(all_labels, all_preds, num_classes)
    metrics["loss"] = round(avg_loss, 4)
    if correction_scale_count:
        metrics["correction_scale"] = round(
            correction_scale_sum / correction_scale_count, 6,
        )
    if router_sample_count:
        mean_weights = router_weight_sum / router_sample_count
        for index, weight in enumerate(mean_weights.tolist()):
            metrics[f"router_expert{index}"] = round(weight, 4)
        metrics["router_entropy"] = round(
            router_entropy_sum / router_sample_count, 4
        )

    return metrics


def _save_best(model, save_dir, model_name, val_loss, best_val_loss):
    save_path = os.path.join(save_dir, f"{model_name}_best.pth")
    torch.save(model.state_dict(), save_path)
    return val_loss


def _run_epoch(epoch, num_epochs, stage_label, model, train_loader, val_loader,
               criterion, optimizer, device, num_classes, scheduler,
               history, best_monitor_score, save_dir, model_name, monitor_metric="loss",
               monitor_mode="min", scaler=None, ema=None, early_stopping=None,
               grad_clip=None, mixup_alpha=0.0, cutmix_alpha=0.0, mix_prob=0.0,
               eval_tta=False, train_scales=None):
    print(f"\nEpoch {epoch}/{num_epochs} [{stage_label}]")

    train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, device, num_classes,
        scaler, ema=ema, grad_clip=grad_clip, epoch=epoch, num_epochs=num_epochs,
        mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, mix_prob=mix_prob,
        train_scales=train_scales,
    )

    # Evaluate with EMA weights swapped in (if enabled) so the reported
    # metrics and best-checkpoint selection reflect the averaged model. The
    # live weights are restored immediately afterwards.
    if ema is not None:
        ema.apply_shadow(model)
    try:
        val_metrics, _ = evaluate_model(
            model, val_loader, criterion, device, num_classes, tta=eval_tta,
        )
        # The official test set is intentionally not inspected during training.
        # It is evaluated once after selecting the checkpoint on validation loss.
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

    print(
        f"  Train - Loss: {train_metrics['loss']:.4f} | "
        f"Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}"
    )
    if "router_entropy" in train_metrics:
        router_keys = sorted(
            (key for key in train_metrics if key.startswith("router_expert")),
            key=lambda key: int(key.removeprefix("router_expert")),
        )
        usage = ", ".join(
            f"{train_metrics[key]:.3f}" for key in router_keys
        )
        print(
            f"          Router usage: [{usage}] | "
            f"Entropy: {train_metrics['router_entropy']:.3f}"
        )
    if "correction_scale" in train_metrics:
        print(
            f"          Residual correction scale: "
            f"{train_metrics['correction_scale']:.4f}"
        )
    print(
        f"  Val   - Loss: {val_metrics['loss']:.4f} | "
        f"Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}"
    )
    if "router_entropy" in val_metrics:
        router_keys = sorted(
            (key for key in val_metrics if key.startswith("router_expert")),
            key=lambda key: int(key.removeprefix("router_expert")),
        )
        usage = ", ".join(f"{val_metrics[key]:.3f}" for key in router_keys)
        print(
            f"          Val router: [{usage}] | "
            f"Entropy: {val_metrics['router_entropy']:.3f}"
        )

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
    mixup_alpha=0.0,
    cutmix_alpha=0.0,
    mix_prob=0.0,
    eval_tta=False,
    train_scales=None,
):
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("-inf") if monitor_mode == "max" else float("inf")
    history = {
        "train": [],
        "val": [],
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
    if mix_prob > 0:
        print(f"Batch mixing: p={mix_prob}, mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha}")
    if eval_tta:
        print("Validation TTA: identity + horizontal/vertical/both flips")
    if train_scales:
        print(f"Multi-scale training: {train_scales}")
    print(f"{'='*60}")

    if skip_freeze or freeze_epochs <= 0:
        unfreeze_all(model)
        if hasattr(model, "protect_loaded_baseline"):
            model.protect_loaded_baseline()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=lr,
            weight_decay=weight_decay,
        )
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
                val_loader, criterion, optimizer, device,
                num_classes, scheduler, history, best_val_loss, save_dir,
                model_name, monitor_metric=monitor_metric, monitor_mode=monitor_mode,
                scaler=scaler, ema=ema_state, early_stopping=stopper,
                grad_clip=grad_clip,
                mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
                mix_prob=mix_prob, eval_tta=eval_tta,
                train_scales=train_scales,
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
                    val_loader, criterion, stage1_optimizer, device,
                    num_classes, stage1_scheduler, history, best_val_loss,
                    save_dir, model_name, monitor_metric=monitor_metric,
                    monitor_mode=monitor_mode, scaler=scaler, ema=ema_state,
                    early_stopping=stopper, grad_clip=grad_clip,
                    mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
                    mix_prob=mix_prob, eval_tta=eval_tta,
                    train_scales=train_scales,
                )
                if should_stop:
                    break

        stage2_epochs = num_epochs - freeze_epochs
        if stage2_epochs > 0 and not (stopper is not None and stopper.should_stop):
            if stopper is not None:
                # Constant baseline-preserving warm-up epochs should not spend
                # the correction phase's patience budget. Keep the best loss,
                # but grant stage 2 a fresh consecutive-no-improvement window.
                stopper.counter = 0
                stopper.should_stop = False
                print("Early-stopping patience reset for stage 2")
            protected = bool(getattr(model, "protect_baseline", False))
            stage2_label = (
                "Training complementary corrections (baseline protected)"
                if protected else "Fine-tuning all layers"
            )
            print(f"\n--- Stage 2: {stage2_label} ({stage2_epochs} epochs) ---")
            unfreeze_all(model)
            if hasattr(model, "protect_loaded_baseline"):
                model.protect_loaded_baseline()

            head = getattr(model, head_name)
            head_params_list = list(head.parameters())
            correction_gate = getattr(model, "correction_gate", None)
            if correction_gate is not None:
                head_params_list.extend(correction_gate.parameters())
            head_param_ids = set(id(p) for p in head_params_list)

            backbone_name = getattr(model, "_backbone_module_name", None)

            if backbone_name is not None:
                # Expert fusion: 3-tier LR (backbone lr/10, branches+fusion lr/5, head lr)
                backbone = getattr(model, backbone_name)
                backbone_params = [
                    parameter for parameter in backbone.parameters()
                    if parameter.requires_grad
                ]
                # A classifier loaded with the baseline backbone is part of
                # that protected baseline path and receives the same low LR.
                if bool(getattr(model, "baseline_loaded_flag", False)):
                    for module_name in ("baseline_norm", "baseline_classifier"):
                        module = getattr(model, module_name, None)
                        if module is not None:
                            backbone_params.extend(
                                parameter for parameter in module.parameters()
                                if parameter.requires_grad
                            )
                backbone_param_ids = set(id(p) for p in backbone_params)
                middle_params = [
                    p for p in model.parameters()
                    if p.requires_grad
                    and id(p) not in backbone_param_ids
                    and id(p) not in head_param_ids
                ]
                parameter_groups = []
                if backbone_params:
                    parameter_groups.append({
                        "params": backbone_params, "lr": lr / 10,
                        "weight_decay": weight_decay,
                    })
                if middle_params:
                    parameter_groups.append({
                        "params": middle_params, "lr": lr / 5,
                        "weight_decay": weight_decay,
                    })
                trainable_head_params = [
                    parameter for parameter in head_params_list
                    if parameter.requires_grad
                ]
                if trainable_head_params:
                    parameter_groups.append({
                        "params": trainable_head_params, "lr": lr,
                        "weight_decay": weight_decay,
                    })
                stage2_optimizer = torch.optim.AdamW(parameter_groups)
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
                    epoch, num_epochs, f"Stage 2 - {stage2_label}", model, train_loader,
                    val_loader, criterion, stage2_optimizer, device,
                    num_classes, scheduler, history, best_val_loss, save_dir,
                    model_name, monitor_metric=monitor_metric, monitor_mode=monitor_mode,
                    scaler=scaler, ema=ema_state, early_stopping=stopper,
                    grad_clip=grad_clip,
                    mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
                    mix_prob=mix_prob, eval_tta=eval_tta,
                    train_scales=train_scales,
                )
                if should_stop:
                    break

    final_metrics = {
        "train": history["train"][-1] if history["train"] else {},
        "val": history["val"][-1] if history["val"] else {},
        "best_val_loss": best_val_loss,
        "history": history,
    }

    return final_metrics
