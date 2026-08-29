import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        if (tn + fp) > 0:
            specificities.append(tn / (tn + fp))
        else:
            specificities.append(0.0)
    specificity = np.mean(specificities)

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "balanced_accuracy": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
    }
    if num_classes == 2:
        malignant_total = cm[1, :].sum()
        metrics["malignant_recall"] = round(
            float(cm[1, 1] / malignant_total) if malignant_total else 0.0, 4,
        )
    return metrics


def compute_per_class_metrics(y_true, y_pred, num_classes, class_names=None):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    if class_names is None:
        class_names = [f"class{i}" for i in range(num_classes)]

    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            "class": name,
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        })
    return rows


@torch.no_grad()
def _forward_with_tta_details(model, images, tta=False):
    """Average logits and compact expert diagnostics over evaluation views."""
    batch_size = images.size(0)
    paired_input = bool(getattr(model, "paired_input", False))
    crop_views = images.size(1) if images.dim() == 5 and not paired_input else 1
    base_images = (
        images.flatten(0, 1)
        if images.dim() == 5 and not paired_input else images
    )
    views = [base_images]
    if tta:
        views.extend([
            torch.flip(base_images, dims=[-1]),
            torch.flip(base_images, dims=[-2]),
            torch.flip(base_images, dims=[-2, -1]),
        ])

    logits = []
    aux_losses = []
    detail_views = {}
    for view in views:
        result = model(view)
        if isinstance(result, dict):
            logits.append(result["logits"])
            aux_losses.append(result.get("aux_loss", 0.0))
            for key in (
                "router_weights", "router_probabilities", "router_active",
                "expert_logits", "expert_embeddings", "expert_disagreement",
                "baseline_logits", "correction_logits", "correction_scale",
            ):
                value = result.get(key)
                if value is not None:
                    detail_views.setdefault(key, []).append(value)
        else:
            logits.append(result)
    output = torch.stack(logits).mean(dim=0)
    if crop_views > 1:
        output = output.reshape(batch_size, crop_views, -1).mean(dim=1)
    aux_loss = sum(aux_losses) / len(aux_losses) if aux_losses else 0.0
    details = {}
    for key, values in detail_views.items():
        value = torch.stack(values).mean(dim=0)
        if crop_views > 1:
            value = value.reshape(batch_size, crop_views, *value.shape[1:]).mean(dim=1)
        details[key] = value
    return output, aux_loss, details


@torch.no_grad()
def forward_with_tta(model, images, tta=False):
    """Average logits over lossless dermoscopy orientation transforms."""
    output, aux_loss, _ = _forward_with_tta_details(model, images, tta=tta)
    return output, aux_loss


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device, num_classes, tta=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    total_classification_loss = 0.0
    total_classification_weight = 0.0
    total_aux_loss = 0.0
    total_samples = 0
    router_weights = []
    correction_scales = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs, aux_loss, details = _forward_with_tta_details(
            model, images, tta=tta,
        )
        classification_loss = criterion(outputs, labels)
        if isinstance(criterion, nn.CrossEntropyLoss) and criterion.weight is not None:
            # Weighted CE with reduction="mean" is normalized by the sum of
            # target-class weights, not batch size. Accumulating by batch size
            # makes validation loss depend on batch boundaries and batch size.
            classification_weight = float(
                criterion.weight[labels].sum().detach().cpu()
            )
        else:
            classification_weight = float(images.size(0))
        total_classification_loss += (
            float(classification_loss.detach().cpu()) * classification_weight
        )
        total_classification_weight += classification_weight
        aux_value = (
            float(aux_loss.detach().cpu())
            if torch.is_tensor(aux_loss) else float(aux_loss)
        )
        total_aux_loss += aux_value * images.size(0)
        total_samples += images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(outputs.cpu().numpy())
        if details.get("router_weights") is not None:
            router_weights.append(details["router_weights"].cpu())
        if details.get("correction_scale") is not None:
            correction_scales.append(details["correction_scale"].cpu())

    avg_classification_loss = (
        total_classification_loss / total_classification_weight
        if total_classification_weight > 0 else 0.0
    )
    avg_aux_loss = total_aux_loss / total_samples if total_samples > 0 else 0.0
    avg_loss = avg_classification_loss + avg_aux_loss
    metrics = compute_metrics(all_labels, all_preds, num_classes)
    metrics["loss"] = round(avg_loss, 4)
    if correction_scales:
        metrics["correction_scale"] = round(
            float(torch.cat(correction_scales).mean()), 6,
        )
    if router_weights:
        weights = torch.cat(router_weights, dim=0)
        for index, value in enumerate(weights.mean(dim=0).tolist()):
            metrics[f"router_expert{index}"] = round(value, 4)
        entropy = -(
            weights * weights.clamp_min(1e-8).log()
        ).sum(dim=1).mean()
        metrics["router_entropy"] = round(float(entropy), 4)

    return metrics, np.array(all_logits)


@torch.no_grad()
def calibrate_router_temperature(
    model, dataloader, criterion, device, num_classes, temperatures,
    tta=False,
):
    """Select a soft-router temperature using validation loss only.

    This is post-training calibration: model parameters and architecture are
    untouched. The selected value remains installed on ``model.router`` for
    the subsequent locked-split evaluation.
    """
    router = getattr(model, "router", None)
    if router is None or not hasattr(router, "temperature"):
        raise ValueError(
            "Router-temperature calibration requires a model exposing "
            "model.router.temperature"
        )
    if getattr(router, "routing_mode", "soft") != "soft":
        raise ValueError(
            "Router-temperature calibration is only valid for soft routing"
        )

    candidates = tuple(float(value) for value in temperatures)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("All router-temperature candidates must be positive")
    # Avoid evaluating duplicate temperatures while preserving the declared
    # order in the saved protocol record.
    candidates = tuple(dict.fromkeys(candidates))
    original = float(router.temperature)
    reports = []
    best = None
    try:
        for temperature in candidates:
            router.temperature = temperature
            metrics, _ = evaluate_model(
                model, dataloader, criterion, device, num_classes, tta=tta,
            )
            report = {
                "temperature": temperature,
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "router_entropy": metrics.get("router_entropy"),
            }
            reports.append(report)
            # The research selection rule is strictly minimum validation
            # loss. Ties prefer the least change from the trained setting.
            key = (
                report["loss"],
                abs(np.log(temperature / original)),
                temperature,
            )
            if best is None or key < best["key"]:
                best = {"key": key, "report": report}
    except Exception:
        router.temperature = original
        raise

    selected = float(best["report"]["temperature"])
    router.temperature = selected
    return {
        "selection_split": "validation",
        "selection_metric": "minimum_validation_loss",
        "architecture_or_weights_changed": False,
        "original_temperature": original,
        "selected_temperature": selected,
        "selected_validation_metrics": best["report"],
        "candidates": reports,
    }


@torch.no_grad()
def compute_expert_diagnostics(model, dataloader, device, num_classes,
                               class_names=None, tta=False):
    """Collect standalone expert and sample-level routing diagnostics."""
    model.eval()
    weights_batches = []
    probability_batches = []
    active_batches = []
    logits_batches = []
    embedding_batches = []
    baseline_batches = []
    correction_batches = []
    correction_scale_batches = []
    labels_batches = []

    for images, labels in dataloader:
        images = images.to(device)
        _, _, details = _forward_with_tta_details(model, images, tta=tta)
        if details.get("router_weights") is None:
            return None
        weights_batches.append(details["router_weights"].cpu())
        probabilities = details.get("router_probabilities")
        if probabilities is not None:
            probability_batches.append(probabilities.cpu())
        active = details.get("router_active")
        if active is not None:
            active_batches.append(active.cpu())
        logits_batches.append(details["expert_logits"].cpu())
        embedding_batches.append(details["expert_embeddings"].cpu())
        if details.get("baseline_logits") is not None:
            baseline_batches.append(details["baseline_logits"].cpu())
        if details.get("correction_logits") is not None:
            correction_batches.append(details["correction_logits"].cpu())
        if details.get("correction_scale") is not None:
            correction_scale_batches.append(details["correction_scale"].cpu())
        labels_batches.append(labels.cpu())

    if not weights_batches:
        return None

    weights = torch.cat(weights_batches)
    raw_probabilities = (
        torch.cat(probability_batches) if probability_batches else None
    )
    active = torch.cat(active_batches) if active_batches else (weights > 0).float()
    expert_logits = torch.cat(logits_batches)
    embeddings = F.normalize(torch.cat(embedding_batches), dim=-1)
    labels = torch.cat(labels_batches)
    num_experts = weights.shape[1]
    names = list(getattr(model, "expert_names", ()))
    if len(names) != num_experts:
        names = [f"expert{index}" for index in range(num_experts)]
    if class_names is None:
        class_names = [f"class{index}" for index in range(num_classes)]

    argmax = weights.argmax(dim=1)
    routing = {}
    for index, name in enumerate(names):
        routing[name] = {
            "mean_weight": round(float(weights[:, index].mean()), 6),
            "std_weight": round(float(weights[:, index].std(unbiased=False)), 6),
            "argmax_frequency": round(float((argmax == index).float().mean()), 6),
            "active_frequency": round(float(active[:, index].mean()), 6),
        }
    entropy = -(
        weights * weights.clamp_min(1e-8).log()
    ).sum(dim=1).mean()

    standalone = {}
    labels_list = labels.numpy()
    for index, name in enumerate(names):
        predictions = expert_logits[:, index].argmax(dim=1).numpy()
        standalone[name] = compute_metrics(labels_list, predictions, num_classes)

    baseline_metrics = None
    if baseline_batches:
        baseline_logits = torch.cat(baseline_batches)
        baseline_metrics = compute_metrics(
            labels_list, baseline_logits.argmax(dim=1).numpy(), num_classes,
        )
    correction_metrics = None
    if correction_batches:
        correction_logits = torch.cat(correction_batches)
        correction_metrics = compute_metrics(
            labels_list, correction_logits.argmax(dim=1).numpy(), num_classes,
        )

    similarity = torch.einsum("bed,bfd->bef", embeddings, embeddings).mean(dim=0)
    similarity_rows = {
        names[row]: {
            names[column]: round(float(similarity[row, column]), 6)
            for column in range(num_experts)
        }
        for row in range(num_experts)
    }

    class_conditional = {}
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        if not mask.any():
            continue
        means = weights[mask].mean(dim=0)
        class_conditional[class_name] = {
            names[index]: round(float(means[index]), 6)
            for index in range(num_experts)
        }

    report = {
        "num_samples": int(labels.numel()),
        "routing": routing,
        "router_entropy": round(float(entropy), 6),
        "maximum_entropy": round(float(torch.tensor(float(num_experts)).log()), 6),
        "standalone_experts": standalone,
        "mean_cosine_similarity": similarity_rows,
        "class_conditional_routing": class_conditional,
    }
    if raw_probabilities is not None:
        raw_argmax = raw_probabilities.argmax(dim=1)
        report["raw_router_entropy"] = round(float(-(
            raw_probabilities
            * raw_probabilities.clamp_min(1e-8).log()
        ).sum(dim=1).mean()), 6)
        report["raw_routing"] = {
            name: {
                "mean_probability": round(
                    float(raw_probabilities[:, index].mean()), 6,
                ),
                "std_probability": round(
                    float(raw_probabilities[:, index].std(unbiased=False)), 6,
                ),
                "argmax_frequency": round(
                    float((raw_argmax == index).float().mean()), 6,
                ),
            }
            for index, name in enumerate(names)
        }
    if baseline_metrics is not None:
        report["baseline_path"] = baseline_metrics
    if correction_metrics is not None:
        report["correction_path"] = correction_metrics
    if correction_scale_batches:
        scales = torch.cat(correction_scale_batches)
        report["correction_scale"] = round(float(scales.mean()), 6)
        report["correction_scale_std"] = round(
            float(scales.std(unbiased=False)), 6,
        )
    return report


@torch.no_grad()
def compute_oracle_diagnostics(model, dataloader, criterion, device,
                               num_classes, tta=False):
    """Compare baseline, uniform, learned, best-expert, and oracle routing.

    The oracle chooses the lowest-loss candidate independently for each
    validation sample. It is an upper-bound diagnostic, never a deployable
    result and never used to update model parameters.
    """
    model.eval()
    baseline_batches = []
    uniform_batches = []
    learned_batches = []
    expert_batches = []
    route_batches = []
    label_batches = []

    for images, labels in dataloader:
        images = images.to(device)
        learned, _, details = _forward_with_tta_details(
            model, images, tta=tta,
        )
        if (
            details.get("baseline_logits") is None
            or details.get("expert_logits") is None
            or details.get("router_probabilities") is None
        ):
            return None
        baseline_batches.append(details["baseline_logits"].cpu())
        uniform_batches.append(details["correction_logits"].cpu())
        learned_batches.append(learned.cpu())
        expert_batches.append(details["expert_logits"].cpu())
        route_batches.append(details["router_probabilities"].cpu())
        label_batches.append(labels.cpu())

    if not label_batches:
        return None

    baseline = torch.cat(baseline_batches)
    uniform = torch.cat(uniform_batches)
    learned = torch.cat(learned_batches)
    experts = torch.cat(expert_batches)
    routes = torch.cat(route_batches)
    labels = torch.cat(label_batches)
    candidate_logits = torch.cat((baseline.unsqueeze(1), experts), dim=1)

    def per_sample_loss(logits):
        if isinstance(criterion, nn.CrossEntropyLoss):
            weight = criterion.weight
            if weight is not None:
                weight = weight.detach().cpu()
            return F.cross_entropy(
                logits,
                labels,
                weight=weight,
                reduction="none",
                label_smoothing=criterion.label_smoothing,
            )
        return F.cross_entropy(logits, labels, reduction="none")

    candidate_losses = torch.stack([
        per_sample_loss(candidate_logits[:, index])
        for index in range(candidate_logits.size(1))
    ], dim=1)
    oracle_indices = candidate_losses.argmin(dim=1)
    row_indices = torch.arange(labels.numel())
    oracle_logits = candidate_logits[row_indices, oracle_indices]

    def metrics_and_loss(logits):
        metrics = compute_metrics(
            labels.numpy(), logits.argmax(dim=1).numpy(), num_classes,
        )
        losses = per_sample_loss(logits)
        if isinstance(criterion, nn.CrossEntropyLoss) and criterion.weight is not None:
            # Match CrossEntropyLoss(reduction="mean"): weighted CE divides
            # by the sum of target-class weights, not by the sample count.
            denominator = criterion.weight.detach().cpu()[labels].sum()
            loss = losses.sum() / denominator.clamp_min(1e-8)
        else:
            loss = losses.mean()
        metrics["loss"] = round(float(loss), 6)
        return metrics

    expert_names = list(getattr(model, "expert_names", ()))
    route_names = list(getattr(model, "route_names", ()))
    if len(expert_names) != experts.size(1):
        expert_names = [f"expert{index}" for index in range(experts.size(1))]
    if len(route_names) != routes.size(1):
        route_names = ["no_correction", *expert_names]

    expert_reports = {
        name: metrics_and_loss(experts[:, index])
        for index, name in enumerate(expert_names)
    }
    best_single_name = min(
        expert_names, key=lambda name: expert_reports[name]["loss"],
    )
    learned_indices = routes.argmax(dim=1)
    route_report = {
        name: {
            "mean_probability": round(float(routes[:, index].mean()), 6),
            "argmax_frequency": round(
                float((learned_indices == index).float().mean()), 6,
            ),
            "oracle_frequency": round(
                float((oracle_indices == index).float().mean()), 6,
            ),
        }
        for index, name in enumerate(route_names)
    }

    baseline_report = metrics_and_loss(baseline)
    uniform_report = metrics_and_loss(uniform)
    learned_report = metrics_and_loss(learned)
    oracle_report = metrics_and_loss(oracle_logits)
    accuracy_oracle_gap = (
        oracle_report["accuracy"] - baseline_report["accuracy"]
    )
    accuracy_gap_recovery = (
        (learned_report["accuracy"] - baseline_report["accuracy"])
        / accuracy_oracle_gap
        if abs(accuracy_oracle_gap) > 1e-8 else 0.0
    )

    return {
        "num_samples": int(labels.numel()),
        "baseline": baseline_report,
        "uniform_correction": uniform_report,
        "learned_router": learned_report,
        "oracle_router": oracle_report,
        "expert_candidates": expert_reports,
        "best_single_expert": best_single_name,
        "routing": route_report,
        "learned_matches_oracle": round(
            float((learned_indices == oracle_indices).float().mean()), 6,
        ),
        "oracle_majority_route_frequency": round(float(
            torch.bincount(
                oracle_indices, minlength=candidate_logits.size(1),
            ).max() / labels.numel()
        ), 6),
        "accuracy_oracle_gap_recovery": round(
            float(accuracy_gap_recovery), 6,
        ),
        "oracle_improves_loss_fraction": round(float(
            (candidate_losses.min(dim=1).values < candidate_losses[:, 0])
            .float().mean()
        ), 6),
    }


@torch.no_grad()
def calibrate_static_expert_fusion(
    model, dataloader, criterion, device, num_classes, tta=False,
    alpha_steps=41, temperatures=(0.01, 0.025, 0.05, 0.1, 0.25, 1.0),
    optimize_weights=True,
):
    """Fit a low-variance residual expert mixture on validation loss.

    Each temperature converts the standalone validation-loss improvements of
    the four experts into one global reliability distribution. A one-
    dimensional grid then selects the residual strength. The grid contains
    alpha=0, which is the protected semantic baseline exactly. Selection is
    strictly by minimum validation loss; accuracy is reported only afterward.
    """
    if not hasattr(model, "configure_static_fusion"):
        raise ValueError(
            "Static expert calibration requires a baseline-preserving "
            "residual expert model."
        )
    if alpha_steps < 2:
        raise ValueError("alpha_steps must be at least 2")
    temperatures = tuple(float(value) for value in temperatures)
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("All static-fusion temperatures must be positive")

    model.eval()
    baseline_batches = []
    expert_batches = []
    label_batches = []
    for images, labels in dataloader:
        images = images.to(device)
        _, _, details = _forward_with_tta_details(model, images, tta=tta)
        if (
            details.get("baseline_logits") is None
            or details.get("expert_logits") is None
        ):
            raise ValueError("Model did not expose baseline/expert logits")
        baseline_batches.append(details["baseline_logits"].cpu())
        expert_batches.append(details["expert_logits"].cpu())
        label_batches.append(labels.cpu())
    if not label_batches:
        raise ValueError("Cannot calibrate static fusion on an empty loader")

    baseline = torch.cat(baseline_batches)
    experts = torch.cat(expert_batches)
    labels = torch.cat(label_batches)
    deltas = experts - baseline.unsqueeze(1)

    weight = None
    label_smoothing = 0.0
    if isinstance(criterion, nn.CrossEntropyLoss):
        if criterion.weight is not None:
            weight = criterion.weight.detach().cpu()
        label_smoothing = criterion.label_smoothing

    def aggregate_loss(logits):
        losses = F.cross_entropy(
            logits, labels, weight=weight, reduction="none",
            label_smoothing=label_smoothing,
        )
        if weight is not None:
            return losses.sum() / weight[labels].sum().clamp_min(1e-8)
        return losses.mean()

    def report(logits):
        values = compute_metrics(
            labels.numpy(), logits.argmax(dim=1).numpy(), num_classes,
        )
        values["loss"] = round(float(aggregate_loss(logits)), 6)
        return values

    baseline_loss = aggregate_loss(baseline)
    expert_losses = torch.stack([
        aggregate_loss(experts[:, index])
        for index in range(experts.size(1))
    ])
    # Dimensionless positive improvements make the temperature grid portable
    # across binary and multiclass criteria. Experts worse than the baseline
    # receive no positive evidence, but remain available at high temperature.
    relative_gains = (
        (baseline_loss - expert_losses) / baseline_loss.abs().clamp_min(1e-8)
    ).clamp_min(0.0)

    best = None
    for temperature in temperatures:
        reliability = torch.softmax(relative_gains / temperature, dim=0)
        correction = (
            reliability.view(1, -1, 1) * deltas
        ).sum(dim=1)
        for alpha_tensor in torch.linspace(0.0, 1.0, alpha_steps):
            alpha = float(alpha_tensor)
            logits = baseline + alpha * correction
            loss = float(aggregate_loss(logits))
            # Deterministic tie-break: prefer the smaller correction, then the
            # smoother (higher-temperature) reliability distribution.
            key = (loss, alpha, -temperature)
            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "loss": loss,
                    "alpha": alpha,
                    "temperature": temperature,
                    "weights": reliability.clone(),
                    "logits": logits.clone(),
                    "optimizer": "reliability_grid",
                }

    # Jointly refine the same global mixture. This remains a four-effective-
    # parameter convex residual (one simplex over specialists plus its total
    # strength), not a sample router. Starting from the reliability solution
    # makes the optimization deterministic and LBFGS avoids another LR knob.
    if optimize_weights:
        initial_alpha = min(max(best["alpha"], 1e-4), 1.0 - 1e-4)
        alpha_logit = torch.tensor(
            np.log(initial_alpha / (1.0 - initial_alpha)),
            dtype=baseline.dtype, requires_grad=True,
        )
        weight_logits = best["weights"].clamp_min(1e-8).log().detach().clone()
        weight_logits.requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            (alpha_logit, weight_logits),
            lr=1.0, max_iter=100, tolerance_grad=1e-9,
            tolerance_change=1e-10, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            alpha = torch.sigmoid(alpha_logit)
            weights = torch.softmax(weight_logits, dim=0)
            correction = (weights.view(1, -1, 1) * deltas).sum(dim=1)
            loss = aggregate_loss(baseline + alpha * correction)
            loss.backward()
            return loss

        with torch.enable_grad():
            optimizer.step(closure)
        optimized_alpha = float(torch.sigmoid(alpha_logit).detach())
        optimized_weights = torch.softmax(
            weight_logits.detach(), dim=0,
        )
        optimized_logits = baseline + optimized_alpha * (
            optimized_weights.view(1, -1, 1) * deltas
        ).sum(dim=1)
        optimized_loss = float(aggregate_loss(optimized_logits))
        optimized_key = (
            optimized_loss, optimized_alpha, -best["temperature"],
        )
        if optimized_key < best["key"]:
            best.update({
                "key": optimized_key,
                "loss": optimized_loss,
                "alpha": optimized_alpha,
                "weights": optimized_weights,
                "logits": optimized_logits,
                "optimizer": "joint_convex_lbfgs",
            })

    model.configure_static_fusion(best["alpha"], best["weights"])
    expert_names = list(getattr(model, "expert_names", ()))
    if len(expert_names) != experts.size(1):
        expert_names = [f"expert{index}" for index in range(experts.size(1))]
    calibration = {
        "selection_metric": "minimum_validation_loss",
        "num_validation_samples": int(labels.numel()),
        "alpha": round(best["alpha"], 6),
        "temperature": best["temperature"],
        "fusion_optimizer": best["optimizer"],
        "expert_weights": {
            name: round(float(best["weights"][index]), 8)
            for index, name in enumerate(expert_names)
        },
        "expert_relative_loss_gains": {
            name: round(float(relative_gains[index]), 8)
            for index, name in enumerate(expert_names)
        },
        "baseline": report(baseline),
        "uniform_full_correction": report(baseline + deltas.mean(dim=1)),
        "calibrated_static_fusion": report(best["logits"]),
    }
    return calibration


def evaluate_complete_expert_fusion_ablation(
    model, val_loader, test_loader, device, num_classes, tta=False,
):
    """Evaluate post-hoc fusion controls for complete-expert MoE models.

    Global static logit weights are fitted exclusively on validation
    cross-entropy. Equal-logit and equal-probability fusion have no fitted
    parameters. The official test labels are used only after the static
    weights have been locked.
    """
    if test_loader is None:
        raise ValueError("Fusion ablation requires an available test split")

    def collect(loader):
        learned_batches = []
        expert_batches = []
        label_batches = []
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                learned, _, details = _forward_with_tta_details(
                    model, images.to(device), tta=tta,
                )
                experts = details.get("expert_logits")
                if experts is None:
                    raise ValueError(
                        "Model did not expose complete expert logits"
                    )
                learned_batches.append(learned.cpu())
                expert_batches.append(experts.cpu())
                label_batches.append(labels.cpu())
        if not label_batches:
            raise ValueError("Fusion ablation received an empty data loader")
        return (
            torch.cat(learned_batches),
            torch.cat(expert_batches),
            torch.cat(label_batches),
        )

    val_learned, val_experts, val_labels = collect(val_loader)
    test_learned, test_experts, test_labels = collect(test_loader)
    num_experts = val_experts.size(1)
    if test_experts.size(1) != num_experts:
        raise ValueError("Validation/test expert counts do not match")

    # The objective is convex in simplex weights. Multiple deterministic
    # starts make the softmax parameterization robust near simplex boundaries.
    candidates = []
    starts = [torch.zeros(num_experts)]
    for index in range(num_experts):
        start = torch.zeros(num_experts)
        start[index] = 2.0
        starts.append(start)
    for start in starts:
        weight_logits = start.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            (weight_logits,), lr=1.0, max_iter=200,
            tolerance_grad=1e-10, tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            weights = torch.softmax(weight_logits, dim=0)
            logits = (val_experts * weights.view(1, -1, 1)).sum(dim=1)
            loss = F.cross_entropy(logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            weights = torch.softmax(weight_logits, dim=0)
            logits = (val_experts * weights.view(1, -1, 1)).sum(dim=1)
            loss = float(F.cross_entropy(logits, val_labels))
        candidates.append((loss, weights.detach().clone()))
    static_val_loss, static_weights = min(
        candidates, key=lambda candidate: candidate[0],
    )

    def metrics_from_predictions(labels, predictions):
        return compute_metrics(
            labels.numpy(), predictions.numpy(), num_classes,
        )

    def report(learned, experts, labels):
        equal_logit = experts.mean(dim=1)
        equal_probability = F.softmax(experts, dim=-1).mean(dim=1)
        static_logit = (
            experts * static_weights.view(1, -1, 1)
        ).sum(dim=1)
        return {
            "equal_logit_average": metrics_from_predictions(
                labels, equal_logit.argmax(dim=1),
            ),
            "equal_probability_average": metrics_from_predictions(
                labels, equal_probability.argmax(dim=1),
            ),
            "static_logit_weights": metrics_from_predictions(
                labels, static_logit.argmax(dim=1),
            ),
            "disagreement_router": metrics_from_predictions(
                labels, learned.argmax(dim=1),
            ),
        }

    names = list(getattr(model, "expert_names", ()))
    if len(names) != num_experts:
        names = [f"expert{index}" for index in range(num_experts)]
    return {
        "protocol": {
            "static_weight_selection_split": "validation",
            "static_weight_selection_metric": "cross_entropy",
            "official_test_used_for_selection": False,
            "tta": bool(tta),
            "num_validation_samples": int(val_labels.numel()),
            "num_test_samples": int(test_labels.numel()),
        },
        "static_validation_cross_entropy": round(static_val_loss, 8),
        "static_expert_weights": {
            name: round(float(static_weights[index]), 8)
            for index, name in enumerate(names)
        },
        "validation": report(
            val_learned, val_experts, val_labels,
        ),
        "test": report(test_learned, test_experts, test_labels),
    }


@torch.no_grad()
def extract_features(model, dataloader, device, head_name="classifier"):
    model.eval()
    all_features = []
    all_labels = []

    handle = None
    features_capture = {}

    def hook_fn(module, input, output):
        features_capture["feat"] = input[0].detach()

    for name, module in model.named_modules():
        if name == head_name:
            handle = module.register_forward_hook(hook_fn)
            break

    for images, labels in dataloader:
        images = images.to(device)
        batch_size = images.size(0)
        paired_input = bool(getattr(model, "paired_input", False))
        crop_views = images.size(1) if images.dim() == 5 and not paired_input else 1
        model_input = (
            images.flatten(0, 1)
            if images.dim() == 5 and not paired_input else images
        )
        model(model_input)

        if "feat" in features_capture:
            feat = features_capture["feat"]
            if feat.dim() == 4:
                feat = feat.mean(dim=[2, 3])
            elif feat.dim() == 3:
                # Complementary delta head input: [batch, experts, channels].
                feat = feat.mean(dim=1)
            elif feat.dim() > 4:
                feat = feat.flatten(2).mean(dim=2)
            if crop_views > 1:
                feat = feat.reshape(batch_size, crop_views, -1).mean(dim=1)
            all_features.append(feat.cpu().numpy())
        else:
            with torch.no_grad():
                feat = model(model_input)
            if isinstance(feat, dict):
                feat = feat["logits"]
            if feat.dim() == 4:
                feat = feat.mean(dim=[2, 3])
            elif feat.dim() == 3:
                feat = feat.mean(dim=1)
            elif feat.dim() > 4:
                feat = feat.flatten(2).mean(dim=2)
            if crop_views > 1:
                feat = feat.reshape(batch_size, crop_views, -1).mean(dim=1)
            all_features.append(feat.cpu().numpy())

        all_labels.extend(labels.numpy())

    if handle is not None:
        handle.remove()

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    return features, labels
