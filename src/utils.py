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
    crop_views = images.size(1) if images.dim() == 5 else 1
    base_images = images.flatten(0, 1) if images.dim() == 5 else images
    views = [base_images]
    if tta:
        views.extend([
            torch.flip(base_images, dims=[3]),
            torch.flip(base_images, dims=[2]),
            torch.flip(base_images, dims=[2, 3]),
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
    total_loss = 0.0
    total_samples = 0
    router_weights = []
    correction_scales = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs, aux_loss, details = _forward_with_tta_details(
            model, images, tta=tta,
        )
        loss = criterion(outputs, labels) + aux_loss

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(outputs.cpu().numpy())
        if details.get("router_weights") is not None:
            router_weights.append(details["router_weights"].cpu())
        if details.get("correction_scale") is not None:
            correction_scales.append(details["correction_scale"].cpu())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
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
def compute_expert_diagnostics(model, dataloader, device, num_classes,
                               class_names=None, tta=False):
    """Collect standalone expert and sample-level routing diagnostics."""
    model.eval()
    weights_batches = []
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
        crop_views = images.size(1) if images.dim() == 5 else 1
        model_input = images.flatten(0, 1) if images.dim() == 5 else images
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
