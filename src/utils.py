import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
    }


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        result = model(images)
        if isinstance(result, dict):
            outputs = result["logits"]
            loss = criterion(outputs, labels) + result.get("aux_loss", 0.0)
        else:
            outputs = result
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(outputs.cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    metrics = compute_metrics(all_labels, all_preds, num_classes)
    metrics["loss"] = round(avg_loss, 4)

    return metrics, np.array(all_logits)


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
        model(images)

        if "feat" in features_capture:
            feat = features_capture["feat"]
            if feat.dim() > 2:
                feat = feat.mean(dim=[2, 3])
            all_features.append(feat.cpu().numpy())
        else:
            with torch.no_grad():
                feat = model(images)
            if feat.dim() > 2:
                feat = feat.mean(dim=[2, 3])
            all_features.append(feat.cpu().numpy())

        all_labels.extend(labels.numpy())

    if handle is not None:
        handle.remove()

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    return features, labels
