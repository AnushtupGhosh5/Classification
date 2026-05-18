import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from src.utils import evaluate_model, extract_features
from src.visualize import plot_confusion_matrix, plot_roc_auc, plot_tsne


def evaluate_all_splits(model, train_loader, val_loader, test_loader, criterion, device, num_classes):
    results = {}

    for split_name, loader in [("train", train_loader), ("validation", val_loader), ("test", test_loader)]:
        if loader is None or len(loader.dataset) == 0:
            print(f"Skipping {split_name}: no data")
            continue

        metrics, logits = evaluate_model(model, loader, criterion, device, num_classes)
        results[split_name] = metrics
        results[f"{split_name}_logits"] = logits

        print(f"\n{split_name.upper()} Results:")
        print(f"  Loss:        {metrics['loss']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")

    return results


def run_test_evaluation(model, test_loader, class_names, num_classes, device,
                        save_dir, model_label, head_name="classifier"):
    os.makedirs(save_dir, exist_ok=True)

    all_preds = []
    all_labels = []
    all_logits = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.extend(outputs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    all_probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)

    from src.utils import compute_metrics
    metrics = compute_metrics(all_labels, all_preds, num_classes)

    print(f"\n{'='*60}")
    print("TEST RESULTS (Best Model)")
    print(f"{'='*60}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")

    plot_confusion_matrix(all_labels, all_preds, class_names, save_dir, model_label)

    macro_auc = plot_roc_auc(all_labels, all_probs, num_classes, class_names, save_dir, model_label)
    print(f"  Macro AUC:   {macro_auc:.4f}")

    features, feat_labels = extract_features(model, test_loader, device, head_name)
    plot_tsne(features, feat_labels, class_names, save_dir, model_label)

    metrics["macro_auc"] = round(macro_auc, 4)
    return metrics
