import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
from src.utils import (
    compute_expert_diagnostics,
    compute_oracle_diagnostics,
    evaluate_model,
    extract_features,
    forward_with_tta,
)
from src.visualize import plot_confusion_matrix, plot_roc_auc, plot_tsne
from src.gradcam import visualize_gradcam, visualize_gradcam_per_expert


def evaluate_all_splits(model, train_loader, val_loader, test_loader, criterion, device, num_classes,
                        tta=False):
    results = {}

    for split_name, loader in [("train", train_loader), ("validation", val_loader), ("test", test_loader)]:
        if loader is None or len(loader.dataset) == 0:
            print(f"Skipping {split_name}: no data")
            continue

        metrics, logits = evaluate_model(model, loader, criterion, device, num_classes, tta=tta)
        results[split_name] = metrics
        results[f"{split_name}_logits"] = logits

        print(f"\n{split_name.upper()} Results:")
        print(f"  Loss:        {metrics['loss']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  Bal. Acc:    {metrics['balanced_accuracy']:.4f}")
        if "malignant_recall" in metrics:
            print(f"  Mal. Recall: {metrics['malignant_recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")

    return results


def run_expert_diagnostics(model, val_loader, test_loader, device, num_classes,
                           class_names, save_dir, model_label, tta=False):
    """Report and persist diagnostics without affecting ordinary models."""
    if not hasattr(model, "expert_names"):
        return None

    reports = {}
    for split_name, loader in (("validation", val_loader), ("test", test_loader)):
        if loader is None or len(loader.dataset) == 0:
            continue
        report = compute_expert_diagnostics(
            model, loader, device, num_classes,
            class_names=class_names, tta=tta,
        )
        if report is None:
            continue
        reports[split_name] = report
        print(f"\n{split_name.upper()} EXPERT/ROUTER DIAGNOSTICS:")
        print(
            f"  Router entropy: {report['router_entropy']:.4f} / "
            f"{report['maximum_entropy']:.4f}"
        )
        if "baseline_path" in report:
            baseline = report["baseline_path"]
            print(
                f"  Baseline path Acc/F1/BAcc={baseline['accuracy']:.3f}/"
                f"{baseline['f1']:.3f}/{baseline['balanced_accuracy']:.3f} | "
                f"correction scale={report.get('correction_scale', 0.0):.4f}"
            )
        for name, routing in report["routing"].items():
            expert_metrics = report["standalone_experts"][name]
            print(
                f"  {name:<10} weight={routing['mean_weight']:.3f}±"
                f"{routing['std_weight']:.3f} | "
                f"argmax={routing['argmax_frequency']:.3f} | "
                f"active={routing['active_frequency']:.3f} | "
                f"standalone Acc/F1={expert_metrics['accuracy']:.3f}/"
                f"{expert_metrics['f1']:.3f}"
            )

    if reports:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_label}_expert_diagnostics.json")
        with open(path, "w") as file:
            json.dump(reports, file, indent=2)
        print(f"Saved expert diagnostics: {path}")
    return reports or None


def run_oracle_diagnostics(model, val_loader, test_loader, criterion, device,
                           num_classes, save_dir, model_label, tta=False):
    """Persist the oracle-first ablation without using it for optimization."""
    if not bool(getattr(model, "oracle_protocol", False)):
        return None
    reports = {}
    for split_name, loader in (("validation", val_loader), ("test", test_loader)):
        if loader is None or len(loader.dataset) == 0:
            continue
        report = compute_oracle_diagnostics(
            model, loader, criterion, device, num_classes, tta=tta,
        )
        if report is None:
            continue
        reports[split_name] = report
        print(f"\n{split_name.upper()} ORACLE-FIRST DIAGNOSTICS:")
        for name in (
            "baseline", "uniform_correction", "learned_router", "oracle_router",
        ):
            metrics = report[name]
            print(
                f"  {name:<20} loss={metrics['loss']:.4f} | "
                f"Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f} | "
                f"BAcc={metrics['balanced_accuracy']:.4f}"
            )
        print(
            f"  Best single expert: {report['best_single_expert']} | "
            f"oracle improves {report['oracle_improves_loss_fraction']:.1%} "
            f"of samples | learned/oracle route agreement "
            f"{report['learned_matches_oracle']:.1%} "
            f"(majority shortcut={report['oracle_majority_route_frequency']:.1%}) | "
            f"accuracy oracle-gap recovery="
            f"{report['accuracy_oracle_gap_recovery']:.1%}"
        )
        for name, route in report["routing"].items():
            print(
                f"  Route {name:<13} p={route['mean_probability']:.3f} | "
                f"argmax={route['argmax_frequency']:.3f} | "
                f"oracle={route['oracle_frequency']:.3f}"
            )

    if reports:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_label}_oracle_diagnostics.json")
        with open(path, "w") as file:
            json.dump(reports, file, indent=2)
        print(f"Saved oracle diagnostics: {path}")
    return reports or None


def run_test_evaluation(model, test_loader, class_names, num_classes, device,
                        save_dir, model_label, head_name="classifier",
                        model_name=None, is_expert_fusion=False, tta=False,
                        val_loader=None, calibrate_binary=False):
    os.makedirs(save_dir, exist_ok=True)

    all_preds = []
    all_labels = []
    all_logits = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs, _ = forward_with_tta(model, images, tta=tta)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.extend(outputs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    shifted_logits = all_logits - all_logits.max(axis=1, keepdims=True)
    all_probs = np.exp(shifted_logits) / np.exp(shifted_logits).sum(axis=1, keepdims=True)

    threshold = 0.5
    validation_accuracy = None
    if calibrate_binary and num_classes == 2 and val_loader is not None:
        val_labels = []
        val_logits = []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs, _ = forward_with_tta(model, images.to(device), tta=tta)
                val_logits.extend(outputs.cpu().numpy())
                val_labels.extend(labels.numpy())
        val_logits = np.asarray(val_logits)
        val_labels = np.asarray(val_labels)
        shifted = val_logits - val_logits.max(axis=1, keepdims=True)
        val_probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        malignant_probs = val_probs[:, 1]
        candidates = np.unique(np.concatenate(([0.5], malignant_probs)))
        best = (-1.0, -1.0, -1.0, 0.5)
        from sklearn.metrics import balanced_accuracy_score
        for candidate in candidates:
            candidate_preds = (malignant_probs >= candidate).astype(int)
            accuracy = float((candidate_preds == val_labels).mean())
            balanced = balanced_accuracy_score(val_labels, candidate_preds)
            score = (accuracy, balanced, -abs(float(candidate) - 0.5), float(candidate))
            if score > best:
                best = score
        validation_accuracy, _, _, threshold = best
        all_preds = (all_probs[:, 1] >= threshold).astype(int)
        calibration_path = os.path.join(save_dir, f"{model_label}_calibration.json")
        with open(calibration_path, "w") as file:
            json.dump({
                "threshold": threshold,
                "selection_split": "validation",
                "selection_metric": "accuracy",
                "validation_accuracy": validation_accuracy,
                "tta": tta,
            }, file, indent=2)
        print(
            f"\nBinary calibration (validation only): threshold={threshold:.4f}, "
            f"validation accuracy={validation_accuracy:.4f}"
        )

    from src.utils import compute_metrics, compute_per_class_metrics
    metrics = compute_metrics(all_labels, all_preds, num_classes)
    per_class_metrics = compute_per_class_metrics(all_labels, all_preds, num_classes, class_names)

    print(f"\n{'='*60}")
    print("TEST RESULTS (Best Model)")
    print(f"{'='*60}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Bal. Acc:    {metrics['balanced_accuracy']:.4f}")
    if "malignant_recall" in metrics:
        print(f"  Mal. Recall: {metrics['malignant_recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print("\n  Per-class:")
    for row in per_class_metrics:
        print(
            f"    {row['class']}: "
            f"P={row['precision']:.4f} R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} N={row['support']}"
        )

    plot_confusion_matrix(all_labels, all_preds, class_names, save_dir, model_label)

    macro_auc = plot_roc_auc(all_labels, all_probs, num_classes, class_names, save_dir, model_label)
    print(f"  Macro AUC:   {macro_auc:.4f}")

    features, feat_labels = extract_features(model, test_loader, device, head_name)
    plot_tsne(features, feat_labels, class_names, save_dir, model_label)

    if is_expert_fusion:
        visualize_gradcam_per_expert(
            model, test_loader, device, save_dir, model_label,
            num_images=4, class_names=class_names,
        )
    else:
        visualize_gradcam(
            model, test_loader, device, save_dir, model_label,
            num_images=8, model_name=model_name, class_names=class_names,
        )

    metrics["macro_auc"] = round(macro_auc, 4)
    if calibrate_binary and num_classes == 2:
        metrics["decision_threshold"] = round(float(threshold), 6)
    return metrics
