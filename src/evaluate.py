import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import json
from torch.utils.data import DataLoader, Subset
from src.utils import (
    compute_expert_diagnostics,
    compute_oracle_diagnostics,
    evaluate_model,
    extract_features,
    forward_with_tta,
)
from src.visualize import (
    plot_confusion_matrix,
    plot_expert_comparison,
    plot_roc_auc,
    plot_tsne,
)
from src.gradcam import visualize_gradcam, visualize_gradcam_per_expert


PAPER_EXPERT_ORDER = (
    ("E1", "texture"),
    ("E2", "morphology"),
    ("E3", "semantic"),
    ("E4", "color"),
    ("E5", "boundary"),
)


@torch.no_grad()
def export_router_sanity_check(
    model,
    test_loader,
    device,
    class_names,
    save_dir,
    model_label,
    samples_per_class=5,
    seed=42,
):
    """Export per-image routing weights for a balanced test subset."""
    if not hasattr(model, "expert_names"):
        raise ValueError("Router sanity check requires an expert-routing model")
    if test_loader is None or len(test_loader.dataset) == 0:
        raise ValueError("Router sanity check requires a non-empty test split")
    if not hasattr(test_loader.dataset, "samples"):
        raise ValueError("Test dataset does not expose image paths via .samples")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    samples = test_loader.dataset.samples
    indices_by_class = {index: [] for index in range(len(class_names))}
    for index, sample in enumerate(samples):
        _path, label = sample[:2]
        indices_by_class[int(label)].append(index)

    rng = random.Random(seed)
    selected_indices = []
    for class_index, class_name in enumerate(class_names):
        candidates = list(indices_by_class[class_index])
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Class {class_name} has only {len(candidates)} test images; "
                f"cannot select {samples_per_class}"
            )
        rng.shuffle(candidates)
        selected_indices.extend(sorted(candidates[:samples_per_class]))

    selected_loader = DataLoader(
        Subset(test_loader.dataset, selected_indices),
        batch_size=min(test_loader.batch_size or 1, len(selected_indices)),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(getattr(test_loader, "pin_memory", False)),
    )

    internal_names = tuple(model.expert_names)
    missing = [name for _label, name in PAPER_EXPERT_ORDER if name not in internal_names]
    if missing:
        raise ValueError(f"Checkpoint is missing required experts: {missing}")
    paper_columns = [
        (paper_label, name, internal_names.index(name))
        for paper_label, name in PAPER_EXPERT_ORDER
    ]

    model.eval()
    rows = []
    offset = 0
    for images, labels in selected_loader:
        images = images.to(device)
        result = model(images)
        if not isinstance(result, dict) or result.get("router_weights") is None:
            raise ValueError("Model forward pass did not return router_weights")
        predictions = result["logits"].argmax(dim=1)
        weights = result["router_weights"]

        for batch_index in range(images.size(0)):
            dataset_index = selected_indices[offset + batch_index]
            image_path, stored_label = samples[dataset_index][:2]
            true_index = int(labels[batch_index])
            if true_index != int(stored_label):
                raise RuntimeError("Selected test image/label alignment failed")
            row = {
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "true_class": class_names[true_index],
                "predicted_class": class_names[int(predictions[batch_index])],
                "correct": int(predictions[batch_index]) == true_index,
            }
            for paper_label, _name, internal_index in paper_columns:
                row[paper_label] = float(
                    weights[batch_index, internal_index].cpu()
                )
            row["weight_sum"] = sum(
                row[paper_label]
                for paper_label, _name, _internal_index in paper_columns
            )
            rows.append(row)
        offset += images.size(0)

    weight_fields = [paper_label for paper_label, _name, _index in paper_columns]
    os.makedirs(save_dir, exist_ok=True)
    summary_rows = []
    for class_name in class_names:
        class_rows = [row for row in rows if row["true_class"] == class_name]
        summary = {"true_class": class_name, "samples": len(class_rows)}
        for field in weight_fields:
            values = np.asarray([row[field] for row in class_rows], dtype=np.float64)
            summary[f"mean_{field}"] = float(values.mean())
            summary[f"std_{field}"] = float(values.std(ddof=0))
        summary_rows.append(summary)

    report_path = os.path.join(save_dir, f"{model_label}_router_sanity.md")
    report = [
        "# Router sanity check",
        "",
        "E1–E5 are neutral branch identifiers in the model's implemented order. "
        "They do not assert that a branch has learned a particular clinical concept.",
        "",
        "## Class-wise routing weights",
        "",
        "| True class | Samples | " + " | ".join(weight_fields) + " |",
        "|---|---:|" + "---:|" * len(weight_fields),
    ]
    for summary in summary_rows:
        cells = [
            f"{summary[f'mean_{field}']:.4f} ± {summary[f'std_{field}']:.4f}"
            for field in weight_fields
        ]
        report.append(
            f"| {summary['true_class']} | {summary['samples']} | "
            + " | ".join(cells) + " |"
        )
    report.extend([
        "",
        "## Per-image routing weights",
        "",
        "| Image | True | Predicted | Correct | "
        + " | ".join(weight_fields) + " | Sum |",
        "|---|---|---|:---:|" + "---:|" * (len(weight_fields) + 1),
    ])
    for row in rows:
        weights = " | ".join(f"{row[field]:.6f}" for field in weight_fields)
        report.append(
            f"| {row['image_name']} | {row['true_class']} | "
            f"{row['predicted_class']} | {'yes' if row['correct'] else 'no'} | "
            f"{weights} | {row['weight_sum']:.6f} |"
        )
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report) + "\n")

    print("\nROUTER SANITY CHECK (five deterministic test images per class):")
    print(
        "  Branch order: E1, E2, E3, E4, E5 (neutral labels)"
    )
    print("\n  Per-image routing weights:")
    for row in rows:
        values = " | ".join(
            f"{paper_label}={row[paper_label]:.6f}"
            for paper_label, _name, _index in paper_columns
        )
        print(
            f"  {row['true_class']:<4} | {row['image_name']:<24} | "
            f"pred={row['predicted_class']:<4} | {values} | "
            f"sum={row['weight_sum']:.6f}"
        )

    print("\n  Class-wise mean routing weights:")
    for summary in summary_rows:
        means = " | ".join(
            f"{paper_label}={summary[f'mean_{paper_label}']:.4f}"
            for paper_label, _name, _index in paper_columns
        )
        print(f"  {summary['true_class']:<4} mean: {means}")
    print(f"Saved router report: {report_path}")
    return {"rows": rows, "class_summary": summary_rows}


@torch.no_grad()
def export_router_repeatability_check(
    model,
    test_loader,
    device,
    class_names,
    save_dir,
    model_label,
    class_name="MEL",
    repeats=20,
    seed=42,
):
    """Forward one fixed test image repeatedly and export every weight vector."""
    if not hasattr(model, "expert_names"):
        raise ValueError("Router repeatability check requires an expert-routing model")
    if test_loader is None or len(test_loader.dataset) == 0:
        raise ValueError("Router repeatability check requires a non-empty test split")
    if not hasattr(test_loader.dataset, "samples"):
        raise ValueError("Test dataset does not expose image paths via .samples")
    if repeats <= 1:
        raise ValueError("Router repeatability check requires at least two repeats")
    if class_name not in class_names:
        raise ValueError(
            f"Unknown repeatability class {class_name!r}; choose from {class_names}"
        )

    class_index = class_names.index(class_name)
    samples = test_loader.dataset.samples
    candidates = [
        index for index, sample in enumerate(samples)
        if int(sample[1]) == class_index
    ]
    if not candidates:
        raise ValueError(f"Test split contains no images for class {class_name}")
    selected_index = random.Random(seed).choice(candidates)
    image_path, stored_label = samples[selected_index][:2]

    internal_names = tuple(model.expert_names)
    missing = [name for _label, name in PAPER_EXPERT_ORDER if name not in internal_names]
    if missing:
        raise ValueError(f"Checkpoint is missing required experts: {missing}")
    paper_columns = [
        (paper_label, name, internal_names.index(name))
        for paper_label, name in PAPER_EXPERT_ORDER
    ]

    model.eval()
    rows = []
    vectors = []
    for repeat_index in range(repeats):
        # Fetch and transform the same dataset item independently on every pass.
        image, label = test_loader.dataset[selected_index]
        if int(label) != int(stored_label):
            raise RuntimeError("Repeated test image/label alignment failed")
        result = model(image.unsqueeze(0).to(device))
        if not isinstance(result, dict) or result.get("router_weights") is None:
            raise ValueError("Model forward pass did not return router_weights")
        weights = result["router_weights"][0]
        prediction = int(result["logits"].argmax(dim=1)[0])
        vector = torch.stack([
            weights[internal_index]
            for _paper_label, _name, internal_index in paper_columns
        ]).detach().cpu()
        vectors.append(vector)
        row = {
            "repeat": repeat_index + 1,
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "true_class": class_names[int(label)],
            "predicted_class": class_names[prediction],
        }
        for column_index, (paper_label, name, _internal_index) in enumerate(
            paper_columns
        ):
            row[paper_label] = float(vector[column_index])
        row["weight_sum"] = float(vector.sum())
        rows.append(row)

    matrix = torch.stack(vectors)
    reference = matrix[0]
    max_abs_deviation = float((matrix - reference).abs().max())
    exactly_identical = bool(torch.equal(matrix, reference.expand_as(matrix)))

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(
        save_dir, f"{model_label}_router_repeatability.md",
    )
    weight_fields = [paper_label for paper_label, _name, _index in paper_columns]
    report = [
        "# Router repeatability check",
        "",
        f"- Image: `{image_path}`",
        f"- True class: {class_name}",
        f"- Independent forwards: {repeats}",
        f"- Exactly identical: {exactly_identical}",
        f"- Maximum absolute weight deviation: {max_abs_deviation:.12g}",
        "",
        "| Repeat | Predicted | " + " | ".join(weight_fields) + " | Sum |",
        "|---:|---|" + "---:|" * (len(weight_fields) + 1),
    ]
    for row in rows:
        values = " | ".join(f"{row[field]:.9f}" for field in weight_fields)
        report.append(
            f"| {row['repeat']} | {row['predicted_class']} | {values} | "
            f"{row['weight_sum']:.9f} |"
        )
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report) + "\n")

    print("\nROUTER REPEATABILITY CHECK:")
    print(f"  Image: {image_path}")
    print(f"  True class: {class_name} | independent forwards: {repeats}")
    for row in rows:
        values = " | ".join(
            f"{paper_label}={row[paper_label]:.9f}"
            for paper_label, _name, _index in paper_columns
        )
        print(
            f"  repeat={row['repeat']:02d} | pred={row['predicted_class']:<4} | "
            f"{values} | sum={row['weight_sum']:.9f}"
        )
    print(f"  Exactly identical: {exactly_identical}")
    print(f"  Maximum absolute weight deviation: {max_abs_deviation:.12g}")
    print(f"Saved repeated router weights: {output_path}")
    return {
        "image_path": image_path,
        "rows": rows,
        "exactly_identical": exactly_identical,
        "max_abs_deviation": max_abs_deviation,
    }


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
        if "raw_router_entropy" in report and abs(
            report["raw_router_entropy"] - report["router_entropy"]
        ) > 1e-6:
            print(
                f"  Raw adaptive-router entropy: "
                f"{report['raw_router_entropy']:.4f} / "
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
        plot_expert_comparison(reports, save_dir, model_label)
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

    try:
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
    except Exception as error:
        # Interpretability plots are optional post-hoc artifacts. A plotting
        # incompatibility must not discard already-computed test metrics or
        # prevent their CSV/history persistence.
        print(
            "GradCAM generation failed; metrics remain valid and will still "
            f"be saved. Error: {error}"
        )

    metrics["macro_auc"] = round(macro_auc, 4)
    if calibrate_binary and num_classes == 2:
        metrics["decision_threshold"] = round(float(threshold), 6)
    return metrics
