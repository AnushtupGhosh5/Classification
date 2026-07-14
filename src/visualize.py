import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


def plot_training_curves(history, save_dir, model_label):
    os.makedirs(save_dir, exist_ok=True)

    train_data = history.get("train", [])
    val_data = history.get("val", [])
    if not train_data or not val_data:
        return

    epochs = [d["epoch"] for d in train_data]
    train_loss = [d["loss"] for d in train_data]
    val_loss = [d["loss"] for d in val_data]
    train_acc = [d["accuracy"] for d in train_data]
    val_acc = [d["accuracy"] for d in val_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs, val_loss, "r-o", label="Val Loss", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, "b-o", label="Train Accuracy", markersize=3)
    axes[1].plot(epochs, val_acc, "r-o", label="Val Accuracy", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_label}_loss_accuracy_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves: {path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir, model_label):
    os.makedirs(save_dir, exist_ok=True)

    from sklearn.metrics import confusion_matrix as cm_fn
    cm = cm_fn(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) + 2), max(6, len(class_names) + 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    path = os.path.join(save_dir, f"{model_label}_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix: {path}")


def plot_roc_auc(y_true, y_probs, num_classes, class_names, save_dir, model_label):
    os.makedirs(save_dir, exist_ok=True)

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.get_cmap("Set1", num_classes)

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors(i), lw=2,
                label=f"{class_names[i]} (AUC = {roc_auc:.4f})")

    fpr_macro, tpr_macro, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    macro_auc = auc(fpr_macro, tpr_macro)
    ax.plot(fpr_macro, tpr_macro, color="navy", lw=2, linestyle="--",
            label=f"Macro-average (AUC = {macro_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"{model_label}_roc_auc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC-AUC plot: {path}")

    return macro_auc


def plot_tsne(features, labels, class_names, save_dir, model_label):
    os.makedirs(save_dir, exist_ok=True)

    n_samples = features.shape[0]
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.get_cmap("Set1", len(class_names))

    for i, name in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=[colors(i)], label=name, alpha=0.6, s=20,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE Visualization")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"{model_label}_tsne.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE plot: {path}")
