import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM as _GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def _navigate_to_backbone(model):
    if hasattr(model, "extractor") and hasattr(model.extractor, "backbone"):
        return model.extractor.backbone
    if hasattr(model, "backbone"):
        return model.backbone
    return model


def get_target_layer(model, model_name=None):
    bb = _navigate_to_backbone(model)

    if model_name in ("mobilenetv2", "mobilenetv3_small", "mobilenetv3_large"):
        return bb.features[-1][-1]
    if model_name == "densenet121":
        return bb.features.denseblock4.denselayer16.conv2
    if model_name in ("resnet34", "resnet50", "resnet101"):
        return bb.layer4[-1].conv3 if model_name != "resnet34" else bb.layer4[-1].conv2
    if model_name and model_name.startswith("efficientnet_b"):
        return bb.features[-1][-1]
    if model_name and model_name.startswith("squeezenet"):
        return bb.features[-1]
    if model_name == "vgg16":
        return bb.features[-1]

    return _find_deepest_conv(bb)


def _find_deepest_conv(module):
    result = [None, -1]

    def _search(m, depth):
        for name, child in m.named_children():
            if isinstance(child, nn.Conv2d):
                if depth > result[1]:
                    result[0] = child
                    result[1] = depth
            _search(child, depth + 1)

    _search(module, 0)
    return result[0]


def _denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = tensor.clone().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return np.clip(img.permute(1, 2, 0).numpy(), 0, 1)


class _DictOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return out["logits"]
        return out


def visualize_gradcam(
    model,
    dataloader,
    device,
    save_dir,
    model_label,
    num_images=8,
    target_layer=None,
    model_name=None,
    class_names=None,
):
    os.makedirs(save_dir, exist_ok=True)

    if target_layer is None:
        target_layer = get_target_layer(model, model_name)
    if target_layer is None:
        print("GradCAM: Could not find target layer. Skipping.")
        return

    wrapped = _DictOutputWrapper(model)
    cam = _GradCAM(model=wrapped, target_layers=[target_layer])

    images_list = []
    labels_list = []
    cams_list = []
    preds_list = []

    model.eval()
    for batch_images, batch_labels in dataloader:
        if len(images_list) >= num_images:
            break

        batch_dev = batch_images.to(device)
        with torch.no_grad():
            out = wrapped(batch_dev)
            batch_preds = out.argmax(dim=1)

        grayscale_cams = cam(input_tensor=batch_dev)

        for i in range(batch_images.size(0)):
            if len(images_list) >= num_images:
                break
            images_list.append(batch_images[i])
            labels_list.append(batch_labels[i].item())
            cams_list.append(grayscale_cams[i])
            preds_list.append(batch_preds[i].item())

    if not images_list:
        return

    cols = min(4, len(images_list))
    rows = (len(images_list) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(len(images_list)):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        img_np = _denormalize(images_list[idx])
        cam_img = show_cam_on_image(img_np, cams_list[idx], use_rgb=True)
        ax.imshow(cam_img)

        true_label = labels_list[idx]
        pred_label = preds_list[idx]
        if class_names:
            true_str = class_names[true_label]
            pred_str = class_names[pred_label]
        else:
            true_str = str(true_label)
            pred_str = str(pred_label)

        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"T: {true_str} | P: {pred_str}", fontsize=9, color=color)
        ax.axis("off")

    for idx in range(len(images_list), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    plt.suptitle(f"GradCAM - {model_label}", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_label}_gradcam.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved GradCAM: {path}")


def visualize_gradcam_per_expert(
    model,
    dataloader,
    device,
    save_dir,
    model_label,
    expert_names=None,
    class_names=None,
    num_images=4,
):
    os.makedirs(save_dir, exist_ok=True)

    expert_targets = []
    if hasattr(model, "expert1"):
        for i, attr in enumerate(["expert1", "expert2", "expert3"]):
            expert = getattr(model, attr)
            target = _find_deepest_conv(expert)
            expert_targets.append(target)
        if expert_names is None:
            expert_names = ["Semantic", "Frequency", "Geometry"]
    elif hasattr(model, "multi_layer_expert"):
        expert_targets = [get_target_layer(model)]
        if expert_names is None:
            expert_names = ["MultiLayer"]
    else:
        print("GradCAM: No expert modules found.")
        return

    images_batch, labels_batch = next(iter(dataloader))
    n = min(num_images, images_batch.size(0))
    images_batch = images_batch[:n].to(device)
    labels_batch = labels_batch[:n]

    model.eval()
    wrapped = _DictOutputWrapper(model)
    with torch.no_grad():
        preds = wrapped(images_batch).argmax(dim=1).cpu()

    n_experts = len(expert_targets)
    fig, axes = plt.subplots(n, n_experts + 1, figsize=(4 * (n_experts + 1), 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img_np = _denormalize(images_batch[i].cpu())

        if class_names:
            t_str = class_names[labels_batch[i].item()]
            p_str = class_names[preds[i].item()]
        else:
            t_str = str(labels_batch[i].item())
            p_str = str(preds[i].item())

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Input (T:{t_str} P:{p_str})", fontsize=8)
        axes[i, 0].axis("off")

        for j, target in enumerate(expert_targets):
            if target is None:
                axes[i, j + 1].set_title(f"{expert_names[j]}\n(no conv)", fontsize=8)
                axes[i, j + 1].axis("off")
                continue

            cam_obj = _GradCAM(model=wrapped, target_layers=[target])
            grayscale_cam = cam_obj(input_tensor=images_batch[i : i + 1])
            cam_img = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)

            axes[i, j + 1].imshow(cam_img)
            axes[i, j + 1].set_title(expert_names[j], fontsize=8)
            axes[i, j + 1].axis("off")

    plt.suptitle(f"Expert GradCAM - {model_label}", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_label}_expert_gradcam.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved expert GradCAM: {path}")
