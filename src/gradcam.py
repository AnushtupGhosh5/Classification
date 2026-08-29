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

    if model_name == "convnext_tiny":
        # Final ConvNeXt stage, final block, spatial depthwise convolution.
        # The later MLP layers operate channel-last and are not convolutional.
        return bb.features[7][-1].block[0]
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


class _ExpertOutputWrapper(nn.Module):
    """Expose one expert's logits without changing the trained model."""

    def __init__(self, model, expert_index):
        super().__init__()
        self.model = model
        self.expert_index = int(expert_index)

    def forward(self, x):
        out = self.model(x)
        if not isinstance(out, dict) or out.get("expert_logits") is None:
            raise RuntimeError(
                "Expert Grad-CAM requires model output['expert_logits']."
            )
        return out["expert_logits"][:, self.expert_index]


class _PairedGradCAM(_GradCAM):
    """Treat ``[B, 2, C, H, W]`` as paired 2D images, not 3D volumes."""

    @staticmethod
    def get_target_width_height(input_tensor):
        return input_tensor.size(-1), input_tensor.size(-2)


_EXPERT_DESCRIPTIONS = {
    "texture": "Texture / fine detail",
    "morphology": "Morphology / lesion structure",
    "semantic": "Semantic / deep context",
    "color": "Color / chromatic cues",
    "boundary": "Boundary / edge cues",
}


def _get_expert_target_layer(expert):
    """Return a spatial convolution in the expert before terminal attention.

    In particular, searching the entire semantic expert would select a 1x1
    convolution inside its pooled channel gate. That is technically inside the
    branch but cannot produce a useful spatial Grad-CAM. The residual stack is
    the last spatial feature transform common to these experts.
    """
    # Paired MILK experts wrap one shared feature processor and then gate its
    # clinical/dermoscopic outputs. Recurse into that processor so the hooked
    # convolution is spatial and belongs to the expert, rather than falling
    # through to a pooled modality gate.
    shared = getattr(expert, "shared", None)
    if shared is not None:
        target = _get_expert_target_layer(shared)
        if target is not None:
            return target

    blocks = getattr(expert, "blocks", None)
    target = _find_deepest_conv(blocks) if blocks is not None else None
    if target is not None:
        return target
    for attribute in ("project", "mix", "stages", "stem"):
        module = getattr(expert, attribute, None)
        target = _find_deepest_conv(module) if module is not None else None
        if target is not None:
            return target
    return _find_deepest_conv(expert)


def _collect_balanced_samples(
    dataloader, class_names, num_images, paired_input=False,
):
    """Collect an approximately equal, deterministic sample count per class."""
    num_classes = len(class_names) if class_names else None
    if not num_classes:
        target_count = num_images
        quotas = None
    else:
        # A class-balanced paper figure must not silently omit classes merely
        # because the caller requested fewer panels than there are classes.
        target_count = max(int(num_images), num_classes)
        base, remainder = divmod(target_count, num_classes)
        quotas = {
            index: base + (1 if index < remainder else 0)
            for index in range(num_classes)
        }

    images = []
    labels = []
    counts = {index: 0 for index in range(num_classes or 0)}
    for batch in dataloader:
        batch_images, batch_labels = batch[:2]
        for index in range(batch_images.size(0)):
            label = int(batch_labels[index])
            if quotas is not None and counts[label] >= quotas[label]:
                continue
            image = batch_images[index]
            if image.dim() == 4 and not paired_input:
                image = image[image.size(0) // 2]
            images.append(image)
            labels.append(label)
            if quotas is not None:
                counts[label] += 1
            if len(images) >= target_count:
                break
        if len(images) >= target_count:
            break
    return images, labels


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
    num_classes = len(class_names) if class_names else None
    per_class_limit = max(1, (num_images + num_classes - 1) // num_classes) if num_classes else None
    class_counts = {index: 0 for index in range(num_classes or 0)}

    # FolderDataset is class-sorted, so taking the first batch visualizes only
    # class 0. Select a balanced sample before computing the expensive CAMs.
    for batch_images, batch_labels in dataloader:
        for i in range(batch_images.size(0)):
            label = batch_labels[i].item()
            if per_class_limit is not None and class_counts[label] >= per_class_limit:
                continue
            # Multi-crop evaluation tensors are [views, C, H, W]. Use the
            # centre view for a single interpretable Grad-CAM panel.
            image = batch_images[i]
            if image.dim() == 4:
                image = image[image.size(0) // 2]
            images_list.append(image)
            labels_list.append(label)
            if per_class_limit is not None:
                class_counts[label] += 1
            if len(images_list) >= num_images:
                break
        if len(images_list) >= num_images:
            break

    if not images_list:
        return

    images_tensor = torch.stack(images_list).to(device)
    model.eval()
    with torch.no_grad():
        preds_list = wrapped(images_tensor).argmax(dim=1).cpu().tolist()
    # A protected/frozen backbone has no trainable parameters. Grad-CAM still
    # needs an autograd path to its target activations, so explicitly request
    # gradients with respect to the visualization input.
    cam_input = images_tensor.detach().requires_grad_(True)
    cams_list = cam(input_tensor=cam_input)

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
    if hasattr(model, "expert_modules"):
        names = list(getattr(model, "expert_names", model.expert_modules.keys()))
        for name in names:
            expert_targets.append(
                _get_expert_target_layer(model.expert_modules[name])
            )
        if expert_names is None:
            expert_names = [
                _EXPERT_DESCRIPTIONS.get(name, name.replace("_", " ").title())
                for name in names
            ]
    elif hasattr(model, "semantic_branch"):
        for i, attr in enumerate(["semantic_branch", "frequency_branch", "geometry_branch"]):
            branch = getattr(model, attr)
            target = _find_deepest_conv(branch)
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

    paired_input = bool(getattr(model, "paired_input", False))
    gradcam_class = _PairedGradCAM if paired_input else _GradCAM
    images_list, labels_list = _collect_balanced_samples(
        dataloader, class_names, num_images, paired_input=paired_input,
    )
    if not images_list:
        return
    images_batch = torch.stack(images_list).to(device)
    labels_batch = torch.tensor(labels_list, dtype=torch.long)
    n = images_batch.size(0)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        details = model(images_batch)
        if not isinstance(details, dict):
            print("GradCAM: Expert model did not return a details dictionary.")
            model.train(was_training)
            return
        fused_preds = details["logits"].argmax(dim=1).cpu()
        expert_logits = details.get("expert_logits")
        router_weights = details.get("router_weights")
        if expert_logits is None or router_weights is None:
            print("GradCAM: Missing expert logits or router weights. Skipping.")
            model.train(was_training)
            return
        expert_preds = expert_logits.argmax(dim=2).cpu()
        router_weights = router_weights.detach().cpu()

    n_experts = len(expert_targets)
    if expert_logits.size(1) != n_experts:
        model.train(was_training)
        raise RuntimeError(
            f"Found {n_experts} expert branches but "
            f"expert_logits has {expert_logits.size(1)} entries."
        )

    # Each wrapper returns only expert_logits[:, i]. Therefore both the score
    # being differentiated and the hooked convolution belong to expert i.
    expert_cams = []
    for expert_index, target in enumerate(expert_targets):
        if target is None:
            expert_cams.append(None)
            continue
        wrapped_expert = _ExpertOutputWrapper(model, expert_index)
        cam_input = images_batch.detach().requires_grad_(True)
        with gradcam_class(
            model=wrapped_expert, target_layers=[target],
        ) as cam_obj:
            expert_cams.append(cam_obj(input_tensor=cam_input))

    fig, axes = plt.subplots(
        n, n_experts + 1,
        figsize=(3.6 * (n_experts + 1), 3.5 * n),
        squeeze=False,
    )

    for i in range(n):
        if paired_input:
            clinical_np = _denormalize(images_batch[i, 0].cpu())
            dermoscopic_np = _denormalize(images_batch[i, 1].cpu())
            original_np = np.concatenate((clinical_np, dermoscopic_np), axis=1)
        else:
            img_np = _denormalize(images_batch[i].cpu())
            original_np = img_np

        if class_names:
            t_str = class_names[labels_batch[i].item()]
            fused_str = class_names[fused_preds[i].item()]
        else:
            t_str = str(labels_batch[i].item())
            fused_str = str(fused_preds[i].item())

        axes[i, 0].imshow(original_np)
        axes[i, 0].set_title(
            (
                "Original: Clinical | Dermoscopic\n"
                if paired_input else "Original\n"
            ) + f"True: {t_str} | Fused: {fused_str}", fontsize=9,
            color="green" if labels_batch[i] == fused_preds[i] else "red",
        )
        axes[i, 0].axis("off")

        for j, grayscale_cams in enumerate(expert_cams):
            neutral_name = f"E{j + 1}"
            description = expert_names[j]
            if grayscale_cams is None:
                axes[i, j + 1].set_title(
                    f"{neutral_name} — {description}\n(no spatial conv)",
                    fontsize=8,
                )
                axes[i, j + 1].axis("off")
                continue
            if paired_input and len(grayscale_cams) == 2 * n:
                # Paired semantic/color processors evaluate the same expert
                # convolution for clinical samples followed by dermoscopic
                # samples. Preserve both modality-specific CAMs.
                clinical_cam = show_cam_on_image(
                    clinical_np, grayscale_cams[i], use_rgb=True,
                )
                dermoscopic_cam = show_cam_on_image(
                    dermoscopic_np, grayscale_cams[n + i], use_rgb=True,
                )
                cam_img = np.concatenate(
                    (clinical_cam, dermoscopic_cam), axis=1,
                )
                modality_note = "Clinical | Dermoscopic"
            elif paired_input:
                expert_key = names[j] if j < len(names) else ""
                use_clinical = expert_key == "boundary"
                base_image = clinical_np if use_clinical else dermoscopic_np
                cam_img = show_cam_on_image(
                    base_image, grayscale_cams[i], use_rgb=True,
                )
                modality_note = "Clinical" if use_clinical else "Dermoscopic"
            else:
                cam_img = show_cam_on_image(
                    img_np, grayscale_cams[i], use_rgb=True,
                )
                modality_note = ""
            expert_pred = int(expert_preds[i, j])
            expert_pred_str = (
                class_names[expert_pred] if class_names else str(expert_pred)
            )
            axes[i, j + 1].imshow(cam_img)
            axes[i, j + 1].set_title(
                f"{neutral_name} — {description}\n"
                + (f"{modality_note}\n" if modality_note else "")
                + f"Pred: {expert_pred_str} | Router: {router_weights[i, j]:.3f}",
                fontsize=8,
            )
            axes[i, j + 1].axis("off")

    plt.suptitle(
        "Expert-specific Grad-CAM (each map uses its own expert logits)",
        fontsize=14, y=1.002,
    )
    plt.tight_layout()
    path = os.path.join(
        save_dir, f"{model_label}_expert_specific_gradcam.png",
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved expert-specific GradCAM: {path}")

    # The overall decision is differentiated from the final fused logits. Use
    # every expert's spatial target: a deep-backbone-only target would exclude
    # the early/intermediate experts and the independent chromatic path.
    fused_targets = [target for target in expert_targets if target is not None]
    if not fused_targets:
        print("GradCAM: Could not find fused-model expert target layers.")
        model.train(was_training)
        return
    wrapped_fused = _DictOutputWrapper(model)
    if paired_input:
        # Expert targets do not all have the same activation batch dimension:
        # texture/morphology/boundary see one modality (N), while the shared
        # semantic/color processors see both modalities (2N). Compute each
        # final-logit CAM independently and aggregate per modality.
        clinical_maps = []
        dermoscopic_maps = []
        for expert_index, target in enumerate(expert_targets):
            if target is None:
                continue
            fused_input = images_batch.detach().requires_grad_(True)
            with gradcam_class(
                model=wrapped_fused, target_layers=[target],
            ) as cam_obj:
                target_cams = cam_obj(input_tensor=fused_input)
            if len(target_cams) == 2 * n:
                clinical_maps.append(target_cams[:n])
                dermoscopic_maps.append(target_cams[n:])
            elif names[expert_index] == "boundary":
                clinical_maps.append(target_cams)
            else:
                dermoscopic_maps.append(target_cams)

        def aggregate_maps(maps):
            if not maps:
                return np.zeros((n, *images_batch.shape[-2:]), dtype=np.float32)
            combined = np.mean(np.stack(maps, axis=0), axis=0)
            flat = combined.reshape(n, -1)
            minimum = flat.min(axis=1)[:, None, None]
            maximum = flat.max(axis=1)[:, None, None]
            return (combined - minimum) / np.maximum(maximum - minimum, 1e-8)

        clinical_fused_cams = aggregate_maps(clinical_maps)
        dermoscopic_fused_cams = aggregate_maps(dermoscopic_maps)
    else:
        fused_input = images_batch.detach().requires_grad_(True)
        with gradcam_class(
            model=wrapped_fused, target_layers=fused_targets,
        ) as cam_obj:
            fused_cams = cam_obj(input_tensor=fused_input)

    fused_columns = 4 if paired_input else 2
    fused_fig, fused_axes = plt.subplots(
        n, fused_columns, figsize=(4 * fused_columns, 3.5 * n), squeeze=False,
    )
    for index in range(n):
        if paired_input:
            clinical = _denormalize(images_batch[index, 0].cpu())
            dermoscopic = _denormalize(images_batch[index, 1].cpu())
        else:
            image = _denormalize(images_batch[index].cpu())
        true_index = labels_batch[index].item()
        pred_index = fused_preds[index].item()
        true_name = class_names[true_index] if class_names else str(true_index)
        pred_name = class_names[pred_index] if class_names else str(pred_index)
        title = f"True: {true_name} | Pred: {pred_name}"
        title_color = "green" if true_index == pred_index else "red"
        if paired_input:
            panels = (
                (clinical, "Original clinical"),
                (show_cam_on_image(
                    clinical, clinical_fused_cams[index], use_rgb=True,
                ), "Fused decision — clinical evidence"),
                (dermoscopic, "Original dermoscopic"),
                (show_cam_on_image(
                    dermoscopic, dermoscopic_fused_cams[index], use_rgb=True,
                ), "Fused decision — dermoscopic evidence"),
            )
            for column, (panel, heading) in enumerate(panels):
                fused_axes[index, column].imshow(panel)
                fused_axes[index, column].set_title(
                    f"{heading}\n{title}", fontsize=9, color=title_color,
                )
                fused_axes[index, column].axis("off")
        else:
            fused_axes[index, 0].imshow(image)
            fused_axes[index, 0].set_title(
                f"Original\n{title}", fontsize=9, color=title_color,
            )
            fused_axes[index, 0].axis("off")
            fused_axes[index, 1].imshow(
                show_cam_on_image(image, fused_cams[index], use_rgb=True),
            )
            fused_axes[index, 1].set_title(
                f"Final fused-model Grad-CAM (E1–E{n_experts})\n{title}",
                fontsize=9, color=title_color,
            )
            fused_axes[index, 1].axis("off")
    fused_fig.suptitle(
        "Final fused-model Grad-CAM (out['logits'])", fontsize=14, y=1.002,
    )
    fused_fig.tight_layout()
    fused_path = os.path.join(
        save_dir, f"{model_label}_final_fused_gradcam.png",
    )
    fused_fig.savefig(fused_path, dpi=300, bbox_inches="tight")
    plt.close(fused_fig)
    model.train(was_training)
    print(f"Saved final fused-model GradCAM: {fused_path}")
