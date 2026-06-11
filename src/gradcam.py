import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._handles = []
        self._handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if isinstance(output, dict):
            output = output["logits"]

        if target_class is None:
            target_class = output.argmax(dim=1)

        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            return None

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze(0).squeeze(0).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def generate_batch(self, input_tensor, target_classes=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if isinstance(output, dict):
            output = output["logits"]

        if target_classes is None:
            target_classes = output.argmax(dim=1)

        cams = []
        for i in range(input_tensor.size(0)):
            self.model.zero_grad()
            out = self.model(input_tensor[i : i + 1])
            if isinstance(out, dict):
                out = out["logits"]
            one_hot = torch.zeros_like(out)
            one_hot[0, target_classes[i]] = 1.0
            out.backward(gradient=one_hot, retain_graph=True)

            if self.gradients is None or self.activations is None:
                cams.append(None)
                continue

            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cams.append(cam)

        return cams

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _get_last_conv_layer(model, model_name):
    if model_name in ("mobilenetv2", "mobilenetv3_small", "mobilenetv3_large"):
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
            return model.backbone.features[-1][0]
        if hasattr(model, 'features'):
            return model.features[-1][0]
        return None
    if model_name == "densenet121":
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
            return model.backbone.features.denseblock4.denselayer16.conv2
        if hasattr(model, 'features'):
            return model.features.denseblock4.denselayer16.conv2
        return None
    if model_name in ("resnet34", "resnet50", "resnet101"):
        conv_name = "conv3" if model_name != "resnet34" else "conv2"
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
            return getattr(model.backbone.layer4[-1], conv_name)
        if hasattr(model, 'layer4'):
            return getattr(model.layer4[-1], conv_name)
        return None
    if model_name.startswith("efficientnet_b"):
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
            return model.backbone.features[-1][0]
        if hasattr(model, 'features'):
            return model.features[-1][0]
        return None
    if model_name.startswith("squeezenet"):
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
            return model.backbone.features[-1]
        if hasattr(model, 'features'):
            return model.features[-1]
        return None
    if model_name == "vgg16":
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
            return model.backbone.features[-1]
        if hasattr(model, 'features'):
            return model.features[-1]
        return None
    return None


def _find_last_conv(module):
    all_convs = []

    def _collect(mod, depth):
        for name, child in mod.named_children():
            if isinstance(child, torch.nn.Conv2d):
                all_convs.append((depth, id(child), child))
            _collect(child, depth + 1)

    _collect(module, 0)
    if not all_convs:
        return None
    all_convs.sort(key=lambda x: (x[0], x[1]))
    return all_convs[-1][2]


def _find_last_conv_in_backbone(model, backbone_attr, model_type):
    if not hasattr(model, backbone_attr):
        return None
    backbone = getattr(model, backbone_attr)

    if hasattr(backbone, 'extractor'):
        backbone = backbone.extractor
    if hasattr(backbone, 'backbone'):
        backbone = backbone.backbone

    if model_type in ("mobilenetv2", "mobilenetv3_small", "mobilenetv3_large"):
        if hasattr(backbone, 'features'):
            return backbone.features[-1][0]
    elif model_type == "densenet121":
        if hasattr(backbone, 'features'):
            return backbone.features.denseblock4.denselayer16.conv2
    elif model_type in ("resnet34", "resnet50", "resnet101"):
        conv_name = "conv3" if model_type != "resnet34" else "conv2"
        if hasattr(backbone, 'layer4'):
            return getattr(backbone.layer4[-1], conv_name)
    elif model_type.startswith("efficientnet_b"):
        if hasattr(backbone, 'features'):
            return backbone.features[-1][0]
    elif model_type.startswith("squeezenet"):
        if hasattr(backbone, 'features'):
            return backbone.features[-1]
    elif model_type == "vgg16":
        if hasattr(backbone, 'features'):
            return backbone.features[-1]

    return _find_last_conv(backbone)


def get_target_layer(model, model_name=None):
    if model_name is not None:
        layer = _get_last_conv_layer(model, model_name)
        if layer is not None:
            return layer

    return _find_last_conv(model)


def _denormalize_image(tensor, mean=None, std=None):
    img = tensor.clone().cpu()
    if mean is not None and std is not None:
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
    return img.permute(1, 2, 0).numpy()


def _resize_cam(cam, target_size):
    cam_tensor = torch.FloatTensor(cam).unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(cam_tensor, size=target_size, mode="bilinear", align_corners=False)
    return cam_resized.squeeze().numpy()


def visualize_gradcam(
    model,
    dataloader,
    device,
    save_dir,
    model_label,
    num_images=8,
    target_layer=None,
    model_name=None,
):
    os.makedirs(save_dir, exist_ok=True)

    if target_layer is None:
        target_layer = get_target_layer(model, model_name)
        if target_layer is None:
            print("GradCAM: Could not find a target conv layer. Skipping.")
            return

    grad_cam = GradCAM(model, target_layer)

    images_collected = []
    labels_collected = []
    cams_collected = []
    preds_collected = []

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    model.eval()
    with torch.enable_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if len(images_collected) >= num_images:
                break

            images_dev = images.to(device)

            with torch.no_grad():
                out = model(images_dev)
                if isinstance(out, dict):
                    out = out["logits"]
                batch_preds = out.argmax(dim=1).cpu()

            batch_cams = []
            for i in range(images.size(0)):
                if len(images_collected) >= num_images:
                    break
                if images.size(0) == 1:
                    images_dev_i = images_dev
                else:
                    images_dev_i = images_dev[i : i + 1]

                target_layer = _get_target_for_model(model, model_name)
                if target_layer is None:
                    batch_cams.append(None)
                    continue

                gc = GradCAM(model, target_layer)
                cam = gc.generate(images_dev_i)
                gc.remove_hooks()
                batch_cams.append(cam)

            for i in range(images.size(0)):
                if len(images_collected) >= num_images:
                    break
                images_collected.append(images[i])
                labels_collected.append(labels[i].item())
                cams_collected.append(batch_cams[i] if i < len(batch_cams) else None)
                preds_collected.append(batch_preds[i].item())

    grad_cam.remove_hooks()

    if not images_collected:
        print("GradCAM: No images to visualize.")
        return

    cols = min(4, len(images_collected))
    rows = (len(images_collected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (img, cam, label, pred) in enumerate(
        zip(images_collected, cams_collected, labels_collected, preds_collected)
    ):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        img_np = _denormalize_image(img, mean, std)
        img_np = np.clip(img_np, 0, 1)

        ax.imshow(img_np)

        if cam is not None:
            cam_resized = _resize_cam(cam, (img_np.shape[0], img_np.shape[1]))
            ax.imshow(cam_resized, cmap="jet", alpha=0.4)

        color = "green" if label == pred else "red"
        ax.set_title(f"True: {label} | Pred: {pred}", fontsize=9, color=color)
        ax.axis("off")

    for idx in range(len(images_collected), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    plt.suptitle(f"GradCAM - {model_label}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_label}_gradcam.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved GradCAM visualizations: {path}")


def visualize_gradcam_per_expert(
    model,
    dataloader,
    device,
    save_dir,
    model_label,
    expert_names=None,
    num_images=4,
):
    os.makedirs(save_dir, exist_ok=True)

    experts = []
    if hasattr(model, "expert1"):
        experts = [model.expert1, model.expert2, model.expert3]
    elif hasattr(model, "multi_layer_expert"):
        experts = [model.multi_layer_expert]
    else:
        print("GradCAM: No expert modules found.")
        return

    if expert_names is None:
        if hasattr(model, "expert1"):
            expert_names = ["Expert1", "Expert2", "Expert3"]
        else:
            expert_names = ["MultiLayer"]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    images_batch, labels_batch = next(iter(dataloader))
    images_batch = images_batch[:num_images].to(device)
    labels_batch = labels_batch[:num_images]

    model.eval()
    with torch.no_grad():
        out = model(images_batch)
        if isinstance(out, dict):
            out = out["logits"]
        preds = out.argmax(dim=1).cpu()

    n_imgs = images_batch.size(0)
    n_experts = len(experts)
    fig, axes = plt.subplots(n_imgs, n_experts + 1, figsize=(4 * (n_experts + 1), 4 * n_imgs))
    if n_imgs == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_imgs):
        img = images_batch[i].cpu()
        img_np = _denormalize_image(img, mean, std)
        img_np = np.clip(img_np, 0, 1)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Input (T:{labels_batch[i].item()} P:{preds[i].item()})", fontsize=8)
        axes[i, 0].axis("off")

        for j, expert in enumerate(experts):
            target_layer = _find_last_conv(expert)
            if target_layer is None:
                axes[i, j + 1].set_title(f"{expert_names[j]}\n(no conv)", fontsize=8)
                axes[i, j + 1].axis("off")
                continue

            grad_cam = GradCAM(expert, target_layer)
            cam = grad_cam.generate(images_batch[i : i + 1])
            grad_cam.remove_hooks()

            axes[i, j + 1].imshow(img_np)
            if cam is not None:
                cam_resized = _resize_cam(cam, (img_np.shape[0], img_np.shape[1]))
                axes[i, j + 1].imshow(cam_resized, cmap="jet", alpha=0.4)
            axes[i, j + 1].set_title(expert_names[j], fontsize=8)
            axes[i, j + 1].axis("off")

    plt.suptitle(f"Expert GradCAM - {model_label}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_label}_expert_gradcam.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved expert GradCAM visualizations: {path}")
