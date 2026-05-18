import torch
import torch.nn as nn
import torchvision.models as models


def _create_vit(weight_cls, model_fn, num_classes, pretrained):
    backbone = model_fn(weights=weight_cls.DEFAULT if pretrained else None)
    backbone.heads.head = nn.Linear(backbone.hidden_dim, num_classes)
    return backbone, "heads"


def create_vit_b16(num_classes=2, pretrained=True, attention=None):
    return _create_vit(
        models.ViT_B_16_Weights, models.vit_b_16,
        num_classes, pretrained,
    )


def create_vit_b32(num_classes=2, pretrained=True, attention=None):
    return _create_vit(
        models.ViT_B_32_Weights, models.vit_b_32,
        num_classes, pretrained,
    )
