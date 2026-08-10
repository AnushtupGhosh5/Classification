import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.attention import get_attention_module


class EfficientNetWithAttention(nn.Module):
    def __init__(self, backbone, attention_module, in_features, num_classes):
        super().__init__()
        self.features = backbone.features
        self.attention = attention_module
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _create_efficientnet(weight_cls, model_fn, num_classes, pretrained, attention):
    backbone = model_fn(weights=weight_cls.DEFAULT if pretrained else None)
    in_features = backbone.classifier[1].in_features
    if attention and attention != "none":
        last_channel = backbone.features[-1].out_channels
        attn_module = get_attention_module(attention, last_channel)
        model = EfficientNetWithAttention(backbone, attn_module, in_features, num_classes)
        return model, "classifier"
    backbone.classifier[1] = nn.Linear(in_features, num_classes)
    return backbone, "classifier"


def create_efficientnet_b0(num_classes=2, pretrained=True, attention=None):
    return _create_efficientnet(
        models.EfficientNet_B0_Weights, models.efficientnet_b0,
        num_classes, pretrained, attention,
    )


def create_efficientnet_b1(num_classes=2, pretrained=True, attention=None):
    return _create_efficientnet(
        models.EfficientNet_B1_Weights, models.efficientnet_b1,
        num_classes, pretrained, attention,
    )


def create_efficientnet_b2(num_classes=2, pretrained=True, attention=None):
    return _create_efficientnet(
        models.EfficientNet_B2_Weights, models.efficientnet_b2,
        num_classes, pretrained, attention,
    )


def create_efficientnet_v2_s(num_classes=2, pretrained=True, attention=None):
    return _create_efficientnet(
        models.EfficientNet_V2_S_Weights, models.efficientnet_v2_s,
        num_classes, pretrained, attention,
    )
