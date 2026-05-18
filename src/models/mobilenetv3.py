import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.attention import get_attention_module


class MobileNetV3WithAttention(nn.Module):
    def __init__(self, backbone, attention_module, in_features, num_classes):
        super().__init__()
        self.features = backbone.features
        self.attention = attention_module
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _create_mobilenetv3(variant, num_classes, pretrained, attention):
    if variant == "small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        in_features = backbone.classifier[0].in_features
    else:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        in_features = backbone.classifier[0].in_features

    if attention and attention != "none":
        last_channel = backbone.features[-1].out_channels
        attn_module = get_attention_module(attention, last_channel)
        model = MobileNetV3WithAttention(backbone, attn_module, in_features, num_classes)
        return model, "classifier"

    backbone.classifier = nn.Sequential(
        nn.Linear(in_features, in_features),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return backbone, "classifier"


def create_mobilenetv3_small(num_classes=2, pretrained=True, attention=None):
    return _create_mobilenetv3("small", num_classes, pretrained, attention)


def create_mobilenetv3_large(num_classes=2, pretrained=True, attention=None):
    return _create_mobilenetv3("large", num_classes, pretrained, attention)
