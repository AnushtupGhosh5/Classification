import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.attention import get_attention_module


class MobileNetV2WithAttention(nn.Module):
    def __init__(self, backbone, attention_module, num_classes):
        super().__init__()
        self.features = backbone.features
        self.attention = attention_module
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone.last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_mobilenetv2(num_classes=2, pretrained=True, attention=None):
    backbone = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    )
    if attention and attention != "none":
        attn_module = get_attention_module(attention, backbone.last_channel)
        model = MobileNetV2WithAttention(backbone, attn_module, num_classes)
        return model, "classifier"
    backbone.classifier[1] = nn.Linear(backbone.last_channel, num_classes)
    return backbone, "classifier"
