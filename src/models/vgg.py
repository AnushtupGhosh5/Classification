import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.attention import get_attention_module


class VGG16WithAttention(nn.Module):
    def __init__(self, backbone, attention_module, in_features, num_classes):
        super().__init__()
        self.features = backbone.features
        self.attention = attention_module
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_vgg16(num_classes=2, pretrained=True, attention=None):
    backbone = models.vgg16(
        weights=models.VGG16_Weights.DEFAULT if pretrained else None
    )
    in_features = 512 * 7 * 7
    if attention and attention != "none":
        attn_module = get_attention_module(attention, 512)
        model = VGG16WithAttention(backbone, attn_module, in_features, num_classes)
        return model, "classifier"
    backbone.classifier[6] = nn.Linear(4096, num_classes)
    return backbone, "classifier"
