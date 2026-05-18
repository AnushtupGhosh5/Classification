import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.attention import get_attention_module


class DenseNet121WithAttention(nn.Module):
    def __init__(self, backbone, attention_module, in_features, num_classes):
        super().__init__()
        self.features = backbone.features
        self.attention = attention_module
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_densenet121(num_classes=2, pretrained=True, attention=None):
    backbone = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
    )
    in_features = backbone.classifier.in_features
    if attention and attention != "none":
        attn_module = get_attention_module(attention, in_features)
        model = DenseNet121WithAttention(backbone, attn_module, in_features, num_classes)
        return model, "classifier"
    backbone.classifier = nn.Linear(in_features, num_classes)
    return backbone, "classifier"
