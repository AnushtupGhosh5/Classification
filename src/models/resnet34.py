import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.attention import get_attention_module


class ResNet34WithAttention(nn.Module):
    def __init__(self, backbone, attention_module, in_features, num_classes):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.attention = attention_module
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )

    def _features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self._features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_resnet34(num_classes=2, pretrained=True, attention=None):
    backbone = models.resnet34(
        weights=models.ResNet34_Weights.DEFAULT if pretrained else None
    )
    in_features = backbone.fc.in_features
    if attention and attention != "none":
        attn_module = get_attention_module(attention, in_features)
        model = ResNet34WithAttention(backbone, attn_module, in_features, num_classes)
        return model, "fc"
    backbone.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return backbone, "fc"
