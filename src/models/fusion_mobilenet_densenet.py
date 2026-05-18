import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.attention import SEBlock, get_attention_module


def create_fusion_mobilenet_densenet(num_classes=2, pretrained=True, attention=None):
    mobilenet = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    )
    mobilenet_dim = mobilenet.last_channel
    mobilenet.classifier = nn.Identity()

    densenet = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
    )
    densenet_dim = densenet.classifier.in_features
    densenet.classifier = nn.Identity()

    fusion_dim = mobilenet_dim + densenet_dim

    model = FusionMobileNetDenseNet(mobilenet, densenet, fusion_dim, num_classes, attention)
    return model, "head"


class _SE1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class FusionMobileNetDenseNet(nn.Module):
    def __init__(self, mobilenet, densenet, fusion_dim, num_classes, attention=None):
        super().__init__()
        self.mobilenet = mobilenet
        self.densenet = densenet
        self.fusion_dim = fusion_dim

        layers = []
        if attention and attention != "none":
            layers.append(_SE1D(fusion_dim, reduction=16))
        layers.extend([
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        ])
        self.head = nn.Sequential(*layers)

    def _extract_mobilenet(self, x):
        x = self.mobilenet.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def _extract_densenet(self, x):
        x = self.densenet.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def forward(self, x):
        f_mob = self._extract_mobilenet(x)
        f_den = self._extract_densenet(x)
        fused = torch.cat([f_mob, f_den], dim=1)
        return self.head(fused)
