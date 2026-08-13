"""Minimal paired-image baseline for MILK10k.

The two modalities are processed in one batch by one shared CNN. Their global
features are averaged per lesion and passed through one classifier. There are
no expert branches, attention modules, routers, or auxiliary MoE losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone_extractor import create_backbone


class PairedBackboneClassifier(nn.Module):
    paired_input = True

    def __init__(
        self,
        backbone_name="convnext_tiny",
        pretrained=True,
        num_classes=11,
        dropout=0.2,
    ):
        super().__init__()
        self.backbone = create_backbone(backbone_name, pretrained)
        self._backbone_module_name = "backbone"
        self.backbone_name = backbone_name
        feature_dim = self.backbone.feature_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim, eps=1e-6)
            if backbone_name == "convnext_tiny" else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, images):
        if images.dim() != 5 or images.size(1) != 2:
            raise ValueError(
                "paired backbone expects [batch, 2, channels, height, width]"
            )
        batch_size = images.size(0)
        # This is one invocation of one shared CNN, not two model instances.
        features = self.backbone(images.flatten(0, 1))
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        # Fixed mean fusion adds no learned fusion parameters and gives the
        # MoE a clean, reproducible paired-modality control.
        lesion_features = features.reshape(batch_size, 2, -1).mean(dim=1)
        return self.classifier(lesion_features)


def create_paired_convnext_tiny(
    num_classes=11,
    pretrained=True,
    attention=None,
    classifier_dropout=0.2,
    **_unused,
):
    model = PairedBackboneClassifier(
        backbone_name="convnext_tiny",
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=classifier_dropout,
    )
    return model, "classifier"
