import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone_extractor import create_backbone
from src.models.expert_branches import (
    SemanticBranch, FrequencyBranch, GeometryBranch, MultiLayerExpert,
)


SHARED_BASE_EXCLUDED = {"vit_b16", "vit_b32"}


class BaseExpertFusion(nn.Module):
    """Shared-base expert fusion architecture.

    Pipeline: Input -> [Shared Backbone] -> F_base
              -> [SemanticBranch | FrequencyBranch | GeometryBranch]
              -> (Fs, Ff, Fg) -> [Fusion Method] -> Head -> logits

    expert_mode:
      - "shared_base" (default): one shared backbone + 3 lightweight branches
      - "multi_layer": one backbone, features from 3 different layers
    """

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        proj_dim=256,
        num_classes=2,
        expert_mode="shared_base",
        multi_layer_expert=None,
        branch_depth=2,
    ):
        super().__init__()
        self.expert_mode = expert_mode
        self.proj_dim = proj_dim

        if expert_mode == "shared_base":
            if backbone_name in SHARED_BASE_EXCLUDED:
                raise ValueError(
                    f"shared_base mode not supported for {backbone_name}. "
                    f"ViT models produce 1D features without spatial structure."
                )
            self.shared_backbone = create_backbone(backbone_name, pretrained)
            in_channels = self.shared_backbone.feature_dim

            self.semantic_branch = SemanticBranch(in_channels, proj_dim, branch_depth)
            self.frequency_branch = FrequencyBranch(in_channels, proj_dim, branch_depth)
            self.geometry_branch = GeometryBranch(in_channels, proj_dim, branch_depth)

            self._backbone_module_name = "shared_backbone"

        elif expert_mode == "multi_layer":
            self.multi_layer_expert = multi_layer_expert
            channels = multi_layer_expert.channels

            self.proj_s = nn.Conv2d(channels[0], proj_dim, 1)
            self.proj_f = nn.Conv2d(channels[1], proj_dim, 1)
            self.proj_g = nn.Conv2d(channels[2], proj_dim, 1)

            self._backbone_module_name = "multi_layer_expert"

        else:
            raise ValueError(f"Unknown expert_mode '{expert_mode}'")

        self.head = nn.Sequential(
            nn.BatchNorm1d(proj_dim),
            nn.Dropout(0.4),
            nn.Linear(proj_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def _align_features(self, features_list):
        min_h = min(f.shape[2] for f in features_list)
        min_w = min(f.shape[3] for f in features_list)
        aligned = []
        for f in features_list:
            h, w = f.shape[2], f.shape[3]
            if h > min_h or w > min_w:
                f = F.adaptive_avg_pool2d(f, (min_h, min_w))
            elif h < min_h or w < min_w:
                f = F.interpolate(
                    f, size=(min_h, min_w), mode="bilinear", align_corners=False
                )
            aligned.append(f)
        return aligned

    def extract_expert_features(self, x):
        if self.expert_mode == "multi_layer":
            f1, f2, f3 = self.multi_layer_expert(x)
            f1 = self.proj_s(f1)
            f2 = self.proj_f(f2)
            f3 = self.proj_g(f3)
        else:
            f_base = self.shared_backbone(x)
            if f_base.dim() != 4:
                raise ValueError(
                    f"Shared backbone must produce 4D features, got shape {f_base.shape}"
                )
            fs = self.semantic_branch(f_base)
            ff = self.frequency_branch(f_base)
            fg = self.geometry_branch(f_base)
            f1, f2, f3 = fs, ff, fg

        f1, f2, f3 = self._align_features([f1, f2, f3])
        return f1, f2, f3

    def pool_and_classify(self, fused):
        pooled = F.adaptive_avg_pool2d(fused, (1, 1))
        flat = torch.flatten(pooled, 1)
        return self.head(flat)

    def forward(self, x):
        raise NotImplementedError
