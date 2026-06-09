import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.expert_branches import (
    SemanticExpert,
    FrequencyExpert,
    GeometryExpert,
    MultiLayerExpert,
)
from src.models.base_expert_fusion import BaseExpertFusion


class ExpertDisagreementFusion(BaseExpertFusion):
    def __init__(
        self,
        semantic_expert=None,
        frequency_expert=None,
        geometry_expert=None,
        proj_dim=256,
        num_classes=2,
        disagreement_type="abs",
        expert_mode="multi_backbone",
        multi_layer_expert=None,
    ):
        super().__init__(
            semantic_expert=semantic_expert,
            frequency_expert=frequency_expert,
            geometry_expert=geometry_expert,
            proj_dim=proj_dim,
            num_classes=num_classes,
            expert_mode=expert_mode,
            multi_layer_expert=multi_layer_expert,
        )
        self.disagreement_type = disagreement_type

        if disagreement_type == "learnable":
            self.mlp_sf = nn.Sequential(
                nn.Conv2d(proj_dim * 2, proj_dim, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(proj_dim, proj_dim, 1, bias=False),
            )
            self.mlp_sg = nn.Sequential(
                nn.Conv2d(proj_dim * 2, proj_dim, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(proj_dim, proj_dim, 1, bias=False),
            )
            self.mlp_fg = nn.Sequential(
                nn.Conv2d(proj_dim * 2, proj_dim, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(proj_dim, proj_dim, 1, bias=False),
            )

        self.aggregate_conv = nn.Sequential(
            nn.Conv2d(proj_dim * 3, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )

        self.attn_conv = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def compute_disagreement(self, fi, fj, mlp=None):
        if self.disagreement_type == "abs":
            return torch.abs(fi - fj)
        if self.disagreement_type == "cosine":
            sim = F.cosine_similarity(fi, fj, dim=1).unsqueeze(1)
            return (1 - sim).expand_as(fi)
        if self.disagreement_type == "learnable":
            return mlp(torch.cat([fi, fj], dim=1))

    def forward(self, x):
        fs, ff, fg = self.extract_expert_features(x)

        if self.disagreement_type == "learnable":
            d_sf = self.compute_disagreement(fs, ff, self.mlp_sf)
            d_sg = self.compute_disagreement(fs, fg, self.mlp_sg)
            d_fg = self.compute_disagreement(ff, fg, self.mlp_fg)
        else:
            d_sf = self.compute_disagreement(fs, ff)
            d_sg = self.compute_disagreement(fs, fg)
            d_fg = self.compute_disagreement(ff, fg)

        d = self.aggregate_conv(torch.cat([d_sf, d_sg, d_fg], dim=1))
        w = self.attn_conv(d)
        f_shared = (fs + ff + fg) / 3.0

        fused = w * d + (1 - w) * f_shared
        return self.pool_and_classify(fused)


def create_edf(
    num_classes=2,
    pretrained=True,
    attention=None,
    backbone1="resnet50",
    backbone2="mobilenetv2",
    backbone3="densenet121",
    proj_dim=256,
    disagreement_type="abs",
    expert_mode="multi_backbone",
):
    if expert_mode == "multi_layer":
        ml_expert = MultiLayerExpert(backbone1, pretrained)
        model = ExpertDisagreementFusion(
            proj_dim=proj_dim,
            num_classes=num_classes,
            disagreement_type=disagreement_type,
            expert_mode="multi_layer",
            multi_layer_expert=ml_expert,
        )
    else:
        semantic = SemanticExpert(backbone1, pretrained)
        frequency = FrequencyExpert(backbone2, pretrained)
        geometry = GeometryExpert(backbone3, pretrained)
        model = ExpertDisagreementFusion(
            semantic_expert=semantic,
            frequency_expert=frequency_expert,
            geometry_expert=geometry,
            proj_dim=proj_dim,
            num_classes=num_classes,
            disagreement_type=disagreement_type,
            expert_mode="multi_backbone",
        )
    return model, "head"
