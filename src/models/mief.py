import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.expert_branches import SemanticExpert, MultiLayerExpert
from src.models.base_expert_fusion import BaseExpertFusion


class MutualInfoExpertFusion(BaseExpertFusion):
    def __init__(
        self,
        expert1=None,
        expert2=None,
        expert3=None,
        proj_dim=256,
        num_classes=2,
        mi_dim=128,
        expert_mode="multi_backbone",
        multi_layer_expert=None,
    ):
        super().__init__(
            expert1=expert1,
            expert2=expert2,
            expert3=expert3,
            proj_dim=proj_dim,
            num_classes=num_classes,
            expert_mode=expert_mode,
            multi_layer_expert=multi_layer_expert,
        )
        self.mi_dim = mi_dim

        self.phi_s = nn.Sequential(
            nn.Conv2d(proj_dim, mi_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.phi_f = nn.Sequential(
            nn.Conv2d(proj_dim, mi_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.phi_g = nn.Sequential(
            nn.Conv2d(proj_dim, mi_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.mi_estimator = nn.Sequential(
            nn.Linear(mi_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.weight_gen = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        fs, ff, fg = self.extract_expert_features(x)

        zs = F.adaptive_avg_pool2d(self.phi_s(fs), (1, 1)).flatten(1)
        zf = F.adaptive_avg_pool2d(self.phi_f(ff), (1, 1)).flatten(1)
        zg = F.adaptive_avg_pool2d(self.phi_g(fg), (1, 1)).flatten(1)

        mi_sf = self.mi_estimator(torch.cat([zs, zf], dim=-1)).squeeze(-1)
        mi_sg = self.mi_estimator(torch.cat([zs, zg], dim=-1)).squeeze(-1)
        mi_fg = self.mi_estimator(torch.cat([zf, zg], dim=-1)).squeeze(-1)

        c_matrix = torch.stack([1 - mi_sf, 1 - mi_sg, 1 - mi_fg], dim=-1)
        weights = F.softmax(self.weight_gen(c_matrix), dim=-1)

        ws = weights[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        wf = weights[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        wg = weights[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        fused = ws * fs + wf * ff + wg * fg

        return self.pool_and_classify(fused)


def create_mief(
    num_classes=2,
    pretrained=True,
    attention=None,
    backbone1="resnet50",
    backbone2="mobilenetv2",
    backbone3="densenet121",
    proj_dim=256,
    expert_mode="multi_backbone",
):
    if expert_mode == "multi_layer":
        ml_expert = MultiLayerExpert(backbone1, pretrained)
        model = MutualInfoExpertFusion(
            proj_dim=proj_dim,
            num_classes=num_classes,
            expert_mode="multi_layer",
            multi_layer_expert=ml_expert,
        )
    else:
        expert1 = SemanticExpert(backbone1, pretrained)
        expert2 = SemanticExpert(backbone2, pretrained)
        expert3 = SemanticExpert(backbone3, pretrained)
        model = MutualInfoExpertFusion(
            expert1=expert1,
            expert2=expert2,
            expert3=expert3,
            proj_dim=proj_dim,
            num_classes=num_classes,
            expert_mode="multi_backbone",
        )
    return model, "head"
