import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_expert_fusion import BaseExpertFusion
from src.models.expert_branches import MultiLayerExpert


class CompetitiveExpertFusion(BaseExpertFusion):
    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        proj_dim=256,
        num_classes=2,
        top_k=2,
        expert_mode="shared_base",
        multi_layer_expert=None,
        branch_depth=2,
    ):
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            proj_dim=proj_dim,
            num_classes=num_classes,
            expert_mode=expert_mode,
            multi_layer_expert=multi_layer_expert,
            branch_depth=branch_depth,
        )
        self.top_k = min(top_k, 3)

        self.router = nn.Sequential(
            nn.Linear(proj_dim * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        fs, ff, fg = self.extract_expert_features(x)

        ps = F.adaptive_avg_pool2d(fs, (1, 1)).flatten(1)
        pf = F.adaptive_avg_pool2d(ff, (1, 1)).flatten(1)
        pg = F.adaptive_avg_pool2d(fg, (1, 1)).flatten(1)
        scores = F.softmax(self.router(torch.cat([ps, pf, pg], dim=1)), dim=1)

        topk_vals, topk_idx = scores.topk(self.top_k, dim=1)

        # Renormalize selected weights to sum to 1
        topk_vals = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-8)

        experts = [fs, ff, fg]
        fused = torch.zeros_like(fs)
        for k in range(self.top_k):
            idx = topk_idx[:, k]
            weight = topk_vals[:, k].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            for e_idx in range(3):
                mask = (idx == e_idx).unsqueeze(1).unsqueeze(2).unsqueeze(3).float()
                fused = fused + weight * mask * experts[e_idx]

        return self.pool_and_classify(fused)


def create_cef(
    num_classes=2,
    pretrained=True,
    attention=None,
    backbone1="resnet50",
    backbone2=None,
    backbone3=None,
    proj_dim=256,
    top_k=2,
    expert_mode="shared_base",
    branch_depth=2,
):
    if expert_mode == "multi_layer":
        ml_expert = MultiLayerExpert(backbone1, pretrained)
        model = CompetitiveExpertFusion(
            backbone_name=backbone1,
            pretrained=pretrained,
            proj_dim=proj_dim,
            num_classes=num_classes,
            top_k=top_k,
            expert_mode="multi_layer",
            multi_layer_expert=ml_expert,
        )
    else:
        model = CompetitiveExpertFusion(
            backbone_name=backbone1,
            pretrained=pretrained,
            proj_dim=proj_dim,
            num_classes=num_classes,
            top_k=top_k,
            expert_mode="shared_base",
            branch_depth=branch_depth,
        )
    return model, "head"
