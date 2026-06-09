import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.expert_branches import SemanticExpert, FrequencyExpert, GeometryExpert
from src.models.base_expert_fusion import BaseExpertFusion


class CompetitiveExpertFusion(BaseExpertFusion):
    def __init__(
        self,
        semantic_expert,
        frequency_expert,
        geometry_expert,
        proj_dim,
        num_classes,
        top_k=2,
    ):
        super().__init__(
            semantic_expert, frequency_expert, geometry_expert, proj_dim, num_classes
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
    backbone2="mobilenetv2",
    backbone3="densenet121",
    proj_dim=256,
    top_k=2,
):
    semantic = SemanticExpert(backbone1, pretrained)
    frequency = FrequencyExpert(backbone2, pretrained)
    geometry = GeometryExpert(backbone3, pretrained)
    model = CompetitiveExpertFusion(
        semantic, frequency, geometry, proj_dim, num_classes, top_k
    )
    return model, "head"
