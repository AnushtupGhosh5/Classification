import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.expert_branches import SemanticExpert, FrequencyExpert, GeometryExpert, MultiLayerExpert
from src.models.base_expert_fusion import BaseExpertFusion


class ConfidenceAwareExpertFusion(BaseExpertFusion):
    def __init__(
        self,
        expert1=None,
        expert2=None,
        expert3=None,
        proj_dim=256,
        num_classes=2,
        confidence_type="scalar",
        fuzzy_lambda=0.1,
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
        self.confidence_type = confidence_type
        self.fuzzy_lambda = fuzzy_lambda

        if confidence_type == "scalar":
            self.conf_head_s = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid()
            )
            self.conf_head_f = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid()
            )
            self.conf_head_g = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid()
            )

        elif confidence_type == "channel":
            self.conf_head_s = nn.Sequential(
                nn.Linear(proj_dim, proj_dim), nn.ReLU(inplace=True), nn.Linear(proj_dim, proj_dim), nn.Sigmoid()
            )
            self.conf_head_f = nn.Sequential(
                nn.Linear(proj_dim, proj_dim), nn.ReLU(inplace=True), nn.Linear(proj_dim, proj_dim), nn.Sigmoid()
            )
            self.conf_head_g = nn.Sequential(
                nn.Linear(proj_dim, proj_dim), nn.ReLU(inplace=True), nn.Linear(proj_dim, proj_dim), nn.Sigmoid()
            )

        elif confidence_type == "uncertainty":
            self.unc_head_s = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, proj_dim), nn.Softplus()
            )
            self.unc_head_f = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, proj_dim), nn.Softplus()
            )
            self.unc_head_g = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, proj_dim), nn.Softplus()
            )

        elif confidence_type == "fuzzy":
            self.mu_head_s = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, proj_dim), nn.Sigmoid()
            )
            self.nu_head_s = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, proj_dim), nn.Sigmoid()
            )
            self.mu_head_f = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, proj_dim), nn.Sigmoid()
            )
            self.nu_head_f = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, proj_dim), nn.Sigmoid()
            )
            self.mu_head_g = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, proj_dim), nn.Sigmoid()
            )
            self.nu_head_g = nn.Sequential(
                nn.Linear(proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, proj_dim), nn.Sigmoid()
            )

    def _pool_features(self, f):
        return F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)

    def forward(self, x):
        fs, ff, fg = self.extract_expert_features(x)
        ps, pf, pg = self._pool_features(fs), self._pool_features(ff), self._pool_features(fg)

        if self.confidence_type == "scalar":
            cs = self.conf_head_s(ps).squeeze(-1)
            cf = self.conf_head_f(pf).squeeze(-1)
            cg = self.conf_head_g(pg).squeeze(-1)
            total = cs + cf + cg + 1e-8
            ws = (cs / total).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            wf = (cf / total).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            wg = (cg / total).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            fused = ws * fs + wf * ff + wg * fg
            return self.pool_and_classify(fused)

        if self.confidence_type == "channel":
            cs = self.conf_head_s(ps)
            cf = self.conf_head_f(pf)
            cg = self.conf_head_g(pg)
            total = cs + cf + cg + 1e-8
            ws = (cs / total).unsqueeze(2).unsqueeze(3)
            wf = (cf / total).unsqueeze(2).unsqueeze(3)
            wg = (cg / total).unsqueeze(2).unsqueeze(3)
            fused = ws * fs + wf * ff + wg * fg
            return self.pool_and_classify(fused)

        if self.confidence_type == "uncertainty":
            sigma_s = self.unc_head_s(ps)
            sigma_f = self.unc_head_f(pf)
            sigma_g = self.unc_head_g(pg)
            conf_s = 1.0 / (sigma_s + 1e-8)
            conf_f = 1.0 / (sigma_f + 1e-8)
            conf_g = 1.0 / (sigma_g + 1e-8)
            total = conf_s + conf_f + conf_g + 1e-8
            ws = (conf_s / total).unsqueeze(2).unsqueeze(3)
            wf = (conf_f / total).unsqueeze(2).unsqueeze(3)
            wg = (conf_g / total).unsqueeze(2).unsqueeze(3)
            fused = ws * fs + wf * ff + wg * fg
            return self.pool_and_classify(fused)

        if self.confidence_type == "fuzzy":
            mu_s = self.mu_head_s(ps)
            nu_s = self.nu_head_s(ps)
            pi_s = 1 - mu_s - nu_s

            mu_f = self.mu_head_f(pf)
            nu_f = self.nu_head_f(pf)
            pi_f = 1 - mu_f - nu_f

            mu_g = self.mu_head_g(pg)
            nu_g = self.nu_head_g(pg)
            pi_g = 1 - mu_g - nu_g

            r_s = mu_s * (1 - pi_s)
            r_f = mu_f * (1 - pi_f)
            r_g = mu_g * (1 - pi_g)

            total = r_s + r_f + r_g + 1e-8
            ws = (r_s / total).unsqueeze(2).unsqueeze(3)
            wf = (r_f / total).unsqueeze(2).unsqueeze(3)
            wg = (r_g / total).unsqueeze(2).unsqueeze(3)
            fused = ws * fs + wf * ff + wg * fg

            logits = self.pool_and_classify(fused)
            aux_loss = self.fuzzy_lambda * (
                torch.mean(torch.abs(mu_s + nu_s + pi_s - 1))
                + torch.mean(torch.abs(mu_f + nu_f + pi_f - 1))
                + torch.mean(torch.abs(mu_g + nu_g + pi_g - 1))
            )
            return {"logits": logits, "aux_loss": aux_loss}


def create_caef(
    num_classes=2,
    pretrained=True,
    attention=None,
    backbone1="resnet50",
    backbone2="mobilenetv2",
    backbone3="densenet121",
    proj_dim=256,
    confidence_type="scalar",
    expert_mode="multi_backbone",
):
    if expert_mode == "multi_layer":
        ml_expert = MultiLayerExpert(backbone1, pretrained)
        model = ConfidenceAwareExpertFusion(
            proj_dim=proj_dim,
            num_classes=num_classes,
            confidence_type=confidence_type,
            expert_mode="multi_layer",
            multi_layer_expert=ml_expert,
        )
    else:
        expert1 = SemanticExpert(backbone1, pretrained)
        expert2 = FrequencyExpert(backbone2, pretrained)
        expert3 = GeometryExpert(backbone3, pretrained)
        model = ConfidenceAwareExpertFusion(
            expert1=expert1,
            expert2=expert2,
            expert3=expert3,
            proj_dim=proj_dim,
            num_classes=num_classes,
            confidence_type=confidence_type,
            expert_mode="multi_backbone",
        )
    return model, "head"
