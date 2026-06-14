import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_expert_fusion import BaseExpertFusion
from src.models.expert_branches import MultiLayerExpert


class ConfidenceAwareExpertFusion(BaseExpertFusion):
    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        proj_dim=256,
        num_classes=2,
        confidence_type="scalar",
        fuzzy_lambda=0.1,
        expert_mode="shared_base",
        multi_layer_expert=None,
    ):
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
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

    def _fuzzy_values(self, pooled, mu_head, nu_head):
        """Compute constrained fuzzy membership values.

        mu = sigma(raw_mu)
        nu = sigma(raw_nu) * (1 - mu)   -- enforces mu + nu <= 1
        pi = 1 - mu - nu                 -- guaranteed >= 0
        """
        mu = mu_head(pooled)
        nu = nu_head(pooled) * (1.0 - mu)
        pi = 1.0 - mu - nu
        return mu, nu, pi

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
            mu_s, nu_s, pi_s = self._fuzzy_values(ps, self.mu_head_s, self.nu_head_s)
            mu_f, nu_f, pi_f = self._fuzzy_values(pf, self.mu_head_f, self.nu_head_f)
            mu_g, nu_g, pi_g = self._fuzzy_values(pg, self.mu_head_g, self.nu_head_g)

            r_s = mu_s * (1 - pi_s)
            r_f = mu_f * (1 - pi_f)
            r_g = mu_g * (1 - pi_g)

            total = r_s + r_f + r_g + 1e-8
            ws = (r_s / total).unsqueeze(2).unsqueeze(3)
            wf = (r_f / total).unsqueeze(2).unsqueeze(3)
            wg = (r_g / total).unsqueeze(2).unsqueeze(3)
            fused = ws * fs + wf * ff + wg * fg

            logits = self.pool_and_classify(fused)

            # Reliability diversity loss: encourage experts to have
            # different confidence levels (promotes specialization)
            r_means = torch.stack([r_s.mean(), r_f.mean(), r_g.mean()])
            aux_loss = self.fuzzy_lambda * (-r_means.var())

            return {"logits": logits, "aux_loss": aux_loss}


def create_caef(
    num_classes=2,
    pretrained=True,
    attention=None,
    backbone1="resnet50",
    backbone2=None,
    backbone3=None,
    proj_dim=256,
    confidence_type="scalar",
    expert_mode="shared_base",
):
    if expert_mode == "multi_layer":
        ml_expert = MultiLayerExpert(backbone1, pretrained)
        model = ConfidenceAwareExpertFusion(
            backbone_name=backbone1,
            pretrained=pretrained,
            proj_dim=proj_dim,
            num_classes=num_classes,
            confidence_type=confidence_type,
            expert_mode="multi_layer",
            multi_layer_expert=ml_expert,
        )
    else:
        model = ConfidenceAwareExpertFusion(
            backbone_name=backbone1,
            pretrained=pretrained,
            proj_dim=proj_dim,
            num_classes=num_classes,
            confidence_type=confidence_type,
            expert_mode="shared_base",
        )
    return model, "head"
