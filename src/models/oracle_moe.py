"""Oracle-first residual experts with explicitly staged routing.

This diagnostic model answers two questions separately:

1. Do frozen-backbone specialist corrections contain complementary signal?
2. Can a learned router recover that signal while choosing no correction when
   every specialist is harmful?

Experts are trained first while the final prediction remains the immutable
baseline. They are then frozen and only the gain router is trained.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lesion_moe import LesionExpertMoE


class GainRouter(nn.Module):
    """Predict expert gains relative to a fixed zero-logit baseline route."""

    def __init__(self, proj_dim, num_experts, hidden_dim=128, dropout=0.1,
                 temperature=1.0):
        super().__init__()
        self.temperature = float(temperature)
        comparison_dim = max(32, proj_dim // 2)
        identity_dim = min(16, hidden_dim)
        self.comparison_projection = nn.Sequential(
            nn.Linear(proj_dim, comparison_dim, bias=False),
            nn.LayerNorm(comparison_dim),
        )
        self.expert_embedding = nn.Parameter(
            torch.empty(num_experts, identity_dim),
        )
        nn.init.normal_(self.expert_embedding, std=0.02)
        router_input_dim = proj_dim * 2 + 2 + identity_dim
        self.scorer = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Initially all expert gains equal the fixed no-correction gain of 0.
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, pooled):
        comparison = F.normalize(self.comparison_projection(pooled), dim=-1)
        num_experts = pooled.shape[1]
        off_diagonal = 1.0 - torch.eye(
            num_experts, device=pooled.device, dtype=pooled.dtype,
        )
        cosine = torch.bmm(comparison, comparison.transpose(1, 2))
        cosine_disagreement = (
            (1.0 - cosine) * off_diagonal
        ).sum(dim=2) / max(num_experts - 1, 1)
        absolute = torch.abs(
            comparison.unsqueeze(2) - comparison.unsqueeze(1)
        ).mean(dim=-1)
        absolute_disagreement = (
            absolute * off_diagonal
        ).sum(dim=2) / max(num_experts - 1, 1)
        disagreement = torch.stack(
            (cosine_disagreement, absolute_disagreement), dim=-1,
        )

        context = pooled.mean(dim=1, keepdim=True).expand_as(pooled)
        identity = self.expert_embedding.unsqueeze(0).expand(
            pooled.shape[0], -1, -1,
        )
        expert_gain_logits = self.scorer(torch.cat(
            (pooled, disagreement, context, identity), dim=-1,
        )).squeeze(-1)
        no_correction_logit = torch.zeros_like(expert_gain_logits[:, :1])
        route_logits = torch.cat(
            (no_correction_logit, expert_gain_logits), dim=1,
        )
        probabilities = F.softmax(
            route_logits / max(self.temperature, 1e-4), dim=1,
        )
        return probabilities[:, 1:], probabilities, comparison, disagreement


class EvidenceGainRouter(GainRouter):
    """Route from baseline uncertainty and observable correction evidence."""

    evidence_dim = 10
    baseline_stat_dim = 3

    def __init__(self, proj_dim, num_experts, num_classes, hidden_dim=128,
                 dropout=0.1, temperature=1.0, class_aware=False):
        nn.Module.__init__(self)
        self.temperature = float(temperature)
        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.class_aware = bool(class_aware)
        # Confidence/entropy alone discards which class each candidate favors.
        # That is especially destructive on multi-class datasets such as
        # MILK10k. V3 retains the full baseline distribution, candidate
        # distribution, and signed probability change for every route.
        self.evidence_dim = 10 + (
            3 * self.num_classes if self.class_aware else 0
        )
        comparison_dim = max(32, proj_dim // 2)
        identity_dim = min(16, hidden_dim)
        self.comparison_projection = nn.Sequential(
            nn.Linear(proj_dim, comparison_dim, bias=False),
            nn.LayerNorm(comparison_dim),
        )
        # Includes route 0 (no correction), unlike the v1 expert-only identity.
        self.route_embedding = nn.Parameter(
            torch.empty(num_experts + 1, identity_dim),
        )
        nn.init.normal_(self.route_embedding, std=0.02)
        router_input_dim = (
            proj_dim * 2 + 2 + self.evidence_dim
            + self.baseline_stat_dim + identity_dim
        )
        self.scorer = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def _prediction_stats(self, logits):
        probabilities = F.softmax(logits, dim=-1)
        top2 = probabilities.topk(min(2, self.num_classes), dim=-1).values
        confidence = top2[..., 0]
        margin = (
            top2[..., 0] - top2[..., 1]
            if top2.shape[-1] > 1 else top2[..., 0]
        )
        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1) / max(math.log(float(self.num_classes)), 1e-8)
        return probabilities, confidence, entropy, margin

    def forward(self, pooled, baseline_logits, expert_delta_logits):
        comparison = F.normalize(self.comparison_projection(pooled), dim=-1)
        num_experts = pooled.shape[1]
        off_diagonal = 1.0 - torch.eye(
            num_experts, device=pooled.device, dtype=pooled.dtype,
        )
        cosine = torch.bmm(comparison, comparison.transpose(1, 2))
        cosine_disagreement = (
            (1.0 - cosine) * off_diagonal
        ).sum(dim=2) / max(num_experts - 1, 1)
        absolute = torch.abs(
            comparison.unsqueeze(2) - comparison.unsqueeze(1)
        ).mean(dim=-1)
        absolute_disagreement = (
            absolute * off_diagonal
        ).sum(dim=2) / max(num_experts - 1, 1)
        expert_disagreement = torch.stack(
            (cosine_disagreement, absolute_disagreement), dim=-1,
        )

        # Router evidence is observational. Detaching prevents phase mistakes
        # from teaching experts to manipulate their own routing statistics.
        baseline_logits = baseline_logits.detach()
        expert_delta_logits = expert_delta_logits.detach()
        candidate_logits = baseline_logits.unsqueeze(1) + expert_delta_logits
        baseline_prob, base_conf, base_entropy, base_margin = (
            self._prediction_stats(baseline_logits)
        )
        candidate_prob, cand_conf, cand_entropy, cand_margin = (
            self._prediction_stats(candidate_logits)
        )
        probability_shift = torch.abs(
            candidate_prob - baseline_prob.unsqueeze(1)
        ).mean(dim=-1)
        candidate_kl = (
            candidate_prob * (
                candidate_prob.clamp_min(1e-8).log()
                - baseline_prob.unsqueeze(1).clamp_min(1e-8).log()
            )
        ).sum(dim=-1)
        correction_magnitude = expert_delta_logits.abs().mean(dim=-1)
        same_prediction = (
            candidate_logits.argmax(dim=-1)
            == baseline_logits.argmax(dim=-1).unsqueeze(1)
        ).to(candidate_logits.dtype)
        expert_evidence = torch.stack((
            cand_conf,
            cand_entropy,
            cand_margin,
            cand_conf - base_conf.unsqueeze(1),
            cand_entropy - base_entropy.unsqueeze(1),
            cand_margin - base_margin.unsqueeze(1),
            probability_shift,
            candidate_kl,
            correction_magnitude,
            same_prediction,
        ), dim=-1)
        if self.class_aware:
            expanded_baseline_prob = baseline_prob.unsqueeze(1).expand(
                -1, num_experts, -1,
            )
            expert_evidence = torch.cat((
                expert_evidence,
                expanded_baseline_prob,
                candidate_prob,
                candidate_prob - expanded_baseline_prob,
            ), dim=-1)

        batch_size = pooled.shape[0]
        global_context = pooled.mean(dim=1)
        no_correction_evidence = torch.stack((
            base_conf,
            base_entropy,
            base_margin,
            torch.zeros_like(base_conf),
            torch.zeros_like(base_conf),
            torch.zeros_like(base_conf),
            torch.zeros_like(base_conf),
            torch.zeros_like(base_conf),
            torch.zeros_like(base_conf),
            torch.ones_like(base_conf),
        ), dim=-1).unsqueeze(1)
        if self.class_aware:
            no_correction_evidence = torch.cat((
                no_correction_evidence,
                baseline_prob.unsqueeze(1),
                baseline_prob.unsqueeze(1),
                torch.zeros_like(baseline_prob).unsqueeze(1),
            ), dim=-1)
        route_evidence = torch.cat(
            (no_correction_evidence, expert_evidence), dim=1,
        )
        route_local = torch.cat(
            (global_context.unsqueeze(1), pooled), dim=1,
        )
        route_context = global_context.unsqueeze(1).expand(
            -1, num_experts + 1, -1,
        )
        no_disagreement = torch.zeros(
            batch_size, 1, 2, device=pooled.device, dtype=pooled.dtype,
        )
        route_disagreement = torch.cat(
            (no_disagreement, expert_disagreement), dim=1,
        )
        baseline_stats = torch.stack(
            (base_conf, base_entropy, base_margin), dim=-1,
        ).unsqueeze(1).expand(-1, num_experts + 1, -1)
        identity = self.route_embedding.unsqueeze(0).expand(
            batch_size, -1, -1,
        )
        route_input = torch.cat((
            route_local,
            route_context,
            route_disagreement,
            route_evidence,
            baseline_stats,
            identity,
        ), dim=-1)
        route_logits = self.scorer(route_input).squeeze(-1)
        probabilities = F.softmax(
            route_logits / max(self.temperature, 1e-4), dim=1,
        )
        return (
            probabilities[:, 1:], probabilities, comparison,
            expert_disagreement, route_logits,
        )


class OracleResidualMoE(LesionExpertMoE):
    """Two-stage experts followed by a baseline-aware gain router."""

    oracle_protocol = True
    route_names = (
        "no_correction", "texture", "morphology", "color", "boundary",
    )
    router_full_lr = True

    def __init__(self, *args, expert_pretrain_epochs=10,
                 oracle_router_version="v1", router_lr_scale=1.0, **kwargs):
        kwargs["protect_baseline"] = True
        kwargs["expert_warmup_epochs"] = expert_pretrain_epochs
        super().__init__(*args, **kwargs)
        self.expert_pretrain_epochs = int(expert_pretrain_epochs)
        self.oracle_router_version = oracle_router_version
        self.router_lr_scale = float(router_lr_scale)
        router_kwargs = dict(
            proj_dim=self.proj_dim,
            num_experts=len(self.expert_names),
            hidden_dim=kwargs.get("router_hidden", 128),
            dropout=kwargs.get("router_dropout", 0.1),
            temperature=kwargs.get("router_temperature", 1.0),
        )
        if oracle_router_version in ("v2", "v3"):
            self.router = EvidenceGainRouter(
                num_classes=kwargs.get("num_classes", 2),
                class_aware=oracle_router_version == "v3",
                **router_kwargs,
            )
        elif oracle_router_version == "v1":
            self.router = GainRouter(**router_kwargs)
        else:
            raise ValueError(
                f"Unknown oracle router version: {oracle_router_version}"
            )
        # Non-persistent inference buffers keep old checkpoints compatible.
        # They are fitted deterministically on validation logits and recorded
        # in a separate JSON artifact, never learned from the official test.
        self.register_buffer(
            "static_expert_weights",
            torch.full((len(self.expert_names),), 1.0 / len(self.expert_names)),
            persistent=False,
        )
        self.register_buffer(
            "static_correction_alpha", torch.tensor(0.0), persistent=False,
        )
        self._static_fusion_enabled = False
        self._oracle_phase = "expert"

    def configure_static_fusion(self, alpha, expert_weights):
        """Use a global baseline-safe residual mixture at inference.

        ``alpha=0`` reproduces the protected semantic baseline exactly.
        Expert weights are normalized here so malformed configuration cannot
        silently amplify the correction.
        """
        weights = torch.as_tensor(
            expert_weights,
            device=self.static_expert_weights.device,
            dtype=self.static_expert_weights.dtype,
        )
        if weights.numel() != len(self.expert_names):
            raise ValueError(
                f"Expected {len(self.expert_names)} expert weights, got "
                f"{weights.numel()}"
            )
        weights = weights.clamp_min(0)
        weights = weights / weights.sum().clamp_min(1e-8)
        self.static_expert_weights.copy_(weights)
        self.static_correction_alpha.fill_(float(max(0.0, min(1.0, alpha))))
        self._static_fusion_enabled = True

    def clear_static_fusion(self):
        self._static_fusion_enabled = False

    @staticmethod
    def _set_module_trainable(module, trainable):
        for parameter in module.parameters():
            parameter.requires_grad = bool(trainable)

    def configure_expert_pretraining(self):
        self._oracle_phase = "expert"
        self._set_module_trainable(self.feature_pyramid, False)
        self._set_module_trainable(self.baseline_norm, False)
        self._set_module_trainable(self.baseline_classifier, False)
        self._set_module_trainable(self.expert_modules, True)
        self._set_module_trainable(self.head, True)
        self._set_module_trainable(self.router, False)
        self._set_module_trainable(self.correction_gate, False)

    def configure_router_training(self):
        self._oracle_phase = "router"
        self._set_module_trainable(self.feature_pyramid, False)
        self._set_module_trainable(self.baseline_norm, False)
        self._set_module_trainable(self.baseline_classifier, False)
        self._set_module_trainable(self.expert_modules, False)
        self._set_module_trainable(self.head, False)
        self._set_module_trainable(self.router, True)
        self._set_module_trainable(self.correction_gate, False)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            # Frozen means deterministic: preserve BatchNorm buffers and turn
            # off stochastic depth/dropout in every protected module.
            self.feature_pyramid.eval()
            self.baseline_norm.eval()
            self.baseline_classifier.eval()
            if self._oracle_phase == "router":
                self.expert_modules.eval()
                self.head.eval()
        return self

    def forward(self, x):
        early, intermediate, deep = self.feature_pyramid(x)
        common_size = intermediate.shape[-2:]
        rgb = self._recover_rgb(x)

        baseline_deep = (
            F.relu(deep) if self.backbone_name == "densenet121" else deep
        )
        baseline_pooled = F.adaptive_avg_pool2d(
            baseline_deep, 1,
        ).flatten(1)
        baseline_logits = self.baseline_classifier(
            self.baseline_norm(baseline_pooled),
        )

        features = (
            self.expert_modules["texture"](early, common_size),
            self.expert_modules["morphology"](intermediate, common_size),
            self.expert_modules["color"](rgb, common_size),
            self.expert_modules["boundary"](early, common_size),
        )
        pooled = torch.stack([
            F.adaptive_avg_pool2d(feature, 1).flatten(1)
            for feature in features
        ], dim=1)
        expert_delta_logits = self.head(pooled)
        if self.oracle_router_version in ("v2", "v3"):
            (
                correction_weights, probabilities, comparison,
                disagreement, router_logits,
            ) = self.router(pooled, baseline_logits, expert_delta_logits)
        else:
            correction_weights, probabilities, comparison, disagreement = (
                self.router(pooled)
            )
            router_logits = probabilities.clamp_min(1e-8).log()
        expert_logits = baseline_logits.unsqueeze(1) + expert_delta_logits
        uniform_correction = expert_delta_logits.mean(dim=1)
        correction_logits = baseline_logits + uniform_correction

        training_epoch = int(self.training_epoch_state.item())
        expert_phase = (
            0 < training_epoch <= self.expert_pretrain_epochs
        )
        if expert_phase:
            probabilities = torch.full_like(
                probabilities, 1.0 / probabilities.shape[1],
            )
            correction_weights = probabilities[:, 1:]
            logits = baseline_logits
            correction_scale = torch.zeros_like(probabilities[:, 0])
        elif self._static_fusion_enabled:
            alpha = self.static_correction_alpha.to(expert_delta_logits)
            static_weights = self.static_expert_weights.to(
                expert_delta_logits,
            ).unsqueeze(0).expand(expert_delta_logits.size(0), -1)
            correction_weights = alpha * static_weights
            probabilities = torch.cat((
                (1.0 - alpha).expand(expert_delta_logits.size(0), 1),
                correction_weights,
            ), dim=1)
            routed_delta = (
                correction_weights.unsqueeze(-1) * expert_delta_logits
            ).sum(dim=1)
            logits = baseline_logits + routed_delta
            correction_scale = alpha.expand(expert_delta_logits.size(0))
            router_logits = probabilities.clamp_min(1e-8).log()
        else:
            routed_delta = (
                correction_weights.unsqueeze(-1) * expert_delta_logits
            ).sum(dim=1)
            logits = baseline_logits + routed_delta
            correction_scale = 1.0 - probabilities[:, 0]

        return {
            "logits": logits,
            "baseline_logits": baseline_logits,
            "expert_logits": expert_logits,
            "expert_delta_logits": expert_delta_logits,
            "correction_logits": correction_logits,
            "correction_scale": correction_scale,
            "router_weights": probabilities,
            "router_probabilities": probabilities,
            "router_logits": router_logits,
            "router_active": (probabilities.detach() > 0).to(probabilities.dtype),
            "expert_embeddings": comparison,
            "expert_disagreement": disagreement,
            "router_gain_enabled": not expert_phase,
            # V2's balanced hard labels encouraged a majority-expert shortcut.
            # V3 instead distils the complete loss-derived oracle distribution,
            # matching the soft mixture used at inference.
            "hard_oracle_router": self.oracle_router_version == "v2",
            "aux_loss": logits.new_zeros(()),
        }


def create_oracle_moe(num_classes=2, pretrained=True, attention=None,
                      backbone1="efficientnet_b0", backbone2=None,
                      backbone3=None, proj_dim=128, branch_depth=1,
                      routing_mode="soft", router_hidden=128,
                      router_dropout=0.1, router_temperature=1.0,
                      expert_aux_weight=0.15,
                      expert_diversity_weight=0.0,
                      router_balance_weight=0.0,
                      expert_warmup_epochs=10, expert_dropout=0.0,
                      correction_aux_weight=0.1,
                      correction_gate_init=0.0,
                      protect_baseline=True,
                      correction_max_scale=1.0,
                      correction_ramp_epochs=0,
                      residual_distill_weight=0.0,
                      router_gain_weight=0.2,
                      router_gain_temperature=0.25,
                      expert_pretrain_epochs=10,
                      oracle_router_version="v1",
                      router_lr_scale=1.0):
    model = OracleResidualMoE(
        backbone_name=backbone1,
        pretrained=pretrained,
        proj_dim=proj_dim,
        num_classes=num_classes,
        branch_depth=branch_depth,
        routing_mode=routing_mode,
        router_hidden=router_hidden,
        router_dropout=router_dropout,
        router_temperature=router_temperature,
        expert_aux_weight=expert_aux_weight,
        expert_diversity_weight=expert_diversity_weight,
        router_balance_weight=router_balance_weight,
        expert_warmup_epochs=expert_pretrain_epochs,
        expert_dropout=0.0,
        correction_aux_weight=correction_aux_weight,
        correction_gate_init=0.0,
        correction_max_scale=1.0,
        correction_ramp_epochs=0,
        residual_distill_weight=residual_distill_weight,
        expert_pretrain_epochs=expert_pretrain_epochs,
        oracle_router_version=oracle_router_version,
        router_lr_scale=router_lr_scale,
    )
    model.router_gain_loss_weight = router_gain_weight
    model.router_gain_temperature = router_gain_temperature
    return model, "head"
