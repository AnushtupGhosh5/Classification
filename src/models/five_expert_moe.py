"""Single-image five-expert lesion MoE with one shared CNN backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lesion_moe import (
    BoundaryExpert,
    ColorExpert,
    DisagreementAwareRouter,
    HierarchicalBackbone,
    MorphologyExpert,
    SemanticExpert,
    TextureExpert,
)


EXPERT_NAMES = ("texture", "morphology", "semantic", "color", "boundary")


class ExpertClassifier(nn.Module):
    def __init__(self, proj_dim, num_classes, num_experts, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(proj_dim),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, num_classes),
            )
            for _ in range(num_experts)
        ])

    def forward(self, pooled):
        return torch.stack([
            head(pooled[:, index]) for index, head in enumerate(self.heads)
        ], dim=1)


class FiveExpertLesionMoE(nn.Module):
    """Route five complete lesion-specialized predictions per image."""

    expert_names = EXPERT_NAMES

    def __init__(
        self,
        backbone_name="convnext_tiny",
        pretrained=True,
        proj_dim=128,
        num_classes=7,
        branch_depth=1,
        expert_attention="hybrid",
        routing_mode="soft",
        router_hidden=128,
        router_dropout=0.1,
        router_temperature=0.7,
        expert_aux_weight=0.15,
        expert_diversity_weight=0.001,
        router_balance_weight=0.001,
        expert_dropout=0.05,
        classifier_dropout=0.25,
    ):
        super().__init__()
        self.feature_pyramid = HierarchicalBackbone(backbone_name, pretrained)
        self._backbone_module_name = "feature_pyramid"
        self.expert_loss_weight = float(expert_aux_weight)
        self.correction_loss_weight = 0.0
        self.expert_diversity_weight = float(expert_diversity_weight)
        self.router_balance_weight = float(router_balance_weight)
        self.expert_dropout = float(expert_dropout)
        self.router_full_lr = True
        self.router_gain_loss_weight = 0.0
        self.router_gain_temperature = 0.25

        early_channels, intermediate_channels, deep_channels = (
            self.feature_pyramid.channels
        )
        self.expert_modules = nn.ModuleDict({
            "texture": TextureExpert(
                early_channels, proj_dim, branch_depth, expert_attention,
            ),
            "morphology": MorphologyExpert(
                intermediate_channels, proj_dim, branch_depth,
                expert_attention,
            ),
            "semantic": SemanticExpert(
                deep_channels, proj_dim, branch_depth,
            ),
            "color": ColorExpert(
                proj_dim, branch_depth, expert_attention,
            ),
            "boundary": BoundaryExpert(
                early_channels, proj_dim, branch_depth, expert_attention,
            ),
        })
        self.router = DisagreementAwareRouter(
            proj_dim=proj_dim,
            num_experts=len(self.expert_names),
            hidden_dim=router_hidden,
            dropout=router_dropout,
            temperature=router_temperature,
            routing_mode=routing_mode,
        )
        self.head = ExpertClassifier(
            proj_dim, num_classes, len(self.expert_names), classifier_dropout,
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )

    def _recover_rgb(self, images):
        return (
            images * self.imagenet_std + self.imagenet_mean
        ).clamp(0.0, 1.0)

    def _structural_loss(self, probabilities, comparison):
        num_experts = probabilities.size(1)
        usage = probabilities.mean(dim=0)
        balance = (
            usage * (
                usage.clamp_min(1e-8).log()
                + usage.new_tensor(float(num_experts)).log()
            )
        ).sum()
        similarities = torch.bmm(comparison, comparison.transpose(1, 2))
        off_diagonal = ~torch.eye(
            num_experts, dtype=torch.bool, device=similarities.device,
        )
        diversity = similarities[:, off_diagonal].square().mean()
        return (
            self.router_balance_weight * balance
            + self.expert_diversity_weight * diversity
        )

    def forward(self, images):
        if images.dim() != 4:
            raise ValueError(
                "five_expert_moe expects [batch, channels, height, width]"
            )
        early, intermediate, deep = self.feature_pyramid(images)
        common_size = intermediate.shape[-2:]
        features = [
            self.expert_modules["texture"](early, common_size),
            self.expert_modules["morphology"](intermediate, common_size),
            self.expert_modules["semantic"](deep, common_size),
            self.expert_modules["color"](
                self._recover_rgb(images), common_size,
            ),
            self.expert_modules["boundary"](early, common_size),
        ]
        pooled = torch.stack([
            F.adaptive_avg_pool2d(feature, 1).flatten(1)
            for feature in features
        ], dim=1)
        weights, probabilities, comparison, disagreement = self.router(pooled)
        if self.training and self.expert_dropout > 0:
            keep = torch.rand_like(weights) >= self.expert_dropout
            empty = ~keep.any(dim=1)
            if empty.any():
                keep[empty, probabilities[empty].argmax(dim=1)] = True
            weights = weights * keep
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        expert_logits = self.head(pooled)
        semantic_logits = expert_logits[:, 2]
        logits = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)
        aux_loss = logits.new_zeros(())
        if self.training:
            aux_loss = self._structural_loss(probabilities, comparison)
        return {
            "logits": logits,
            "expert_logits": expert_logits,
            "router_weights": weights,
            "router_probabilities": probabilities,
            "router_active": (weights.detach() > 0).to(weights.dtype),
            "expert_embeddings": comparison,
            "expert_disagreement": disagreement,
            # Router-gain supervision treats the semantic expert as the
            # matched non-routed reference and learns when another expert helps.
            "baseline_logits": semantic_logits,
            "aux_loss": aux_loss,
        }


def create_five_expert_moe(
    num_classes=7,
    pretrained=True,
    attention=None,
    backbone1="convnext_tiny",
    proj_dim=128,
    branch_depth=1,
    expert_attention="hybrid",
    routing_mode="soft",
    router_hidden=128,
    router_dropout=0.1,
    router_temperature=0.7,
    expert_aux_weight=0.15,
    expert_diversity_weight=0.001,
    router_balance_weight=0.001,
    expert_dropout=0.05,
    classifier_dropout=0.25,
    router_gain_weight=0.0,
    router_gain_temperature=0.25,
    router_lr_scale=1.0,
    **_unused,
):
    model = FiveExpertLesionMoE(
        backbone_name=backbone1,
        pretrained=pretrained,
        proj_dim=proj_dim,
        num_classes=num_classes,
        branch_depth=branch_depth,
        expert_attention=expert_attention,
        routing_mode=routing_mode,
        router_hidden=router_hidden,
        router_dropout=router_dropout,
        router_temperature=router_temperature,
        expert_aux_weight=expert_aux_weight,
        expert_diversity_weight=expert_diversity_weight,
        router_balance_weight=router_balance_weight,
        expert_dropout=expert_dropout,
        classifier_dropout=classifier_dropout,
    )
    model.router_gain_loss_weight = float(router_gain_weight)
    model.router_gain_temperature = float(router_gain_temperature)
    model.router_lr_scale = float(router_lr_scale)
    return model, "head"
