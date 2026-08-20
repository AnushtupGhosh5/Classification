"""MILK10k paired-modality lesion-specialized mixture of experts.

One shared CNN is evaluated for the clinical and dermoscopic images (the two
modalities are concatenated along the batch dimension).  Five specialists use
different inputs/stages and a disagreement-aware router forms a per-lesion
convex mixture of their class predictions.  The semantic expert is a routed
expert, not an external residual baseline.
"""

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


class PairedColorExpert(nn.Module):
    """Use both raw modalities without introducing another backbone."""

    def __init__(self, proj_dim, depth, attention):
        super().__init__()
        # The established chromatic expert is shared across modalities; a
        # learned gate then selects clinical versus dermoscopic evidence.
        self.shared = ColorExpert(proj_dim, depth, attention)
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.SiLU(inplace=True),
            nn.Linear(proj_dim, 2),
        )

    def forward(self, clinical, dermoscopic, common_size):
        batch = clinical.size(0)
        features = self.shared(
            torch.cat((clinical, dermoscopic), dim=0), common_size,
        )
        clinical_feature, dermoscopic_feature = features.split(batch, dim=0)
        descriptors = torch.cat((
            F.adaptive_avg_pool2d(clinical_feature, 1).flatten(1),
            F.adaptive_avg_pool2d(dermoscopic_feature, 1).flatten(1),
        ), dim=1)
        weights = F.softmax(self.gate(descriptors), dim=1)
        return (
            weights[:, 0, None, None, None] * clinical_feature
            + weights[:, 1, None, None, None] * dermoscopic_feature
        )


class PairedSemanticExpert(nn.Module):
    def __init__(self, in_channels, proj_dim, depth):
        super().__init__()
        self.shared = SemanticExpert(in_channels, proj_dim, depth)
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.SiLU(inplace=True),
            nn.Linear(proj_dim, 2),
        )

    def forward(self, clinical, dermoscopic, common_size):
        batch = clinical.size(0)
        feature = self.shared(
            torch.cat((clinical, dermoscopic), dim=0), common_size,
        )
        clinical_feature, dermoscopic_feature = feature.split(batch, dim=0)
        descriptors = torch.cat((
            F.adaptive_avg_pool2d(clinical_feature, 1).flatten(1),
            F.adaptive_avg_pool2d(dermoscopic_feature, 1).flatten(1),
        ), dim=1)
        weights = F.softmax(self.gate(descriptors), dim=1)
        return (
            weights[:, 0, None, None, None] * clinical_feature
            + weights[:, 1, None, None, None] * dermoscopic_feature
        )


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


class PairedLesionMoE(nn.Module):
    paired_input = True

    def __init__(
        self,
        backbone_name="convnext_tiny",
        pretrained=True,
        proj_dim=128,
        num_classes=11,
        branch_depth=1,
        expert_attention="hybrid",
        routing_mode="soft",
        router_hidden=128,
        router_dropout=0.1,
        router_temperature=0.7,
        expert_aux_weight=0.15,
        expert_diversity_weight=0.005,
        router_balance_weight=0.002,
        expert_dropout=0.0,
        classifier_dropout=0.25,
        baseline_only=False,
        enabled_experts=EXPERT_NAMES,
    ):
        super().__init__()
        requested = tuple(str(name).lower() for name in enabled_experts)
        unknown = sorted(set(requested) - set(EXPERT_NAMES))
        if unknown:
            raise ValueError(f"Unknown paired expert ablation names: {unknown}")
        if len(requested) != len(set(requested)):
            raise ValueError("enabled_experts must not contain duplicates")
        self.expert_names = tuple(
            name for name in EXPERT_NAMES if name in requested
        )
        if len(self.expert_names) < 2:
            raise ValueError("At least two paired experts are required")
        if baseline_only and "semantic" not in self.expert_names:
            raise ValueError("paired baseline-only mode requires semantic expert")
        self.feature_pyramid = HierarchicalBackbone(backbone_name, pretrained)
        self._backbone_module_name = "feature_pyramid"
        self.backbone_name = backbone_name
        self.proj_dim = proj_dim
        self.baseline_only = bool(baseline_only)
        self.expert_loss_weight = float(expert_aux_weight)
        if self.baseline_only:
            self.expert_loss_weight = 0.0
        self.correction_loss_weight = 0.0
        self.expert_diversity_weight = float(expert_diversity_weight)
        self.router_balance_weight = float(router_balance_weight)
        self.expert_dropout = float(expert_dropout)
        self.router_full_lr = True
        self.router_gain_loss_weight = 0.0
        self.router_gain_temperature = 0.25
        self.register_buffer(
            "training_epoch_state", torch.tensor(0, dtype=torch.long),
            persistent=True,
        )

        early_channels, intermediate_channels, deep_channels = self.feature_pyramid.channels
        available_modules = {
            "texture": lambda: TextureExpert(
                early_channels, proj_dim, branch_depth, expert_attention,
            ),
            "morphology": lambda: MorphologyExpert(
                intermediate_channels, proj_dim, branch_depth, expert_attention,
            ),
            "semantic": lambda: PairedSemanticExpert(
                deep_channels, proj_dim, branch_depth,
            ),
            "color": lambda: PairedColorExpert(
                proj_dim, branch_depth, expert_attention,
            ),
            "boundary": lambda: BoundaryExpert(
                early_channels, proj_dim, branch_depth, expert_attention,
            ),
        }
        self.expert_modules = nn.ModuleDict({
            name: available_modules[name]() for name in self.expert_names
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
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )

    def set_training_epoch(self, epoch):
        self.training_epoch_state.fill_(int(epoch))

    def _recover_rgb(self, images):
        return (images * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)

    def _structural_loss(self, probabilities, comparison):
        num_experts = probabilities.size(1)
        usage = probabilities.mean(dim=0)
        balance = (
            usage * (usage.clamp_min(1e-8).log() + usage.new_tensor(float(num_experts)).log())
        ).sum()
        similarities = torch.bmm(comparison, comparison.transpose(1, 2))
        off_diagonal = ~torch.eye(
            num_experts, dtype=torch.bool, device=similarities.device,
        )
        diversity = similarities[:, off_diagonal].square().mean()
        return self.router_balance_weight * balance + self.expert_diversity_weight * diversity

    def forward(self, images):
        if images.dim() != 5 or images.size(1) != 2:
            raise ValueError(
                "paired_lesion_moe expects [batch, 2, channels, height, width]"
            )
        batch = images.size(0)
        clinical, dermoscopic = images[:, 0], images[:, 1]
        both = torch.cat((clinical, dermoscopic), dim=0)
        early, intermediate, deep = self.feature_pyramid(both)
        clinical_early, dermoscopic_early = early.split(batch, dim=0)
        _, dermoscopic_intermediate = intermediate.split(batch, dim=0)
        clinical_deep, dermoscopic_deep = deep.split(batch, dim=0)
        common_size = dermoscopic_intermediate.shape[-2:]

        clinical_rgb = self._recover_rgb(clinical)
        dermoscopic_rgb = self._recover_rgb(dermoscopic)
        features_by_name = {}
        if "texture" in self.expert_modules:
            # Dermoscopy preserves the local microstructure needed here.
            features_by_name["texture"] = self.expert_modules["texture"](
                dermoscopic_early, common_size,
            )
        if "morphology" in self.expert_modules:
            features_by_name["morphology"] = self.expert_modules["morphology"](
                dermoscopic_intermediate, common_size,
            )
        if "semantic" in self.expert_modules:
            features_by_name["semantic"] = self.expert_modules["semantic"](
                clinical_deep, dermoscopic_deep, common_size,
            )
        if "color" in self.expert_modules:
            features_by_name["color"] = self.expert_modules["color"](
                clinical_rgb, dermoscopic_rgb, common_size,
            )
        if "boundary" in self.expert_modules:
            # Clinical framing retains useful macroscopic border/asymmetry.
            features_by_name["boundary"] = self.expert_modules["boundary"](
                clinical_early, common_size,
            )
        features = [features_by_name[name] for name in self.expert_names]
        pooled = torch.stack([
            F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in features
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
        logits = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)
        baseline_logits = None
        if "semantic" in self.expert_names:
            # The control path is exactly the semantic expert from this same
            # model, so paired baseline comparisons do not confound heads.
            semantic_index = self.expert_names.index("semantic")
            baseline_logits = expert_logits[:, semantic_index]
            if self.baseline_only:
                logits = baseline_logits
        aux_loss = logits.new_zeros(())
        if self.training and not self.baseline_only:
            aux_loss = self._structural_loss(probabilities, comparison)
        result = {
            "logits": logits,
            "expert_logits": expert_logits,
            "router_weights": weights,
            "router_probabilities": probabilities,
            "router_active": (weights.detach() > 0).to(weights.dtype),
            "expert_embeddings": comparison,
            "expert_disagreement": disagreement,
            "aux_loss": aux_loss,
        }
        if baseline_logits is not None:
            result["baseline_logits"] = baseline_logits
        return result


def create_paired_lesion_moe(
    num_classes=11,
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
    expert_diversity_weight=0.005,
    router_balance_weight=0.002,
    expert_dropout=0.0,
    classifier_dropout=0.25,
    paired_baseline_only=False,
    router_gain_weight=0.0,
    router_gain_temperature=0.25,
    router_lr_scale=1.0,
    enabled_experts=EXPERT_NAMES,
    **_unused,
):
    model = PairedLesionMoE(
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
        baseline_only=paired_baseline_only,
        enabled_experts=enabled_experts,
    )
    model.router_gain_loss_weight = float(router_gain_weight)
    model.router_gain_temperature = float(router_gain_temperature)
    model.router_lr_scale = float(router_lr_scale)
    return model, "head"
