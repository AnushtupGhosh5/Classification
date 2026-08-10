import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_expert_fusion import BaseExpertFusion
from src.models.expert_branches import MultiLayerExpert


class DisagreementAwareMixtureOfExperts(BaseExpertFusion):
    """Soft MoE whose router sees both expert features and disagreement.

    The expensive image backbone is shared. Semantic, frequency, and geometry
    branches remain lightweight and each owns a classifier. A soft per-image
    router combines expert predictions and the fused-feature prediction.
    """

    def __init__(
        self,
        backbone_name="efficientnet_b0",
        pretrained=True,
        proj_dim=128,
        num_classes=7,
        expert_mode="shared_base",
        multi_layer_expert=None,
        branch_depth=1,
        disagreement_type="abs",
        router_hidden=128,
        router_dropout=0.1,
        router_temperature=1.0,
        disagreement_scale=1.0,
        load_balance_weight=0.01,
        diversity_weight=0.01,
        expert_loss_weight=0.2,
        expert_vote_weight=0.5,
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
        self.disagreement_type = disagreement_type
        self.router_temperature = router_temperature
        self.disagreement_scale = disagreement_scale
        self.load_balance_weight = load_balance_weight
        self.diversity_weight = diversity_weight
        self.expert_loss_weight = expert_loss_weight
        self.expert_vote_weight = expert_vote_weight

        self.expert_heads = nn.ModuleList([
            nn.Linear(proj_dim, num_classes) for _ in range(3)
        ])
        if disagreement_type == "learnable":
            self.disagreement_estimators = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(proj_dim * 2, router_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(router_hidden, 1),
                    nn.Softplus(),
                )
                for _ in range(3)
            ])
        self.router = nn.Sequential(
            nn.Linear(proj_dim * 3 + 3, router_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(router_dropout),
            nn.Linear(router_hidden, 3),
        )

    @staticmethod
    def _pool(feature):
        return F.adaptive_avg_pool2d(feature, 1).flatten(1)

    def _pair_disagreement(self, first, second, pair_index):
        if self.disagreement_type == "abs":
            return torch.abs(first - second).mean(dim=(1, 2, 3))
        if self.disagreement_type == "cosine":
            spatial_similarity = F.cosine_similarity(first, second, dim=1)
            return 1.0 - spatial_similarity.mean(dim=(1, 2))
        if self.disagreement_type == "learnable":
            pair_features = torch.cat(
                [self._pool(first), self._pool(second)],
                dim=1,
            )
            return self.disagreement_estimators[pair_index](
                pair_features
            ).squeeze(1)
        raise ValueError(
            f"Unsupported disagreement type: {self.disagreement_type}"
        )

    def _structural_aux_loss(self, pooled, weights):
        # Encourage balanced aggregate utilization without forcing every image
        # to use all experts equally.
        mean_usage = weights.mean(dim=0)
        target = torch.full_like(mean_usage, 1.0 / len(pooled))
        balance_loss = (mean_usage - target).pow(2).sum()

        # Penalize redundant intermediate representations. Squared cosine
        # targets orthogonality rather than arbitrary opposite predictions.
        normalized = [F.normalize(feature, dim=1) for feature in pooled]
        diversity_loss = (
            (normalized[0] * normalized[1]).sum(dim=1).pow(2).mean()
            + (normalized[0] * normalized[2]).sum(dim=1).pow(2).mean()
            + (normalized[1] * normalized[2]).sum(dim=1).pow(2).mean()
        ) / 3.0
        return (
            self.load_balance_weight * balance_loss
            + self.diversity_weight * diversity_loss
        )

    def forward(self, x):
        features = list(self.extract_expert_features(x))
        pooled = [self._pool(feature) for feature in features]

        disagreements = torch.stack([
            self._pair_disagreement(features[0], features[1], 0),
            self._pair_disagreement(features[0], features[2], 1),
            self._pair_disagreement(features[1], features[2], 2),
        ], dim=1)
        router_input = torch.cat(
            pooled + [self.disagreement_scale * disagreements],
            dim=1,
        )
        temperature = max(float(self.router_temperature), 1e-4)
        weights = F.softmax(self.router(router_input) / temperature, dim=1)

        expert_logits = torch.stack([
            head(feature) for head, feature in zip(self.expert_heads, pooled)
        ], dim=1)
        routed_logits = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)

        fused_feature = sum(
            weights[:, index, None, None, None] * feature
            for index, feature in enumerate(features)
        )
        fused_logits = self.pool_and_classify(fused_feature)
        logits = fused_logits + self.expert_vote_weight * routed_logits

        aux_loss = logits.new_zeros(())
        if self.training:
            aux_loss = self._structural_aux_loss(pooled, weights)

        return {
            "logits": logits,
            "expert_logits": expert_logits,
            "router_weights": weights,
            "aux_loss": aux_loss,
        }


def create_moe_edf(
    num_classes=2,
    pretrained=True,
    attention=None,
    backbone1="efficientnet_b0",
    backbone2=None,
    backbone3=None,
    proj_dim=128,
    expert_mode="shared_base",
    branch_depth=1,
    disagreement_type="abs",
    router_hidden=128,
    router_dropout=0.1,
    router_temperature=1.0,
    disagreement_scale=1.0,
    load_balance_weight=0.01,
    diversity_weight=0.01,
    expert_loss_weight=0.2,
    expert_vote_weight=0.5,
):
    common = dict(
        backbone_name=backbone1,
        pretrained=pretrained,
        proj_dim=proj_dim,
        num_classes=num_classes,
        branch_depth=branch_depth,
        disagreement_type=disagreement_type,
        router_hidden=router_hidden,
        router_dropout=router_dropout,
        router_temperature=router_temperature,
        disagreement_scale=disagreement_scale,
        load_balance_weight=load_balance_weight,
        diversity_weight=diversity_weight,
        expert_loss_weight=expert_loss_weight,
        expert_vote_weight=expert_vote_weight,
    )
    if expert_mode == "multi_layer":
        common.update(
            expert_mode="multi_layer",
            multi_layer_expert=MultiLayerExpert(backbone1, pretrained),
        )
    else:
        common.update(expert_mode="shared_base")
    return DisagreementAwareMixtureOfExperts(**common), "head"
