"""Hierarchical lesion-specialized mixture of experts.

The model runs one shared CNN and sends deliberately chosen hierarchy levels
to texture, morphology, and semantic specialists.  Color and boundary experts
retain input/early spatial information.  A sample-level router uses compact
expert disagreement descriptors to weight the actual expert feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone_extractor import create_backbone


EXPERT_NAMES = ("texture", "morphology", "semantic", "color", "boundary")

# These stages are semantic choices, not equal-length guesses. EfficientNet
# stage 2 retains fine detail, stage 5 is the last 1/16 stage, and stage 8 is
# the final representation. ResNet/DenseNet mappings follow the same intent.
_PYRAMID_TARGETS = {
    "efficientnet_b0": (("features", "2"), ("features", "5"), ("features", "8")),
    "efficientnet_b1": (("features", "2"), ("features", "5"), ("features", "8")),
    "efficientnet_b2": (("features", "2"), ("features", "5"), ("features", "8")),
    "resnet34": (("layer1",), ("layer2",), ("layer4",)),
    "resnet50": (("layer1",), ("layer2",), ("layer4",)),
    "resnet101": (("layer1",), ("layer2",), ("layer4",)),
    "densenet121": (
        ("features", "transition1"),
        ("features", "transition2"),
        ("features", "norm5"),
    ),
}


def _resolve_module(root, path):
    module = root
    for component in path:
        if component.isdigit():
            module = module[int(component)]
        else:
            module = getattr(module, component)
    return module


class HierarchicalBackbone(nn.Module):
    """Expose early, intermediate, and deep maps from one CNN forward pass."""

    def __init__(self, backbone_name, pretrained=True):
        super().__init__()
        if backbone_name not in _PYRAMID_TARGETS:
            supported = ", ".join(sorted(_PYRAMID_TARGETS))
            raise ValueError(
                f"lesion_moe does not have a validated hierarchy for "
                f"'{backbone_name}'. Supported backbones: {supported}"
            )
        self.backbone_name = backbone_name
        self.extractor = create_backbone(backbone_name, pretrained)
        self._captured = {}
        self._stage_names = ("early", "intermediate", "deep")
        self._hooks = []

        for name, path in zip(self._stage_names, _PYRAMID_TARGETS[backbone_name]):
            module = _resolve_module(self.extractor.backbone, path)
            self._hooks.append(module.register_forward_hook(self._capture(name)))

        self.channels = self._detect_channels()

    @property
    def backbone(self):
        # Used by checkpoint initialization without registering the same module
        # twice in this container's state_dict.
        return self.extractor.backbone

    def _capture(self, name):
        def hook(_module, _inputs, output):
            self._captured[name] = output
        return hook

    def _detect_channels(self):
        was_training = self.training
        self.eval()
        parameter = next(self.parameters())
        with torch.no_grad():
            self.forward(torch.zeros(1, 3, 224, 224, device=parameter.device))
        channels = tuple(self._captured[name].shape[1] for name in self._stage_names)
        self._captured = {}
        self.train(was_training)
        return channels

    def forward(self, x):
        self._captured = {}
        self.extractor(x)
        missing = [name for name in self._stage_names if name not in self._captured]
        if missing:
            raise RuntimeError(f"Backbone hooks did not capture stages: {missing}")
        return tuple(self._captured[name] for name in self._stage_names)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, groups=1, activation=True):
        padding = dilation * (kernel_size // 2)
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class LightweightResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = ConvNormAct(channels, channels, groups=channels)
        self.pointwise = ConvNormAct(channels, channels, kernel_size=1, activation=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.pointwise(self.depthwise(x)))


def _residual_stack(channels, depth):
    return nn.Sequential(*[LightweightResidual(channels) for _ in range(max(depth, 0))])


def _resize(feature, size):
    if feature.shape[-2:] == size:
        return feature
    if feature.shape[-2] >= size[0] and feature.shape[-1] >= size[1]:
        return F.adaptive_avg_pool2d(feature, size)
    return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)


def _gradient_magnitude(feature):
    channels = feature.shape[1]
    kernel_x = feature.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)))
    kernel_y = feature.new_tensor(((-1, -2, -1), (0, 0, 0), (1, 2, 1)))
    kernel_x = kernel_x.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    dx = F.conv2d(feature, kernel_x, padding=1, groups=channels)
    dy = F.conv2d(feature, kernel_y, padding=1, groups=channels)
    return torch.sqrt(dx.square() + dy.square() + 1e-6)


class TextureExpert(nn.Module):
    def __init__(self, in_channels, proj_dim, depth):
        super().__init__()
        self.project = ConvNormAct(in_channels, proj_dim, kernel_size=1)
        self.local = ConvNormAct(proj_dim, proj_dim, groups=proj_dim)
        self.dilated = ConvNormAct(
            proj_dim, proj_dim, dilation=2, groups=proj_dim,
        )
        self.mix = ConvNormAct(proj_dim * 2, proj_dim, kernel_size=1)
        self.blocks = _residual_stack(proj_dim, depth)

    def forward(self, early, common_size):
        feature = self.project(early)
        feature = self.mix(torch.cat([self.local(feature), self.dilated(feature)], dim=1))
        return _resize(self.blocks(feature), common_size)


class MorphologyExpert(nn.Module):
    def __init__(self, in_channels, proj_dim, depth):
        super().__init__()
        self.project = ConvNormAct(in_channels, proj_dim, kernel_size=1)
        self.mix = ConvNormAct(proj_dim * 2, proj_dim, kernel_size=1)
        self.blocks = _residual_stack(proj_dim, depth)

    def forward(self, intermediate, common_size):
        learned = _resize(self.project(intermediate), common_size)
        geometry = _gradient_magnitude(learned)
        return self.blocks(self.mix(torch.cat([learned, geometry], dim=1)))


class SemanticExpert(nn.Module):
    def __init__(self, in_channels, proj_dim, depth):
        super().__init__()
        self.project = ConvNormAct(in_channels, proj_dim, kernel_size=1)
        self.blocks = _residual_stack(proj_dim, depth)
        hidden = max(proj_dim // 4, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(proj_dim, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, proj_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, deep, common_size):
        feature = self.blocks(self.project(F.relu(deep)))
        feature = feature * self.channel_gate(feature)
        return _resize(feature, common_size)


class ColorExpert(nn.Module):
    """Lightweight chromatic CNN operating on recovered [0, 1] RGB."""

    def __init__(self, proj_dim, depth):
        super().__init__()
        widths = (24, 32, 48, 64)
        self.stem = ConvNormAct(6, widths[0], stride=2)
        stages = []
        for in_channels, out_channels in zip(widths[:-1], widths[1:]):
            stages.extend([
                ConvNormAct(
                    in_channels, in_channels, stride=2, groups=in_channels,
                ),
                ConvNormAct(in_channels, out_channels, kernel_size=1),
            ])
        self.stages = nn.Sequential(*stages)
        self.project = ConvNormAct(widths[-1], proj_dim, kernel_size=1)
        self.blocks = _residual_stack(proj_dim, depth)

    @staticmethod
    def chromatic_channels(rgb):
        red, green, blue = rgb.unbind(dim=1)
        spread = rgb.amax(dim=1) - rgb.amin(dim=1)
        return torch.stack([
            red, green, blue,
            red - green,
            blue - 0.5 * (red + green),
            spread,
        ], dim=1)

    def forward(self, rgb, common_size):
        feature = self.stages(self.stem(self.chromatic_channels(rgb)))
        return _resize(self.blocks(self.project(feature)), common_size)


class BoundaryExpert(nn.Module):
    def __init__(self, in_channels, proj_dim, depth):
        super().__init__()
        self.project = ConvNormAct(in_channels, proj_dim, kernel_size=1)
        self.mix = ConvNormAct(proj_dim * 4, proj_dim, kernel_size=1)
        self.blocks = _residual_stack(proj_dim, depth)

    def forward(self, early, common_size):
        learned = _resize(self.project(early), common_size)
        gradients = _gradient_magnitude(learned)
        horizontal = torch.abs(learned - torch.flip(learned, dims=(3,)))
        vertical = torch.abs(learned - torch.flip(learned, dims=(2,)))
        cues = torch.cat([learned, gradients, horizontal, vertical], dim=1)
        return self.blocks(self.mix(cues))


class DisagreementAwareRouter(nn.Module):
    """Score each expert using its representation, disagreement, and context."""

    def __init__(self, proj_dim, num_experts, hidden_dim, dropout,
                 temperature, routing_mode):
        super().__init__()
        if routing_mode not in ("soft", "top2", "top1"):
            raise ValueError(f"Unknown routing mode: {routing_mode}")
        self.temperature = temperature
        self.routing_mode = routing_mode
        comparison_dim = max(32, proj_dim // 2)
        embedding_dim = min(16, hidden_dim)
        self.comparison_projection = nn.Sequential(
            nn.Linear(proj_dim, comparison_dim, bias=False),
            nn.LayerNorm(comparison_dim),
        )
        self.expert_embedding = nn.Parameter(
            torch.empty(num_experts, embedding_dim)
        )
        nn.init.normal_(self.expert_embedding, std=0.02)
        router_input_dim = proj_dim * 2 + 2 + embedding_dim
        self.scorer = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled):
        # pooled: [B, E, D]. The same comparison projection makes pairwise
        # cosine and absolute distances meaningful across all five experts.
        comparison = F.normalize(self.comparison_projection(pooled), dim=-1)
        num_experts = pooled.shape[1]
        off_diagonal = 1.0 - torch.eye(
            num_experts, device=pooled.device, dtype=pooled.dtype,
        )

        cosine_similarity = torch.bmm(comparison, comparison.transpose(1, 2))
        cosine_disagreement = (
            (1.0 - cosine_similarity) * off_diagonal
        ).sum(dim=2) / (num_experts - 1)

        pairwise_abs = torch.abs(
            comparison.unsqueeze(2) - comparison.unsqueeze(1)
        ).mean(dim=-1)
        absolute_disagreement = (
            pairwise_abs * off_diagonal
        ).sum(dim=2) / (num_experts - 1)
        disagreement = torch.stack(
            [cosine_disagreement, absolute_disagreement], dim=-1,
        )

        global_context = pooled.mean(dim=1, keepdim=True).expand_as(pooled)
        identity = self.expert_embedding.unsqueeze(0).expand(
            pooled.shape[0], -1, -1,
        )
        router_input = torch.cat(
            [pooled, disagreement, global_context, identity], dim=-1,
        )
        logits = self.scorer(router_input).squeeze(-1)
        probabilities = F.softmax(
            logits / max(float(self.temperature), 1e-4), dim=1,
        )

        if self.routing_mode == "soft":
            weights = probabilities
        else:
            k = 1 if self.routing_mode == "top1" else 2
            selected = probabilities.topk(k, dim=1).indices
            mask = torch.zeros_like(probabilities).scatter_(1, selected, 1.0)
            hard_weights = probabilities * mask
            hard_weights = hard_weights / hard_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            # Straight-through routing: forward values are sparse, while the
            # router still receives gradients through the soft probabilities.
            weights = (
                hard_weights + probabilities - probabilities.detach()
                if self.training else hard_weights
            )

        return weights, probabilities, comparison, disagreement


class LesionExpertMoE(nn.Module):
    """Five lesion specialists with proper sample-level feature routing."""

    expert_names = EXPERT_NAMES

    def __init__(self, backbone_name="efficientnet_b0", pretrained=True,
                 proj_dim=128, num_classes=7, branch_depth=1,
                 routing_mode="soft", router_hidden=128,
                 router_dropout=0.1, router_temperature=1.0,
                 expert_aux_weight=0.15, expert_diversity_weight=0.01,
                 router_balance_weight=0.01):
        super().__init__()
        self.feature_pyramid = HierarchicalBackbone(backbone_name, pretrained)
        self._backbone_module_name = "feature_pyramid"
        self.proj_dim = proj_dim
        self.expert_loss_weight = expert_aux_weight
        self.expert_diversity_weight = expert_diversity_weight
        self.router_balance_weight = router_balance_weight

        early_channels, intermediate_channels, deep_channels = self.feature_pyramid.channels
        self.expert_modules = nn.ModuleDict({
            "texture": TextureExpert(early_channels, proj_dim, branch_depth),
            "morphology": MorphologyExpert(
                intermediate_channels, proj_dim, branch_depth,
            ),
            "semantic": SemanticExpert(deep_channels, proj_dim, branch_depth),
            "color": ColorExpert(proj_dim, branch_depth),
            "boundary": BoundaryExpert(early_channels, proj_dim, branch_depth),
        })
        self.router = DisagreementAwareRouter(
            proj_dim=proj_dim,
            num_experts=len(self.expert_names),
            hidden_dim=router_hidden,
            dropout=router_dropout,
            temperature=router_temperature,
            routing_mode=routing_mode,
        )
        self.expert_heads = nn.ModuleList([
            nn.Linear(proj_dim, num_classes) for _ in self.expert_names
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Dropout(0.35),
            nn.Linear(proj_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )

    def _recover_rgb(self, normalized_images):
        return (
            normalized_images * self.imagenet_std + self.imagenet_mean
        ).clamp(0.0, 1.0)

    def _structural_auxiliary_loss(self, probabilities, comparison):
        num_experts = probabilities.shape[1]
        mean_usage = probabilities.mean(dim=0)
        target = torch.full_like(mean_usage, 1.0 / num_experts)
        balance = (mean_usage - target).square().mean()

        similarities = torch.bmm(comparison, comparison.transpose(1, 2))
        mask = ~torch.eye(
            num_experts, device=similarities.device, dtype=torch.bool,
        )
        diversity = similarities[:, mask].square().mean()
        return (
            self.router_balance_weight * balance
            + self.expert_diversity_weight * diversity
        )

    def forward(self, x):
        early, intermediate, deep = self.feature_pyramid(x)
        common_size = intermediate.shape[-2:]
        rgb = self._recover_rgb(x)

        features = [
            self.expert_modules["texture"](early, common_size),
            self.expert_modules["morphology"](intermediate, common_size),
            self.expert_modules["semantic"](deep, common_size),
            self.expert_modules["color"](rgb, common_size),
            self.expert_modules["boundary"](early, common_size),
        ]
        pooled = torch.stack([
            F.adaptive_avg_pool2d(feature, 1).flatten(1)
            for feature in features
        ], dim=1)
        weights, probabilities, comparison, disagreement = self.router(pooled)

        fused = sum(
            weights[:, index, None, None, None] * feature
            for index, feature in enumerate(features)
        )
        fused_pooled = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        logits = self.head(fused_pooled)
        expert_logits = torch.stack([
            head(pooled[:, index])
            for index, head in enumerate(self.expert_heads)
        ], dim=1)

        aux_loss = logits.new_zeros(())
        if self.training:
            aux_loss = self._structural_auxiliary_loss(
                probabilities, comparison,
            )

        return {
            "logits": logits,
            "expert_logits": expert_logits,
            "router_weights": weights,
            "router_probabilities": probabilities,
            "router_active": (weights.detach() > 0).to(weights.dtype),
            "expert_embeddings": comparison,
            "expert_disagreement": disagreement,
            "aux_loss": aux_loss,
        }


def create_lesion_moe(num_classes=2, pretrained=True, attention=None,
                      backbone1="efficientnet_b0", backbone2=None,
                      backbone3=None, proj_dim=128, branch_depth=1,
                      routing_mode="soft", router_hidden=128,
                      router_dropout=0.1, router_temperature=1.0,
                      expert_aux_weight=0.15,
                      expert_diversity_weight=0.01,
                      router_balance_weight=0.01):
    model = LesionExpertMoE(
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
    )
    return model, "head"
