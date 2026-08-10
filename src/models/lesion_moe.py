"""Hierarchical complementary residual mixture of experts.

The shared CNN and its classifier are the semantic baseline. Four specialists
produce only complementary logit corrections: texture, morphology, color, and
boundary. A sample-level router uses compact disagreement descriptors to route
those corrections without duplicating or replacing the semantic baseline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone_extractor import create_backbone


EXPERT_NAMES = ("texture", "morphology", "color", "boundary")

# These stages are semantic choices, not equal-length guesses. EfficientNet
# stage 2 retains fine detail, stage 5 is the last 1/16 stage, and stage 8 is
# the final representation. ConvNeXt stages 1/5/7 retain 1/4, 1/16, and 1/32
# resolution respectively. ResNet/DenseNet mappings follow the same intent.
_PYRAMID_TARGETS = {
    "efficientnet_b0": (("features", "2"), ("features", "5"), ("features", "8")),
    "efficientnet_b1": (("features", "2"), ("features", "5"), ("features", "8")),
    "efficientnet_b2": (("features", "2"), ("features", "5"), ("features", "8")),
    "convnext_tiny": (("features", "1"), ("features", "5"), ("features", "7")),
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
        # cosine and absolute distances meaningful across all experts.
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


class ComplementaryDeltaHeads(nn.Module):
    """One zero-initialized class-logit correction head per expert."""

    def __init__(self, proj_dim, num_classes, num_experts, dropout=0.25):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(proj_dim),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, num_classes),
            )
            for _ in range(num_experts)
        ])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, pooled):
        return torch.stack([
            head(pooled[:, index])
            for index, head in enumerate(self.heads)
        ], dim=1)


class SelectiveCorrectionGate(nn.Module):
    """Per-sample gate from baseline uncertainty and expert disagreement."""

    def __init__(self, num_classes, hidden_dim=16, initial_bias=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, float(initial_bias))

    def forward(self, baseline_logits, disagreement):
        # Stop gate gradients from changing the protected baseline merely to
        # manipulate its uncertainty values.
        probabilities = F.softmax(baseline_logits.detach(), dim=1)
        top2 = probabilities.topk(min(2, self.num_classes), dim=1).values
        confidence = top2[:, 0]
        margin = (
            top2[:, 0] - top2[:, 1]
            if top2.shape[1] > 1 else top2[:, 0]
        )
        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1) / max(math.log(float(self.num_classes)), 1e-8)
        mean_disagreement = disagreement.mean(dim=1)
        features = torch.stack([
            1.0 - confidence,
            entropy,
            1.0 - margin,
            mean_disagreement[:, 0],
            mean_disagreement[:, 1],
        ], dim=1)
        return torch.tanh(self.network(features).squeeze(1))


class LesionExpertMoE(nn.Module):
    """Semantic baseline plus four routed complementary corrections."""

    expert_names = EXPERT_NAMES

    def __init__(self, backbone_name="efficientnet_b0", pretrained=True,
                 proj_dim=128, num_classes=7, branch_depth=1,
                 routing_mode="soft", router_hidden=128,
                 router_dropout=0.1, router_temperature=1.0,
                 expert_aux_weight=0.15, expert_diversity_weight=0.01,
                 router_balance_weight=0.01, expert_warmup_epochs=0,
                 expert_dropout=0.0, correction_aux_weight=0.1,
                 correction_gate_init=0.0):
        super().__init__()
        self.feature_pyramid = HierarchicalBackbone(backbone_name, pretrained)
        self._backbone_module_name = "feature_pyramid"
        self.proj_dim = proj_dim
        self.expert_loss_weight = expert_aux_weight
        self.expert_diversity_weight = expert_diversity_weight
        self.router_balance_weight = router_balance_weight
        self.expert_warmup_epochs = expert_warmup_epochs
        self.expert_dropout = expert_dropout
        self.correction_loss_weight = correction_aux_weight
        self.router_gain_loss_weight = 0.0
        self.router_gain_temperature = 0.25
        self.current_training_epoch = 0
        self.backbone_name = backbone_name
        self.register_buffer(
            "baseline_loaded_flag", torch.tensor(False), persistent=True,
        )

        early_channels, intermediate_channels, deep_channels = self.feature_pyramid.channels
        self.baseline_norm = (
            nn.LayerNorm(deep_channels, eps=1e-6)
            if backbone_name == "convnext_tiny" else nn.Identity()
        )
        self.baseline_classifier = nn.Linear(deep_channels, num_classes)
        self.expert_modules = nn.ModuleDict({
            "texture": TextureExpert(early_channels, proj_dim, branch_depth),
            "morphology": MorphologyExpert(
                intermediate_channels, proj_dim, branch_depth,
            ),
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
        self.head = ComplementaryDeltaHeads(
            proj_dim, num_classes, len(self.expert_names), dropout=0.25,
        )
        # A zero-initialized per-sample gate makes the initial prediction
        # exactly the baseline while retaining selective correction capacity.
        self.correction_gate = SelectiveCorrectionGate(
            num_classes, initial_bias=correction_gate_init,
        )

        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )

    def set_training_epoch(self, epoch):
        self.current_training_epoch = int(epoch)

    def load_baseline_classifier(self, state_dict):
        """Load the classifier paired with a backbone-only initialization."""
        if self.backbone_name == "convnext_tiny":
            norm_keys = ("classifier.0.weight", "classifier.0.bias")
            linear_keys = ("classifier.3.weight", "classifier.3.bias")
        elif self.backbone_name.startswith("efficientnet"):
            norm_keys = None
            linear_keys = ("classifier.1.weight", "classifier.1.bias")
        elif self.backbone_name.startswith("resnet"):
            norm_keys = None
            linear_keys = ("fc.weight", "fc.bias")
        elif self.backbone_name == "densenet121":
            norm_keys = None
            linear_keys = ("classifier.weight", "classifier.bias")
        else:
            return 0

        weight, bias = (state_dict.get(key) for key in linear_keys)
        if (
            weight is None or bias is None
            or weight.shape != self.baseline_classifier.weight.shape
            or bias.shape != self.baseline_classifier.bias.shape
        ):
            return 0
        with torch.no_grad():
            self.baseline_classifier.weight.copy_(weight)
            self.baseline_classifier.bias.copy_(bias)
            loaded = 2
            if norm_keys is not None and isinstance(self.baseline_norm, nn.LayerNorm):
                norm_weight, norm_bias = (state_dict.get(key) for key in norm_keys)
                if (
                    norm_weight is not None and norm_bias is not None
                    and norm_weight.shape == self.baseline_norm.weight.shape
                    and norm_bias.shape == self.baseline_norm.bias.shape
                ):
                    self.baseline_norm.weight.copy_(norm_weight)
                    self.baseline_norm.bias.copy_(norm_bias)
                    loaded += 2
        self.baseline_loaded_flag.fill_(True)
        return loaded

    def freeze_loaded_baseline(self):
        if not bool(self.baseline_loaded_flag):
            return
        for module in (self.baseline_norm, self.baseline_classifier):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def _recover_rgb(self, normalized_images):
        return (
            normalized_images * self.imagenet_std + self.imagenet_mean
        ).clamp(0.0, 1.0)

    def _structural_auxiliary_loss(self, probabilities, comparison):
        num_experts = probabilities.shape[1]
        mean_usage = probabilities.mean(dim=0)
        # KL(mean usage || uniform) strongly penalizes global collapse while
        # still permitting different samples to choose different experts.
        balance = (
            mean_usage * (mean_usage.clamp_min(1e-8).log() + torch.log(
                mean_usage.new_tensor(float(num_experts))
            ))
        ).sum()

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

        baseline_deep = F.relu(deep) if self.backbone_name == "densenet121" else deep
        baseline_pooled = F.adaptive_avg_pool2d(baseline_deep, 1).flatten(1)
        baseline_logits = self.baseline_classifier(
            self.baseline_norm(baseline_pooled)
        )

        features = [
            self.expert_modules["texture"](early, common_size),
            self.expert_modules["morphology"](intermediate, common_size),
            self.expert_modules["color"](rgb, common_size),
            self.expert_modules["boundary"](early, common_size),
        ]
        pooled = torch.stack([
            F.adaptive_avg_pool2d(feature, 1).flatten(1)
            for feature in features
        ], dim=1)
        weights, probabilities, comparison, disagreement = self.router(pooled)

        in_warmup = (
            self.expert_warmup_epochs > 0
            and 0 < self.current_training_epoch <= self.expert_warmup_epochs
        )
        if in_warmup:
            weights = torch.full_like(weights, 1.0 / len(self.expert_names))
        elif self.training and self.expert_dropout > 0:
            keep = torch.rand_like(weights) >= self.expert_dropout
            # Every sample must retain at least one route.
            empty = ~keep.any(dim=1)
            if empty.any():
                keep[empty, probabilities[empty].argmax(dim=1)] = True
            weights = weights * keep.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        expert_delta_logits = self.head(pooled)
        correction_delta = (
            weights.unsqueeze(-1) * expert_delta_logits
        ).sum(dim=1)
        # The correction path is supervised as a complete prediction during
        # warm-up even though its contribution to final logits is gated off.
        correction_logits = baseline_logits + correction_delta
        correction_scale = self.correction_gate(
            baseline_logits, disagreement,
        )
        if in_warmup:
            correction_scale = correction_scale * 0.0
        logits = baseline_logits + correction_scale.unsqueeze(1) * correction_delta
        expert_logits = baseline_logits.unsqueeze(1) + expert_delta_logits

        aux_loss = logits.new_zeros(())
        if self.training:
            aux_loss = self._structural_auxiliary_loss(
                probabilities, comparison,
            )

        return {
            "logits": logits,
            "expert_logits": expert_logits,
            "expert_delta_logits": expert_delta_logits,
            "router_weights": weights,
            "router_probabilities": probabilities,
            "router_active": (weights.detach() > 0).to(weights.dtype),
            "expert_embeddings": comparison,
            "expert_disagreement": disagreement,
            "baseline_logits": baseline_logits,
            "correction_logits": correction_logits,
            "correction_scale": correction_scale,
            "aux_loss": aux_loss,
        }


def create_lesion_moe(num_classes=2, pretrained=True, attention=None,
                      backbone1="efficientnet_b0", backbone2=None,
                      backbone3=None, proj_dim=128, branch_depth=1,
                      routing_mode="soft", router_hidden=128,
                      router_dropout=0.1, router_temperature=1.0,
                      expert_aux_weight=0.15,
                      expert_diversity_weight=0.01,
                      router_balance_weight=0.01,
                      expert_warmup_epochs=0, expert_dropout=0.0,
                      correction_aux_weight=0.1,
                      correction_gate_init=0.0,
                      router_gain_weight=0.0,
                      router_gain_temperature=0.25):
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
        expert_warmup_epochs=expert_warmup_epochs,
        expert_dropout=expert_dropout,
        correction_aux_weight=correction_aux_weight,
        correction_gate_init=correction_gate_init,
    )
    model.router_gain_loss_weight = router_gain_weight
    model.router_gain_temperature = router_gain_temperature
    return model, "head"
