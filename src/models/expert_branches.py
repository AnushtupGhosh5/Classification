import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone_extractor import create_backbone, BACKBONE_CHOICES


def reset_batchnorm(module):
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.running_mean.zero_()
                m.running_var.fill_(1.0)
                if m.weight is not None:
                    m.weight.fill_(1.0)
                if m.bias is not None:
                    m.bias.zero_()
                m.num_batches_tracked.zero_()


# ---------------------------------------------------------------------------
# Lightweight expert branches — operate on shared backbone features F_base
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Standard ResNet-style residual block for expert branch depth."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class SemanticBranch(nn.Module):
    """Pure learned conv with residual blocks — specializes on semantic content."""

    def __init__(self, in_channels, proj_dim, num_blocks=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(proj_dim) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(self.proj(x))


class FrequencyBranch(nn.Module):
    """FFT-based frequency extraction on shared features + residual blocks."""

    def __init__(self, in_channels, proj_dim, num_blocks=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(proj_dim) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.proj(x)
        fft = torch.fft.fft2(x, norm="ortho")
        magnitude = torch.log1p(torch.abs(fft))
        magnitude = torch.fft.fftshift(magnitude, dim=(-2, -1))
        return self.blocks(magnitude)


class GeometryBranch(nn.Module):
    """Sobel gradient extraction on shared features + residual blocks."""

    def __init__(self, in_channels, proj_dim, num_blocks=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(proj_dim) for _ in range(num_blocks)]
        )

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer(
            "sobel_x_kernel", sobel_x.unsqueeze(0).unsqueeze(0).expand(proj_dim, 1, 3, 3).contiguous()
        )
        self.register_buffer(
            "sobel_y_kernel", sobel_y.unsqueeze(0).unsqueeze(0).expand(proj_dim, 1, 3, 3).contiguous()
        )

    def forward(self, x):
        x = self.proj(x)
        gx = F.conv2d(x, self.sobel_x_kernel, padding=1, groups=x.shape[1])
        gy = F.conv2d(x, self.sobel_y_kernel, padding=1, groups=x.shape[1])
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return self.blocks(grad)


# ---------------------------------------------------------------------------
# Multi-layer expert (kept as alternative expert mode)
# ---------------------------------------------------------------------------

MULTI_LAYER_EXCLUDED = {"vit_b16", "vit_b32"}

_HOOK_TARGETS = {
    "resnet34": [("layer2",), ("layer3",), ("layer4",)],
    "resnet50": [("layer2",), ("layer3",), ("layer4",)],
    "resnet101": [("layer2",), ("layer3",), ("layer4",)],
    "densenet121": [
        ("features", "transition1"),
        ("features", "transition2"),
        ("features", "norm5"),
    ],
}


def _get_sequential_split_indices(features_seq):
    n = len(features_seq)
    c1 = n // 3
    c2 = 2 * n // 3
    return [c1 - 1, c2 - 1, n - 1]


class MultiLayerExpert(nn.Module):
    def __init__(self, backbone_name, pretrained=True):
        super().__init__()
        if backbone_name in MULTI_LAYER_EXCLUDED:
            raise ValueError(
                f"Multi-layer mode not supported for {backbone_name}. "
                f"ViT models produce 1D features without spatial structure."
            )
        if backbone_name not in BACKBONE_CHOICES:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. Available: {BACKBONE_CHOICES}"
            )
        self.backbone_name = backbone_name
        self.extractor = create_backbone(backbone_name, pretrained)
        self.backbone = self.extractor.backbone

        self._captured = {}
        self._hook_names = []
        self._hooks = []
        hook_modules = self._get_hook_modules()
        for name, module in hook_modules:
            self._hook_names.append(name)
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

        self.channels = self._detect_channels()

    def _make_hook(self, name):
        def hook(module, input, output):
            self._captured[name] = output
        return hook

    def _get_hook_modules(self):
        name = self.backbone_name
        b = self.backbone

        if name in _HOOK_TARGETS:
            modules = []
            for path in _HOOK_TARGETS[name]:
                m = b
                for attr in path:
                    m = getattr(m, attr)
                modules.append(("_".join(path), m))
            return modules

        features = b.features
        indices = _get_sequential_split_indices(features)
        return [(f"features_{idx}", features[idx]) for idx in indices]

    def _detect_channels(self):
        device = next(self.parameters()).device
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            self._captured = {}
            self.extractor(dummy)
            channels = []
            for name in self._hook_names:
                feat = self._captured[name]
                if feat.dim() == 4:
                    channels.append(feat.shape[1])
                else:
                    raise ValueError(
                        f"Unexpected feature shape {feat.shape} from {name}. "
                        f"Multi-layer mode requires 4D feature maps."
                    )
            return channels

    def forward(self, x):
        self._captured = {}
        self.extractor(x)
        f1 = self._captured[self._hook_names[0]]
        f2 = self._captured[self._hook_names[1]]
        f3 = self._captured[self._hook_names[2]]
        return f1, f2, f3
