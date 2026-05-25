import math
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.squeeze(x)
        s = self.excitation(s)
        return x * s


class _CBAMChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(nn.functional.adaptive_avg_pool2d(x, 1))
        max_out = self.shared_mlp(nn.functional.adaptive_max_pool2d(x, 1))
        return x * self.sigmoid(avg_out + max_out)


class _CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg_out, max_out], dim=1)
        s = self.sigmoid(self.conv(s))
        return x * s


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = _CBAMChannelAttention(channels, reduction)
        self.spatial_attention = _CBAMSpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        kernel_size = int(math.ceil((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.squeeze(x)
        s = s.squeeze(-1).transpose(-1, -2)
        s = self.conv(s)
        s = self.sigmoid(s).transpose(-1, -2).unsqueeze(-1)
        return x * s


ATTENTION_REGISTRY = {
    "se": SEBlock,
    "cbam": CBAM,
    "eca": ECA,
}


class SE1DBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


def get_attention_module(name, channels):
    if name is None or name == "none":
        return nn.Identity()
    if name not in ATTENTION_REGISTRY:
        available = ", ".join(["none"] + list(ATTENTION_REGISTRY.keys()))
        raise ValueError(f"Unknown attention '{name}'. Available: {available}")
    return ATTENTION_REGISTRY[name](channels)
