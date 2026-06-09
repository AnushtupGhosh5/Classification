import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone_extractor import create_backbone


class SemanticExpert(nn.Module):
    def __init__(self, backbone_name, pretrained=True):
        super().__init__()
        self.extractor = create_backbone(backbone_name, pretrained)
        self.feature_dim = self.extractor.feature_dim
        self.is_2d = self.extractor.is_2d

    def forward(self, x):
        return self.extractor(x)


class FrequencyExpert(nn.Module):
    def __init__(self, backbone_name, pretrained=True):
        super().__init__()
        self.extractor = create_backbone(backbone_name, pretrained)
        self.feature_dim = self.extractor.feature_dim
        self.is_2d = self.extractor.is_2d

    def preprocess(self, x):
        x_complex = torch.fft.fft2(x, norm="ortho")
        magnitude = torch.abs(x_complex)
        magnitude = torch.log1p(magnitude)
        B, C, H, W = magnitude.shape
        flat = magnitude.view(B, C, -1)
        mag_min = flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        mag_max = flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)
        return magnitude

    def forward(self, x):
        x_freq = self.preprocess(x)
        return self.extractor(x_freq)


class GeometryExpert(nn.Module):
    def __init__(self, backbone_name, pretrained=True):
        super().__init__()
        self.extractor = create_backbone(backbone_name, pretrained)
        self.feature_dim = self.extractor.feature_dim
        self.is_2d = self.extractor.is_2d

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer(
            "sobel_x_kernel",
            sobel_x.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1),
        )
        self.register_buffer(
            "sobel_y_kernel",
            sobel_y.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1),
        )

    def preprocess(self, x):
        gx = F.conv2d(x, self.sobel_x_kernel, padding=1, groups=3)
        gy = F.conv2d(x, self.sobel_y_kernel, padding=1, groups=3)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        B, C, H, W = magnitude.shape
        flat = magnitude.view(B, C, -1)
        mag_min = flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        mag_max = flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)
        return magnitude

    def forward(self, x):
        x_geo = self.preprocess(x)
        return self.extractor(x_geo)
