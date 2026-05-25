import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import get_attention_module, SE1DBlock
from src.models.backbone_extractor import create_backbone


class DualFusion(nn.Module):
    def __init__(self, extractor1, extractor2, fusion_dim, num_classes,
                 attention=None, fusion_mode="both"):
        super().__init__()
        self.extractor1 = extractor1
        self.extractor2 = extractor2
        self.fusion_dim = fusion_dim
        self.fusion_mode = fusion_mode

        use_pre = attention and attention != "none" and fusion_mode in ("pre_fusion", "both")
        use_post = attention and attention != "none" and fusion_mode in ("post_fusion", "both")

        self.pre_attn1 = None
        self.pre_attn2 = None
        if use_pre:
            if extractor1.is_2d:
                self.pre_attn1 = get_attention_module(attention, extractor1.feature_dim)
            if extractor2.is_2d:
                self.pre_attn2 = get_attention_module(attention, extractor2.feature_dim)

        self.post_attn = None
        if use_post:
            self.post_attn = SE1DBlock(fusion_dim)

        layers = [
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        ]
        self.head = nn.Sequential(*layers)

    def _pool_flat(self, x):
        return torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)

    def _extract_features(self, extractor, pre_attn, x):
        features = extractor(x)
        if extractor.is_2d:
            if pre_attn is not None:
                features = pre_attn(features)
            return self._pool_flat(features)
        return features

    def forward(self, x):
        f1 = self._extract_features(self.extractor1, self.pre_attn1, x)
        f2 = self._extract_features(self.extractor2, self.pre_attn2, x)
        fused = torch.cat([f1, f2], dim=1)
        if self.post_attn is not None:
            fused = self.post_attn(fused)
        return self.head(fused)


def create_dual_fusion(num_classes=2, pretrained=True, attention=None,
                       backbone1="mobilenetv2", backbone2="densenet121",
                       fusion_mode="both"):
    ext1 = create_backbone(backbone1, pretrained)
    ext2 = create_backbone(backbone2, pretrained)
    fusion_dim = ext1.feature_dim + ext2.feature_dim

    if (attention and attention != "none"
            and fusion_mode in ("pre_fusion", "both")
            and not ext1.is_2d and not ext2.is_2d):
        print(f"  Warning: pre-fusion 2D attention not applicable when both "
              f"backbones are ViT ('{backbone1}', '{backbone2}')")

    model = DualFusion(ext1, ext2, fusion_dim, num_classes, attention, fusion_mode)
    return model, "head"
