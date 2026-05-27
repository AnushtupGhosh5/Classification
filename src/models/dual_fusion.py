import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import get_attention_module
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
            self.post_attn = get_attention_module(attention, fusion_dim)

        layers = [
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        ]
        self.head = nn.Sequential(*layers)

    def _to_4d(self, features, extractor):
        if not extractor.is_2d:
            return features.unsqueeze(-1).unsqueeze(-1)
        return features

    def _extract_features(self, extractor, pre_attn, x):
        features = extractor(x)
        if extractor.is_2d and pre_attn is not None:
            features = pre_attn(features)
        return features

    def _align_and_cat(self, f1, ext1, f2, ext2):
        f1 = self._to_4d(f1, ext1)
        f2 = self._to_4d(f2, ext2)

        h1, w1 = f1.shape[2], f1.shape[3]
        h2, w2 = f2.shape[2], f2.shape[3]

        if h1 != h2 or w1 != w2:
            target_h = min(h1, h2)
            target_w = min(w1, w2)
            if h1 > target_h or w1 > target_w:
                f1 = F.adaptive_avg_pool2d(f1, (target_h, target_w))
            if h2 > target_h or w2 > target_w:
                f2 = F.adaptive_avg_pool2d(f2, (target_h, target_w))
            if h1 < target_h or w1 < target_w:
                f1 = F.interpolate(f1, size=(target_h, target_w), mode="bilinear", align_corners=False)
            if h2 < target_h or w2 < target_w:
                f2 = F.interpolate(f2, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return torch.cat([f1, f2], dim=1)

    def forward(self, x):
        f1 = self._extract_features(self.extractor1, self.pre_attn1, x)
        f2 = self._extract_features(self.extractor2, self.pre_attn2, x)

        fused = self._align_and_cat(f1, self.extractor1, f2, self.extractor2)

        if self.post_attn is not None:
            fused = self.post_attn(fused)

        fused = torch.flatten(F.adaptive_avg_pool2d(fused, (1, 1)), 1)
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
