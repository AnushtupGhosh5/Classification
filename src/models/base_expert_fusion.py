import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.expert_branches import SemanticExpert, FrequencyExpert, GeometryExpert, MultiLayerExpert


class BaseExpertFusion(nn.Module):
    def __init__(
        self,
        expert1=None,
        expert2=None,
        expert3=None,
        proj_dim=256,
        num_classes=2,
        expert_mode="multi_backbone",
        multi_layer_expert=None,
    ):
        super().__init__()
        self.expert_mode = expert_mode
        self.proj_dim = proj_dim

        if expert_mode == "multi_backbone":
            self.expert1 = expert1
            self.expert2 = expert2
            self.expert3 = expert3

            self.proj_s = nn.Conv2d(expert1.feature_dim, proj_dim, 1)
            self.proj_f = nn.Conv2d(expert2.feature_dim, proj_dim, 1)
            self.proj_g = nn.Conv2d(expert3.feature_dim, proj_dim, 1)

        elif expert_mode == "multi_layer":
            self.multi_layer_expert = multi_layer_expert
            channels = multi_layer_expert.channels

            self.proj_s = nn.Conv2d(channels[0], proj_dim, 1)
            self.proj_f = nn.Conv2d(channels[1], proj_dim, 1)
            self.proj_g = nn.Conv2d(channels[2], proj_dim, 1)

        self.head = nn.Sequential(
            nn.BatchNorm1d(proj_dim),
            nn.Dropout(0.4),
            nn.Linear(proj_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def _to_4d(self, features, expert):
        if not expert.is_2d:
            return features.unsqueeze(-1).unsqueeze(-1)
        return features

    def _align_features(self, features_list):
        min_h = min(f.shape[2] for f in features_list)
        min_w = min(f.shape[3] for f in features_list)
        aligned = []
        for f in features_list:
            h, w = f.shape[2], f.shape[3]
            if h > min_h or w > min_w:
                f = F.adaptive_avg_pool2d(f, (min_h, min_w))
            elif h < min_h or w < min_w:
                f = F.interpolate(
                    f, size=(min_h, min_w), mode="bilinear", align_corners=False
                )
            aligned.append(f)
        return aligned

    def extract_expert_features(self, x):
        if self.expert_mode == "multi_layer":
            f1, f2, f3 = self.multi_layer_expert(x)
        else:
            f1 = self.expert1(x)
            f2 = self.expert2(x)
            f3 = self.expert3(x)

            f1 = self._to_4d(f1, self.expert1)
            f2 = self._to_4d(f2, self.expert2)
            f3 = self._to_4d(f3, self.expert3)

        f1 = self.proj_s(f1)
        f2 = self.proj_f(f2)
        f3 = self.proj_g(f3)

        f1, f2, f3 = self._align_features([f1, f2, f3])
        return f1, f2, f3

    def pool_and_classify(self, fused):
        pooled = F.adaptive_avg_pool2d(fused, (1, 1))
        flat = torch.flatten(pooled, 1)
        return self.head(flat)

    def forward(self, x):
        raise NotImplementedError
