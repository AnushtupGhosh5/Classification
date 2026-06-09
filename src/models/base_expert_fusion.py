import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.expert_branches import SemanticExpert, FrequencyExpert, GeometryExpert


class BaseExpertFusion(nn.Module):
    def __init__(
        self,
        semantic_expert,
        frequency_expert,
        geometry_expert,
        proj_dim,
        num_classes,
    ):
        super().__init__()
        self.semantic_expert = semantic_expert
        self.frequency_expert = frequency_expert
        self.geometry_expert = geometry_expert
        self.proj_dim = proj_dim

        self.proj_s = nn.Conv2d(semantic_expert.feature_dim, proj_dim, 1)
        self.proj_f = nn.Conv2d(frequency_expert.feature_dim, proj_dim, 1)
        self.proj_g = nn.Conv2d(geometry_expert.feature_dim, proj_dim, 1)

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
        fs = self.semantic_expert(x)
        ff = self.frequency_expert(x)
        fg = self.geometry_expert(x)

        fs = self._to_4d(fs, self.semantic_expert)
        ff = self._to_4d(ff, self.frequency_expert)
        fg = self._to_4d(fg, self.geometry_expert)

        fs = self.proj_s(fs)
        ff = self.proj_f(ff)
        fg = self.proj_g(fg)

        fs, ff, fg = self._align_features([fs, ff, fg])
        return fs, ff, fg

    def pool_and_classify(self, fused):
        pooled = F.adaptive_avg_pool2d(fused, (1, 1))
        flat = torch.flatten(pooled, 1)
        return self.head(flat)

    def forward(self, x):
        raise NotImplementedError
