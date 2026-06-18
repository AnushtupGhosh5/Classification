import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_expert_fusion import BaseExpertFusion
from src.models.expert_branches import MultiLayerExpert


class MutualInfoExpertFusion(BaseExpertFusion):
    """Mutual Information Expert Fusion with histogram-based MI estimation.

    Pipeline:
      1. Project each expert feature to a low-dim embedding via phi networks
      2. Estimate pairwise MI using differentiable soft histograms:
         I(X,Y) = H(X) + H(Y) - H(X,Y)
      3. Compute per-expert complementarity from pairwise scores
      4. Generate fusion weights via softmax over learned MLP
    """

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        proj_dim=256,
        num_classes=2,
        mi_dim=8,
        mi_bins=16,
        expert_mode="shared_base",
        multi_layer_expert=None,
        branch_depth=2,
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
        self.mi_dim = mi_dim
        self.mi_bins = mi_bins

        self.phi_s = nn.Sequential(
            nn.Conv2d(proj_dim, mi_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.phi_f = nn.Sequential(
            nn.Conv2d(proj_dim, mi_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.phi_g = nn.Sequential(
            nn.Conv2d(proj_dim, mi_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.weight_gen = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
        )

    def _soft_hist_1d(self, x, bins, sigma, min_val, max_val):
        """Differentiable 1D soft histogram. x: [B] -> hist: [bins]."""
        edges = torch.linspace(min_val.item(), max_val.item(), bins, device=x.device)
        diff = x.unsqueeze(1) - edges.unsqueeze(0)  # [B, bins]
        w = torch.exp(-0.5 * (diff / sigma) ** 2)
        hist = w.sum(dim=0)
        hist = hist / (hist.sum() + 1e-8)
        return hist

    def _soft_hist_2d(self, x, y, bins, sigma, min_x, max_x, min_y, max_y):
        """Differentiable 2D soft histogram. x: [B], y: [B] -> hist: [bins, bins]."""
        edges_x = torch.linspace(min_x.item(), max_x.item(), bins, device=x.device)
        edges_y = torch.linspace(min_y.item(), max_y.item(), bins, device=y.device)

        diff_x = x.unsqueeze(1) - edges_x.unsqueeze(0)  # [B, bins]
        diff_y = y.unsqueeze(1) - edges_y.unsqueeze(0)  # [B, bins]
        wx = torch.exp(-0.5 * (diff_x / sigma) ** 2)
        wy = torch.exp(-0.5 * (diff_y / sigma) ** 2)

        joint = torch.bmm(wx.unsqueeze(2), wy.unsqueeze(1))  # [B, bins, bins]
        hist = joint.sum(dim=0)
        hist = hist / (hist.sum() + 1e-8)
        return hist

    @staticmethod
    def _entropy(hist):
        """Shannon entropy from a probability distribution (1D or 2D)."""
        hist = hist + 1e-8
        return -(hist * torch.log(hist)).sum()

    def _mutual_information(self, x, y):
        """Estimate MI using soft histograms, averaged over dimensions.

        x: [B, D], y: [B, D] -> scalar MI
        """
        D = x.shape[1]
        mi_total = x.new_zeros(1).squeeze()

        for d in range(D):
            x_d = x[:, d]
            y_d = y[:, d]

            min_x, max_x = x_d.min().detach(), x_d.max().detach()
            min_y, max_y = y_d.min().detach(), y_d.max().detach()

            range_x = (max_x - min_x).clamp(min=1e-6)
            range_y = (max_y - min_y).clamp(min=1e-6)
            sigma_x = range_x / self.mi_bins * 0.5
            sigma_y = range_y / self.mi_bins * 0.5

            h_x = self._entropy(self._soft_hist_1d(x_d, self.mi_bins, sigma_x, min_x, max_x))
            h_y = self._entropy(self._soft_hist_1d(y_d, self.mi_bins, sigma_y, min_y, max_y))
            h_xy = self._entropy(
                self._soft_hist_2d(x_d, y_d, self.mi_bins, max(sigma_x, sigma_y),
                                   min_x, max_x, min_y, max_y)
            )

            mi_total = mi_total + (h_x + h_y - h_xy)

        return mi_total / D

    def forward(self, x):
        fs, ff, fg = self.extract_expert_features(x)

        zs = F.adaptive_avg_pool2d(self.phi_s(fs), (1, 1)).flatten(1)
        zf = F.adaptive_avg_pool2d(self.phi_f(ff), (1, 1)).flatten(1)
        zg = F.adaptive_avg_pool2d(self.phi_g(fg), (1, 1)).flatten(1)

        mi_sf = self._mutual_information(zs, zf)
        mi_sg = self._mutual_information(zs, zg)
        mi_fg = self._mutual_information(zf, zg)

        # Per-expert complementarity: average of pairwise complementarities
        # involving that expert
        comp_s = ((1 - mi_sf) + (1 - mi_sg)) / 2
        comp_f = ((1 - mi_sf) + (1 - mi_fg)) / 2
        comp_g = ((1 - mi_sg) + (1 - mi_fg)) / 2
        comp = torch.stack([comp_s, comp_f, comp_g])  # [3]

        weights = F.softmax(self.weight_gen(comp.unsqueeze(0)), dim=-1)  # [1, 3]
        ws = weights[0, 0]
        wf = weights[0, 1]
        wg = weights[0, 2]

        fused = ws * fs + wf * ff + wg * fg

        return self.pool_and_classify(fused)


def create_mief(
    num_classes=2,
    pretrained=True,
    attention=None,
    backbone1="resnet50",
    backbone2=None,
    backbone3=None,
    proj_dim=256,
    expert_mode="shared_base",
    branch_depth=2,
):
    if expert_mode == "multi_layer":
        ml_expert = MultiLayerExpert(backbone1, pretrained)
        model = MutualInfoExpertFusion(
            backbone_name=backbone1,
            pretrained=pretrained,
            proj_dim=proj_dim,
            num_classes=num_classes,
            expert_mode="multi_layer",
            multi_layer_expert=ml_expert,
        )
    else:
        model = MutualInfoExpertFusion(
            backbone_name=backbone1,
            pretrained=pretrained,
            proj_dim=proj_dim,
            num_classes=num_classes,
            expert_mode="shared_base",
            branch_depth=branch_depth,
        )
    return model, "head"
