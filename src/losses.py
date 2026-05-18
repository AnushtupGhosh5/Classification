import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)

    def forward(self, logits, targets):
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
            return self._binary_forward(logits, targets)
        if logits.dim() == 2 and logits.size(1) > 1:
            return self._multiclass_forward(logits, targets)
        return self._binary_forward(logits.squeeze(), targets)

    def _binary_forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma

        if hasattr(self, "alpha"):
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[1] * targets + alpha[0] * (1 - targets)
            loss = alpha_t * focal_factor * bce
        else:
            loss = focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _multiclass_forward(self, logits, targets):
        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        ce = F.nll_loss(log_probs, targets, reduction="none")
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1 - pt) ** self.gamma

        if hasattr(self, "alpha"):
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = alpha_t * focal_factor * ce
        else:
            loss = focal_factor * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.pos_weight = None
        if class_weights is not None:
            pos_weight = class_weights[1] / class_weights[0]
            self.register_buffer("pos_weight", pos_weight.unsqueeze(0))

    def forward(self, logits, targets):
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        targets = targets.float()

        kwargs = {"reduction": "mean"}
        if self.pos_weight is not None:
            kwargs["pos_weight"] = self.pos_weight

        return F.binary_cross_entropy_with_logits(logits, targets, **kwargs)
