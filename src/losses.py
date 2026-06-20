import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
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
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1 - pt) ** self.gamma

        if self.label_smoothing > 0:
            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            smooth = -log_probs.mean(dim=1)
            ce = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth
        else:
            ce = F.nll_loss(log_probs, targets, reduction="none")

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


# ---------------------------------------------------------------------------
# Noise-robust losses for small / noisy datasets
# ---------------------------------------------------------------------------

def _log_t(u, t):
    """Compute log(u) in the tempered space: (u^(1-t) - 1) / (1 - t).

    At t=0 this reduces to log(u); at t=1 it reduces to u - 1.
    """
    if t == 1.0:
        return u - 1.0
    return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def _logsumexp_t(a, b, t):
    """Tempered log-sum-exp: log_t(a + b) ≈ log_t(a) + log_t(1 + exp(b - a))."""
    if t == 1.0:
        m = torch.max(a, b)
        return m + torch.log(torch.exp(a - m) + torch.exp(b - m))
    m = torch.max(a, b)
    return _log_t(
        torch.pow(torch.exp(a - m), 1 - t) + torch.pow(torch.exp(b - m), 1 - t),
        t,
    ) + m


class BiTemperedLogisticLoss(nn.Module):
    """Bi-Tempered Logistic Loss (Pereyra et al., 2020).

    Uses a temperature parameter *t1* on the softmax (controls tail
    heaviness — larger t1 makes the model assign non-negligible probability
    to even unlikely classes, absorbing label noise) and a temperature *t2*
    on the log (controls boundedness — t2 < 1 limits the loss magnitude so
    outliers cannot dominate the average).

    These two properties make it particularly effective for small datasets
    with noisy labels where standard cross-entropy overfits to the noise.

    Args:
        t1: Softmax temperature.  1.0 = standard softmax.  < 1 = sharper.
            Typical noise-robust values: 0.8–1.0.
        t2: Log temperature.     1.0 = standard log (unbounded loss).
            Typical noise-robust values: 0.2–0.8 (lower = more bounded).
        label_smoothing: Standard label-smoothing factor (0 = off).
        alpha: Optional per-class weights tensor.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, t1=0.8, t2=0.4, label_smoothing=0.0, alpha=None, reduction="mean"):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)

    def forward(self, logits, targets):
        if logits.dim() == 2 and logits.size(1) > 1:
            return self._multiclass_forward(logits, targets)
        return self._multiclass_forward(logits, targets)

    def _multiclass_forward(self, logits, targets):
        targets = targets.long()
        num_classes = logits.size(1)

        # --- tempered softmax (t1) ---
        # softmax_t(x)_i = softmax(x / t1)_i  (equivalent formulation)
        log_probs = F.log_softmax(logits / self.t1, dim=1)
        probs_t1 = torch.exp(log_probs)  # already tempered softmax

        # --- label smoothing (in the tempered probability space) ---
        if self.label_smoothing > 0:
            targets_onehot = torch.zeros_like(probs_t1).scatter(
                1, targets.unsqueeze(1), 1.0
            )
            targets_onehot = (
                (1 - self.label_smoothing) * targets_onehot
                + self.label_smoothing / num_classes
            )
        else:
            targets_onehot = torch.zeros_like(probs_t1).scatter(
                1, targets.unsqueeze(1), 1.0
            )

        # --- tempered log (t2) loss per sample ---
        if self.t2 == 1.0:
            loss_per_sample = -(targets_onehot * torch.log(probs_t1 + 1e-10)).sum(dim=1)
        else:
            # log_t2(p) = (p^(1-t2) - 1) / (1 - t2)
            log_t2_probs = _log_t(probs_t1.clamp(min=1e-10), self.t2)
            loss_per_sample = -(targets_onehot * log_t2_probs).sum(dim=1)

        # Apply class weights if provided
        if hasattr(self, "alpha"):
            alpha = self.alpha.to(logits.device)
            loss_per_sample = alpha[targets] * loss_per_sample

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample


class GeneralizedCrossEntropyLoss(nn.Module):
    """Generalized Cross Entropy (Zhang & Sabuncu, 2018).

    GCE(p, y) = (1 - p_y^q) / q

    When q = 0 it is equivalent to the negative log-likelihood (standard CE),
    and when q = 1 it reduces to the MAE (mean absolute error) loss.  An
    intermediate q (typically 0.5–0.7) interpolates between them, inheriting
    CE's strong-learning properties while limiting the maximum loss so noisy
    samples cannot dominate the gradient.

    Args:
        q: Truncation parameter. 0.0 → CE, 1.0 → MAE. Typical: 0.5–0.7.
        label_smoothing: Label smoothing factor (0 = off).
        alpha: Optional per-class weights tensor.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, q=0.7, label_smoothing=0.0, alpha=None, reduction="mean"):
        super().__init__()
        self.q = q
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)

    def forward(self, logits, targets):
        if logits.dim() == 2 and logits.size(1) > 1:
            return self._multiclass_forward(logits, targets)
        return self._multiclass_forward(logits, targets)

    def _multiclass_forward(self, logits, targets):
        targets = targets.long()
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)

        if self.label_smoothing > 0:
            targets_onehot = torch.zeros_like(probs).scatter(
                1, targets.unsqueeze(1), 1.0
            )
            targets_onehot = (
                (1 - self.label_smoothing) * targets_onehot
                + self.label_smoothing / num_classes
            )
            # GCE with soft targets: sum_j q_j * (1 - p_j^q) / q
            loss_per_sample = (targets_onehot * (1.0 - probs.pow(self.q))).sum(dim=1) / self.q
        else:
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            if self.q == 0.0:
                loss_per_sample = -torch.log(pt.clamp(min=1e-10))
            else:
                loss_per_sample = (1.0 - pt.pow(self.q)) / self.q

        if hasattr(self, "alpha"):
            alpha = self.alpha.to(logits.device)
            if self.label_smoothing > 0:
                # Weight by alpha per target class (hard label index)
                loss_per_sample = alpha[targets] * loss_per_sample
            else:
                loss_per_sample = alpha[targets] * loss_per_sample

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample


class SymmetricCrossEntropyLoss(nn.Module):
    """Symmetric Cross Entropy (Wang et al., 2019).

    SCE = alpha * CE + beta * RCE

    CE is the standard cross-entropy; RCE (Reverse Cross Entropy) is
    CE with the roles of prediction and target swapped:

        RCE = -sum_j  p_j * log(q_j)

    where q is the target distribution.  RCE treats every training sample
    as a "label" for the model — mislabelled samples naturally incur a
    bounded penalty because their noisy one-hot q assigns probability 1 to
    the wrong class, which the model (correctly) predicts low probability
    for.  The symmetry makes the loss inherently noise-robust.

    Args:
        alpha: Weight for CE term. Typical: 0.1–1.0.
        beta: Weight for RCE term. Typical: 0.1–1.0.
        label_smoothing: Label smoothing factor (0 = off).
        alpha_weight: Optional per-class weights tensor.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, alpha=1.0, beta=1.0, label_smoothing=0.0,
                 alpha_weight=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha_weight is not None:
            self.register_buffer("alpha_weight", alpha_weight)

    def forward(self, logits, targets):
        if logits.dim() == 2 and logits.size(1) > 1:
            return self._multiclass_forward(logits, targets)
        return self._multiclass_forward(logits, targets)

    def _multiclass_forward(self, logits, targets):
        targets = targets.long()
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)

        if self.label_smoothing > 0:
            targets_onehot = torch.zeros_like(probs).scatter(
                1, targets.unsqueeze(1), 1.0
            )
            targets_onehot = (
                (1 - self.label_smoothing) * targets_onehot
                + self.label_smoothing / num_classes
            )
        else:
            targets_onehot = torch.zeros_like(probs).scatter(
                1, targets.unsqueeze(1), 1.0
            )

        # --- CE term: -sum_j q_j * log(p_j) ---
        ce = -(targets_onehot * torch.log(probs.clamp(min=1e-10))).sum(dim=1)

        # --- RCE term: -sum_j p_j * log(q_j) ---
        # q is the (possibly smoothed) target; clamp to avoid log(0).
        rce = -(probs * torch.log(targets_onehot.clamp(min=1e-10))).sum(dim=1)

        loss_per_sample = self.alpha * ce + self.beta * rce

        if hasattr(self, "alpha_weight"):
            w = self.alpha_weight.to(logits.device)
            loss_per_sample = w[targets] * loss_per_sample

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample
