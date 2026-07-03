"""Pre-training heads and loss functions."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveHead(nn.Module):
    """Projection head for InfoNCE contrastive learning (CoLES-style).

    Projects h_CLS into a lower-dimensional normalized space.
    Uses learnable temperature.
    """

    def __init__(self, d_model: int = 128, d_proj: int = 64):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_proj),
        )
        # Learnable temperature (log scale for stability), init ~ 0.07
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def forward(self, h_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_cls: (B, d_model)
        Returns:
            (B, d_proj) — L2-normalized projection
        """
        z = self.projector(h_cls)
        return F.normalize(z, dim=-1)


def info_nce_loss(
    z: torch.Tensor,
    client_ids: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """InfoNCE (NT-Xent) contrastive loss with in-batch negatives.

    Positive pairs: subsequences from the same client.
    Negatives: all other samples in the batch.

    Args:
        z: (B, d_proj) — L2-normalized projections
        client_ids: (B,) — client identifiers (same ID = positive pair)
        temperature: scalar — learnable temperature
    Returns:
        scalar loss
    """
    # Similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (B, B)

    # Positive mask: same client, different sample
    pos_mask = (client_ids.unsqueeze(0) == client_ids.unsqueeze(1))  # (B, B)
    pos_mask.fill_diagonal_(False)  # exclude self

    # If no positive pairs exist, return 0
    if not pos_mask.any():
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    # Mask out self-similarity
    self_mask = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, torch.finfo(sim.dtype).min)

    # Log-softmax over columns (all other samples)
    log_probs = F.log_softmax(sim, dim=1)  # (B, B)

    # Average log-prob over positive pairs
    loss = -(log_probs * pos_mask.float()).sum() / pos_mask.float().sum()
    return loss


class MTMHead(nn.Module):
    """Masked Token Modeling head.

    Reconstructs masked fields:
    - Categorical fields: linear → vocab logits → cross-entropy
    - Numeric fields: linear → scalar → smooth-L1
    """

    def __init__(
        self,
        d_model: int = 128,
        vocab_sizes: dict[str, int] | None = None,
        numeric_names: list[str] | None = None,
    ):
        super().__init__()
        if vocab_sizes is None:
            vocab_sizes = {}
        if numeric_names is None:
            numeric_names = []

        # One classification head per categorical field
        self.cat_heads = nn.ModuleDict({
            name: nn.Linear(d_model, vocab_size)
            for name, vocab_size in vocab_sizes.items()
        })

        # One regression head per numeric field
        self.num_heads = nn.ModuleDict({
            name: nn.Linear(d_model, 1)
            for name in numeric_names
        })

    def forward(self, hidden_states: torch.Tensor) -> dict:
        """
        Args:
            hidden_states: (B, T, d_model) — sequence transformer outputs (excl. [CLS])
        Returns:
            dict with logits/predictions for each field
        """
        preds = {}
        for name, head in self.cat_heads.items():
            preds[f"cat_{name}"] = head(hidden_states)              # (B, T, vocab)
        for name, head in self.num_heads.items():
            preds[f"num_{name}"] = head(hidden_states).squeeze(-1)  # (B, T)
        return preds

def mtm_loss(
    preds: dict,
    targets: dict,
    mask: dict,
    return_breakdown: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combined MTM loss: cross-entropy for categoricals, smooth-L1 for numerics.

    Args:
        preds: dict from MTMHead.forward()
        targets: dict with ground-truth field values
        mask: dict with boolean masks (True = this position was masked)
        return_breakdown: if True also return per-field detached losses
    Returns:
        scalar loss, or (scalar, dict[name -> scalar]) when return_breakdown
    """
    total_loss = torch.tensor(0.0, device=next(iter(preds.values())).device)
    n_terms = 0
    breakdown: dict[str, torch.Tensor] = {}

    # Categorical losses — driven by whatever cat_heads are present in preds
    for key in preds:
        if not key.startswith("cat_"):
            continue
        name = key[len("cat_"):]
        if name in mask and mask[name].any():
            logits = preds[key][mask[name]]        # (N_masked, vocab)
            target = targets[name][mask[name]]     # (N_masked,)
            field_loss = F.cross_entropy(logits, target)
            total_loss = total_loss + field_loss
            n_terms += 1
            breakdown[key] = field_loss.detach()

    # Numeric losses — driven by whatever num_heads are present in preds
    for key in preds:
        if not key.startswith("num_"):
            continue
        name = key[len("num_"):]
        if name in mask and mask[name].any():
            pred_vals = preds[key][mask[name]]       # (N_masked,)
            target_vals = targets[name][mask[name]]  # (N_masked,)
            field_loss = F.smooth_l1_loss(pred_vals, target_vals)
            total_loss = total_loss + field_loss
            n_terms += 1
            breakdown[key] = field_loss.detach()

    total = total_loss / max(n_terms, 1)
    if return_breakdown:
        return total, breakdown
    return total


def info_nce_metrics(
    z: torch.Tensor, client_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Top-1 retrieval accuracy + random baseline + normalized lift.

    - ``infonce_acc``: fraction of anchors whose nearest neighbour (excluding
      self) belongs to the same client.
    - ``infonce_acc_random``: expected accuracy of picking uniformly at
      random among the ``B-1`` non-self candidates, averaged over anchors.
      Per anchor it equals ``(positives_in_batch) / (B - 1)`` which, with
      ``k`` windows per client, simplifies to ``(k-1)/(B-1)``.
    - ``infonce_lift``: ``(acc - acc_random) / (1 - acc_random)`` — 0 means
      no better than chance, 1 means perfect retrieval.

    All metrics are 0 when no positive pairs exist in the batch.
    """
    zero = torch.tensor(0.0, device=z.device)
    out = {"infonce_acc": zero, "infonce_acc_random": zero, "infonce_lift": zero}
    B = z.size(0)
    if B < 2:
        return out

    pos_mask = (client_ids.unsqueeze(0) == client_ids.unsqueeze(1))
    pos_mask.fill_diagonal_(False)
    pos_exists = pos_mask.any(dim=1)
    if not pos_exists.any():
        return out

    sim = z @ z.t()
    sim.fill_diagonal_(float("-inf"))
    nn_idx = sim.argmax(dim=1)
    correct = (client_ids[nn_idx] == client_ids).float()

    valid = pos_exists.float()
    n_valid = valid.sum()
    acc = (correct * valid).sum() / n_valid

    # Per-anchor random baseline: positives / (B-1), averaged over valid anchors.
    pos_count = pos_mask.sum(dim=1).float()
    acc_random = (pos_count / (B - 1) * valid).sum() / n_valid

    denom = (1.0 - acc_random).clamp(min=1e-8)
    lift = (acc - acc_random) / denom

    return {"infonce_acc": acc, "infonce_acc_random": acc_random, "infonce_lift": lift}


class PretrainLoss(nn.Module):
    """Combined pre-training loss: L = L_MTM + λ * L_contrastive."""

    def __init__(self, contrastive_weight: float = 0.5):
        super().__init__()
        self.contrastive_weight = contrastive_weight

    def forward(
        self,
        output: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mtm_mask: dict[str, torch.Tensor],
        client_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_mtm, mtm_breakdown = mtm_loss(
            output["mtm_preds"],
            targets,
            mtm_mask,
            return_breakdown=True,
        )
        loss_contrastive = info_nce_loss(
            output["contrastive_z"],
            client_ids,
            output["temperature"],
        )
        loss = loss_mtm + self.contrastive_weight * loss_contrastive

        return {
            "loss": loss,
            "loss_mtm": loss_mtm.detach(),
            "loss_contrastive": loss_contrastive.detach(),
            "mtm_breakdown": mtm_breakdown,
        }