"""
Pre-training heads and loss functions.

MTMHead              — Masked Token Modeling head (categorical + numeric reconstruction)
ContrastiveHead      — InfoNCE projection head with learnable temperature
info_nce_loss        — InfoNCE (NT-Xent) contrastive loss with in-batch negatives
mtm_loss             — combined MTM loss (cross-entropy + smooth-L1)
combined_pretrain_loss — L_MTM + λ * L_contrastive
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://towardsdatascience.com/self-supervised-learning-using-projection-heads-b77af3911d33/
class MTMHead(nn.Module):
    """Masked Token Modeling head.

    Reconstructs masked fields:
    - Categorical fields: linear → vocab logits → cross-entropy
    - Numeric fields: linear → scalar → smooth-L1
    - Full transaction masking: linear → d_model vector → MSE
    """

    def __init__(self, d_model: int = 128, vocab_sizes: dict[str, int] | None = None):
        super().__init__()
        if vocab_sizes is None:
            vocab_sizes = {}

        # One classification head per categorical field
        self.cat_heads = nn.ModuleDict({
            name: nn.Linear(d_model, vocab_size)
            for name, vocab_size in vocab_sizes.items()
        })

        # Numeric regression heads (importo, saldo_post, delta_t)
        self.num_heads = nn.ModuleDict({
            name: nn.Linear(d_model, 1)
            for name in ["importo", "saldo_post", "delta_t"]
        })

        # Full transaction reconstruction head
        self.full_recon = nn.Linear(d_model, d_model)

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
        preds["full_recon"] = self.full_recon(hidden_states)        # (B, T, d_model)
        return preds


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


def info_nce_loss(z: torch.Tensor, client_ids: torch.Tensor,
                  temperature: torch.Tensor) -> torch.Tensor:
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
    sim = sim.masked_fill(self_mask, -1e9)

    # Log-softmax over columns (all other samples)
    log_probs = F.log_softmax(sim, dim=1)  # (B, B)

    # Average log-prob over positive pairs
    loss = -(log_probs * pos_mask.float()).sum() / pos_mask.float().sum()
    return loss


def mtm_loss(preds: dict, targets: dict, mask: dict) -> torch.Tensor:
    """Combined MTM loss: cross-entropy for categoricals, smooth-L1 for numerics.

    Args:
        preds: dict from MTMHead.forward()
        targets: dict with ground-truth field values
        mask: dict with boolean masks (True = this position was masked)
    Returns:
        scalar loss
    """
    total_loss = torch.tensor(0.0, device=next(iter(preds.values())).device)
    n_terms = 0

    # Categorical losses — driven by whatever cat_heads are present in preds
    for key in preds:
        if not key.startswith("cat_"):
            continue
        name = key[len("cat_"):]
        if name in mask and mask[name].any():
            logits = preds[key][mask[name]]        # (N_masked, vocab)
            target = targets[name][mask[name]]     # (N_masked,)
            total_loss = total_loss + F.cross_entropy(logits, target)
            n_terms += 1

    # Numeric losses
    for name in ["importo", "saldo_post", "delta_t"]:
        key = f"num_{name}"
        if key in preds and name in mask and mask[name].any():
            pred_vals = preds[key][mask[name]]       # (N_masked,)
            target_vals = targets[name][mask[name]]  # (N_masked,)
            total_loss = total_loss + F.smooth_l1_loss(pred_vals, target_vals)
            n_terms += 1

    return total_loss / max(n_terms, 1)


def combined_pretrain_loss(
    output: dict,
    targets: dict,
    mtm_mask: dict,
    client_ids: torch.Tensor,
    contrastive_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Combined pre-training loss: L = L_MTM + λ * L_contrastive.

    Returns dict with individual losses for logging.
    """
    l_mtm = mtm_loss(output["mtm_preds"], targets, mtm_mask)
    l_contrastive = info_nce_loss(
        output["contrastive_z"],
        client_ids,
        output.get("temperature", torch.tensor(0.07)),
    )
    total = l_mtm + contrastive_weight * l_contrastive

    return {
        "loss": total,
        "loss_mtm": l_mtm.detach(),
        "loss_contrastive": l_contrastive.detach(),
    }
