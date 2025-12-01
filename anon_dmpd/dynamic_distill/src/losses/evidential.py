from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _dirichlet_kl(alpha: torch.Tensor, alpha_prior: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(alpha_prior))."""
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    sum_prior = alpha_prior.sum(dim=-1, keepdim=True)

    log_norm_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
    log_norm_prior = torch.lgamma(sum_prior) - torch.lgamma(alpha_prior).sum(dim=-1, keepdim=True)

    digamma_diff = torch.digamma(alpha) - torch.digamma(sum_alpha)
    kl = (log_norm_alpha - log_norm_prior).squeeze(-1)
    kl += ((alpha - alpha_prior) * digamma_diff).sum(dim=-1)
    return kl


def dirichlet_evidential_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    annealing_coef: float = 1.0,
    alpha_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes evidential cross-entropy with KL regularisation.

    Args:
        alpha: Dirichlet concentration parameters (B, C).
        targets: Ground-truth labels either as (B,) ints or (B, C) one-hot.
        num_classes: Number of classes.
        annealing_coef: Weight for the KL prior term (schedule externally).
        alpha_prior: Optional prior; defaults to uniform Dirichlet with alpha=1.
    """
    if targets.dim() == 1:
        target_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    else:
        target_one_hot = targets.float()

    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    digamma_term = torch.digamma(alpha) - torch.digamma(sum_alpha)
    expected_log = (target_one_hot * (torch.digamma(sum_alpha) - torch.digamma(alpha))).sum(dim=-1)
    nll = expected_log

    if alpha_prior is None:
        alpha_prior = torch.ones_like(alpha)

    kl = _dirichlet_kl(alpha, alpha_prior)
    loss = nll + annealing_coef * kl
    return loss.mean()
