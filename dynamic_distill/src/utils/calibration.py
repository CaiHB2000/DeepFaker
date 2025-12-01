from __future__ import annotations

import torch


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Computes the Expected Calibration Error (ECE).

    Args:
        probs: Tensor of shape (B, C) with class probabilities.
        labels: Tensor of shape (B,) with ground-truth indices.
        n_bins: Number of calibration bins.

    Returns:
        Scalar float ECE value.
    """

    confidences, predictions = probs.max(dim=-1)
    labels = labels.to(confidences.device)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            bin_conf = confidences[mask].mean()
            bin_acc = (predictions[mask] == labels[mask]).float().mean()
            ece += (mask.float().mean()) * torch.abs(bin_conf - bin_acc)

    return float(ece.item())
