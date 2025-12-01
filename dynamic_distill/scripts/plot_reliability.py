#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    accs, confs = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            accs.append(np.nan)
            confs.append((bins[b] + bins[b + 1]) / 2)
            continue
        accs.append((preds[mask] == labels[mask]).mean())
        confs.append(conf[mask].mean())
    return np.array(confs), np.array(accs)


def main():
    ap = argparse.ArgumentParser(description="Plot reliability diagram from evaluate_model output json + predictions csv")
    ap.add_argument("--metrics", type=Path, required=True, help="JSON from evaluate_model --output <path>.json")
    ap.add_argument("--preds", type=Path, required=True, help="CSV produced by train_mvp or evaluate_model including probs and labels")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    with args.metrics.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    import pandas as pd
    df = pd.read_csv(args.preds)
    # Expect columns prob_fake or prob_0..prob_K-1; fall back to prob_fake
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        raise RuntimeError("No prob_* columns in preds CSV")
    probs = df[prob_cols].to_numpy()
    labels = df["label"].to_numpy()

    confs, accs = reliability_curve(probs, labels, n_bins=args.bins)

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=180)
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.plot(confs, accs, 'o-', color='#1f77b4', label='Model')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f"Reliability (ECE={meta['metrics']['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

