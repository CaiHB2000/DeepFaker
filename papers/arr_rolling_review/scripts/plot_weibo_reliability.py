#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[3]
PLOTS_ROOT = ROOT / "paper_results" / "weibo_plots"
OUT_FIG = ROOT / "papers" / "arr_rolling_review" / "figures" / "reliability_main.pdf"


def load_conf_and_labels(base: str):
    metrics_path = PLOTS_ROOT / f"{base}.json"
    csv_path = PLOTS_ROOT / f"{base}.csv"
    with metrics_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_csv(csv_path)
    logit_cols = [c for c in df.columns if c.startswith("logit_")]
    logits = torch.from_numpy(df[logit_cols].to_numpy()).float()
    labels = df["label"].to_numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    confidences = probs.max(axis=1)
    ece = meta["metrics"].get("ece", None)
    return confidences, labels, ece


def main():
    # Load confidence and labels for each model
    conf_eann, y_eann, ece_eann = load_conf_and_labels("weibo_eann_eval")
    conf_safe, y_safe, ece_safe = load_conf_and_labels("weibo_safe_eval")
    conf_dmpd, y_dmpd, ece_dmpd = load_conf_and_labels("weibo_dmpd_eval")

    # Uniform-bin calibration curves (binary task; confidences in [0.5,1])
    n_bins = 10
    true_eann, pred_eann = calibration_curve(y_eann, conf_eann, n_bins=n_bins, strategy="uniform")
    true_safe, pred_safe = calibration_curve(y_safe, conf_safe, n_bins=n_bins, strategy="uniform")
    true_dmpd, pred_dmpd = calibration_curve(y_dmpd, conf_dmpd, n_bins=n_bins, strategy="uniform")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=180, sharex=True, sharey=True)

    # Helper to plot a single model vs diagonal
    def plot_single(ax, preds, trues, title, ece):
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.plot(preds, trues, "o-", color="#1f77b4", markersize=3, linewidth=1)
        ax.set_title(f"{title} (ECE={ece:.3f})", fontsize=8)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.5, 1.0)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=7)

    plot_single(axes[0], pred_eann, true_eann, "EANN", ece_eann)
    plot_single(axes[1], pred_safe, true_safe, "SAFE", ece_safe)
    plot_single(axes[2], pred_dmpd, true_dmpd, "DMPD", ece_dmpd)

    axes[0].set_ylabel("Accuracy", fontsize=8)
    for ax in axes:
        ax.set_xlabel("Confidence", fontsize=8)

    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
