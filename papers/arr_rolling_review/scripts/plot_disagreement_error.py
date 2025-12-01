#!/usr/bin/env python3
"""
Plot cross-modal disagreement vs error rate for Weibo and WeFEND.
Disagreement = |p_text(fake) - p_image(fake)|.
Error = student (DMPD) prediction != gold.

Outputs:
  - figures/disagreement_error.pdf (two subplots)
  - paper_results/<dataset>_disagreement_error.csv (bucketed stats)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]


def load_probs(path: Path):
    df = pd.read_csv(path)
    if "prob_fake" in df.columns:
        p_fake = df["prob_fake"].values
    elif "prob_1" in df.columns:
        p_fake = df["prob_1"].values
    else:
        raise ValueError(f"probability column not found in {path}")
    return df[["id", "label"]].copy(), p_fake


def bucket_disagreement(text_csv, img_csv, student_csv, out_csv, n_bins=10):
    df_t, p_t = load_probs(text_csv)
    df_v, p_v = load_probs(img_csv)
    df_s, p_s = load_probs(student_csv)

    # join on id
    df = df_t.merge(df_v, on="id", suffixes=("_text", "_vision"))
    df = df.merge(df_s, on="id", suffixes=("", "_student"))
    # ensure labels align
    df = df[df["label_text"] == df["label"]]
    labels = df["label"].to_numpy()
    # disagreement
    p_text = p_t[df_t.index[df_t["id"].isin(df["id"])]]
    p_img = p_v[df_v.index[df_v["id"].isin(df["id"])]]
    p_student = p_s[df_s.index[df_s["id"].isin(df["id"])]]
    disagree = np.abs(p_text - p_img)
    # student error
    preds_student = (p_student >= 0.5).astype(int)
    errors = (preds_student != labels.astype(int)).astype(int)

    # quantile buckets
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(disagree, quantiles)
    # ensure increasing with small epsilon
    edges[0] = disagree.min()
    edges[-1] = disagree.max()

    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        idx = (disagree >= lo) & (disagree <= hi if b == n_bins - 1 else disagree < hi)
        n = idx.sum()
        if n == 0:
            err_rate = 0.0
        else:
            err_rate = errors[idx].mean()
        rows.append(
            {
                "bucket": b,
                "n": n,
                "disagree_min": float(lo),
                "disagree_max": float(hi),
                "error_rate": float(err_rate),
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows


def main():
    configs = [
        dict(
            name="Weibo",
            text_csv=ROOT
            / "paper_results/weibo_text_only/weibo_text_only_seed00/test_predictions.csv",
            img_csv=ROOT
            / "paper_results/weibo_image_only/weibo_image_only_seed00/test_predictions.csv",
            student_csv=ROOT
            / "paper_results/weibo_dynamic_distill/weibo_dynamic_distill_seed01/test_predictions.csv",
            out_csv=ROOT / "paper_results/weibo_disagreement_error.csv",
        ),
        dict(
            name="WeFEND",
            text_csv=ROOT
            / "paper_results/wefend_text_only/wefend_text_only_seed00/test_predictions.csv",
            img_csv=ROOT
            / "paper_results/wefend_image_only_seed00/wefend_image_only_seed00/test_predictions.csv",
            student_csv=ROOT
            / "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2/pseudolabel_dynamic_softquota_seed00/test_predictions.csv",
            out_csv=ROOT / "paper_results/wefend_disagreement_error.csv",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), dpi=200, sharey=True)
    colors = ["#1f77b4", "#ff7f0e"]

    for ax, cfg, color in zip(axes, configs, colors):
        rows = bucket_disagreement(
            cfg["text_csv"], cfg["img_csv"], cfg["student_csv"], cfg["out_csv"]
        )
        xs = np.arange(len(rows))
        ys = [r["error_rate"] for r in rows]
        ax.plot(xs, ys, "o-", color=color, linewidth=1.6, markersize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i + 1) for i in xs], fontsize=7)
        ax.set_xlabel("Disagreement decile (low → high)", fontsize=9)
        ax.set_title(cfg["name"], fontsize=10)
        ax.grid(alpha=0.3, linewidth=0.5)
        if ax is axes[0]:
            ax.set_ylabel("Student error rate", fontsize=9)
        ax.set_ylim(0, max(ys) * 1.2 + 0.01)

    plt.tight_layout()
    out_fig = ROOT / "papers" / "arr_rolling_review" / "figures" / "disagreement_error.pdf"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
