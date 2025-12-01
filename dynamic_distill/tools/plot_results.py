#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots for dynamic distillation experiments.")
    parser.add_argument("runs", nargs="*", type=Path, help="Run directories containing metrics files.")
    parser.add_argument("--summary", type=Path, default=None, help="Experiment summary CSV/YAML to infer runs.")
    parser.add_argument("--output", type=Path, default=Path("plots"), help="Output directory for figures.")
    parser.add_argument("--title", type=str, default="Dynamic Distillation Ablations", help="Plot title prefix.")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins for calibration curves.")
    return parser.parse_args()


def load_runs(args: argparse.Namespace) -> List[Path]:
    runs: List[Path] = []
    if args.summary is not None:
        summary_path = args.summary
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        if summary_path.suffix.lower() == ".csv":
            df = pd.read_csv(summary_path)
            runs.extend(Path(path) for path in df["run_dir"].dropna().unique())
        else:
            with summary_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            runs.extend(Path(item["run_dir"]) for item in data if "run_dir" in item)
    runs.extend(args.runs)
    if not runs:
        raise ValueError("No runs provided. Specify run directories or a summary file.")
    deduped = []
    for run in runs:
        run = run.resolve()
        if run not in deduped:
            deduped.append(run)
    return deduped


def read_summary(run_dir: Path) -> Dict:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def plot_val_curves(runs: List[Path], output_dir: Path, title_prefix: str) -> None:
    plt.figure(figsize=(8, 5))
    for run in runs:
        val_csv = run / "val_metrics.csv"
        if not val_csv.exists():
            continue
        df = pd.read_csv(val_csv)
        plt.plot(df["epoch"], df["f1_macro"], marker="o", label=run.name)
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title(f"{title_prefix} – Validation Macro F1")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "val_f1_curves.png", dpi=300)
    plt.close()


def plot_test_bars(runs: List[Path], output_dir: Path, title_prefix: str) -> None:
    labels = []
    f1_scores = []
    acc_scores = []
    for run in runs:
        summary = read_summary(run)
        test_metrics = summary.get("test_metrics") or {}
        if not test_metrics:
            continue
        labels.append(run.name)
        f1_scores.append(test_metrics.get("f1_macro"))
        acc_scores.append(test_metrics.get("acc"))
    if not labels:
        return
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, acc_scores, width=width, label="Accuracy")
    plt.bar(x + width / 2, f1_scores, width=width, label="Macro F1")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{title_prefix} – Test Metrics")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "test_metrics_bar.png", dpi=300)
    plt.close()


def calibration_curve(df: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc = [], []
    for idx in range(n_bins):
        lower, upper = bins[idx], bins[idx + 1]
        mask = (df["confidence"] > lower) & (df["confidence"] <= upper)
        if mask.any():
            conf.append(df.loc[mask, "confidence"].mean())
            acc.append(df.loc[mask, "correct"].mean())
    return np.array(conf), np.array(acc)


def plot_calibration(runs: List[Path], output_dir: Path, title_prefix: str, bins: int) -> None:
    plt.figure(figsize=(6, 6))
    plotted = False
    for run in runs:
        preds_csv = run / "test_predictions.csv"
        if not preds_csv.exists():
            continue
        df = pd.read_csv(preds_csv)
        conf, acc = calibration_curve(df, bins)
        if len(conf) == 0:
            continue
        plt.plot(conf, acc, marker="o", label=run.name)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title(f"{title_prefix} – Reliability Diagram")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "calibration_curve.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    runs = load_runs(args)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_val_curves(runs, output_dir, args.title)
    plot_test_bars(runs, output_dir, args.title)
    plot_calibration(runs, output_dir, args.title, args.bins)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
