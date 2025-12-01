#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


def evaluate(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs[:, 1] >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_pos = f1_score(labels, preds, average=None)[1]
    return {"acc": acc, "f1_macro": f1_macro, "f1_pos": f1_pos}


def search_threshold(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> float:
    best_th = thresholds[0]
    best_score = -np.inf
    for th in thresholds:
        metrics = evaluate(probs, labels, th)
        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_th = th
    return float(best_th)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probability threshold calibration")
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)
    prob_cols = [c for c in val_df.columns if c.startswith("prob_") and not c.startswith("prob_calibrated")]
    if len(prob_cols) < 2:
        raise ValueError("probability columns not found")

    val_probs = val_df[prob_cols].values.astype(np.float64)
    val_labels = val_df["label"].values.astype(int)
    thresholds = np.linspace(0.3, 0.9, args.steps)
    best_th = search_threshold(val_probs, val_labels, thresholds)

    def metrics(df: pd.DataFrame, threshold: float) -> dict[str, float]:
        probs = df[prob_cols].values.astype(np.float64)
        labels = df["label"].values.astype(int)
        return evaluate(probs, labels, threshold)

    val_metrics = metrics(val_df, best_th)
    test_metrics = metrics(test_df, best_th)

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "threshold": best_th,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    probs_test = test_df[prob_cols].values.astype(np.float64)
    preds = (probs_test[:, 1] >= best_th).astype(int)
    calibrated = test_df.copy()
    calibrated["prediction_threshold"] = preds
    calibrated_path = args.output.parent / f"{args.output.stem}_threshold.csv"
    calibrated.to_csv(calibrated_path, index=False)


if __name__ == "__main__":
    main()
