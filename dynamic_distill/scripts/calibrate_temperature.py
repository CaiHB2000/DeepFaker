#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def evaluate(logits: np.ndarray, labels: np.ndarray, temperature: float) -> dict[str, float]:
    probs = softmax(logits / temperature)
    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_pos = f1_score(labels, preds, average=None)[1]
    return {"acc": acc, "f1_macro": f1_macro, "f1_pos": f1_pos}


def search_temperature(logits: np.ndarray, labels: np.ndarray, grid: Sequence[float]) -> float:
    best_t = grid[0]
    best_metric = -np.inf
    for t in grid:
        metrics = evaluate(logits, labels, t)
        score = metrics["f1_macro"]
        if score > best_metric:
            best_metric = score
            best_t = t
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature calibration using validation logits")
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-min", type=float, default=0.5)
    parser.add_argument("--grid-max", type=float, default=3.0)
    parser.add_argument("--grid-steps", type=int, default=100)
    args = parser.parse_args()

    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)
    logit_cols = [col for col in val_df.columns if col.startswith("logit_")]
    if not logit_cols:
        raise ValueError("Logit columns not found; run evaluate_model.py with --save-logits")

    val_logits = val_df[logit_cols].values.astype(np.float64)
    val_labels = val_df["label"].values.astype(int)

    grid = np.linspace(args.grid_min, args.grid_max, args.grid_steps)
    best_t = search_temperature(val_logits, val_labels, grid)

    def compute_metrics(df: pd.DataFrame, temperature: float) -> dict[str, float]:
        logits = df[logit_cols].values.astype(np.float64)
        labels = df["label"].values.astype(int)
        return evaluate(logits, labels, temperature)

    val_metrics = compute_metrics(val_df, best_t)
    test_metrics = compute_metrics(test_df, best_t)

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "temperature": best_t,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    # also save calibrated predictions
    probs_test = softmax(test_df[logit_cols].values.astype(np.float64) / best_t)
    preds = probs_test.argmax(axis=1)
    calibrated = test_df.copy()
    calibrated["prediction_calibrated"] = preds
    for idx, col in enumerate(logit_cols):
        calibrated[f"calibrated_prob_{idx}"] = probs_test[:, idx]
    calibrated_path = args.output.parent / f"{args.output.stem}_calibrated.csv"
    calibrated.to_csv(calibrated_path, index=False)


if __name__ == "__main__":
    main()
