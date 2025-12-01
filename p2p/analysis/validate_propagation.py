# -*- coding: utf-8 -*-
"""
Validation utilities for propagation-feature correlations.

Example:
    python -m p2p.analysis.validate_propagation \
        --features-csv tmp/reddit_seed_run_mass/analysis/posts_with_features.csv \
        --out-dir tmp/reddit_seed_run_mass/analysis/validation \
        --min-samples 40
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


NUMERIC_FEATURES = [
    "dfbench_prob_mean_std",
    "observed_score_24h",
    "observed_score_6h",
    "score_rate_per_hour",
    "comment_rate_per_hour",
    "logistic_growth_rate",
    "gompertz_growth_rate",
    "hawkes_base_mu",
    "hawkes_peak_intensity",
    "hawkes_early_frac_6h",
]

REGRESSION_FEATURES = [
    "score_rate_per_hour",
    "comment_rate_per_hour",
    "logistic_growth_rate",
    "gompertz_growth_rate",
    "hawkes_base_mu",
    "hawkes_peak_intensity",
    "hawkes_early_frac_6h",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate propagation correlations across slices.")
    ap.add_argument("--features-csv", required=True, help="posts_with_features.csv produced by analyze_correlations.")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory to emit validation report (default alongside CSV).")
    ap.add_argument("--min-samples", type=int, default=40, help="Minimum rows per slice to compute statistics.")
    ap.add_argument("--top-subreddits", type=int, default=6, help="How many subreddits to examine (ordered by coverage).")
    ap.add_argument("--random-seed", type=int, default=42, help="Random seed for regression split.")
    return ap.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "" or str(value).lower() in {"nan", "none"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def spearman_safe(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    if len(xs) < 3 or len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return float("nan"), float("nan")
    r, p = spearmanr(xs, ys)
    return float(r), float(p)


def gather_numeric(rows: Iterable[Dict[str, str]], field: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        v = to_float(row.get(field))
        if v is not None and not math.isnan(v):
            vals.append(v)
    return vals


def subgroup_rows(rows: List[Dict[str, str]], key: str, target: str) -> List[Dict[str, str]]:
    return [row for row in rows if row.get(key) == target]


def compute_correlations(rows: List[Dict[str, str]], feature: str, target: str) -> Dict[str, float]:
    xs, ys = [], []
    for row in rows:
        x = to_float(row.get(feature))
        y = to_float(row.get(target))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    r, p = spearman_safe(xs, ys)
    return {"spearman": r, "p_value": p, "n": len(xs)}


def top_subreddits(rows: List[Dict[str, str]], top_k: int) -> List[str]:
    counter = Counter()
    for row in rows:
        if to_float(row.get("dfbench_prob_mean_std")) is not None:
            counter[row.get("subreddit", "")] += 1
    return [sub for sub, _ in counter.most_common(top_k) if sub]


def media_categories(row: Dict[str, str]) -> str:
    has_video = row.get("has_video") == "1"
    has_image = row.get("has_image") == "1"
    has_gif = row.get("has_gif") == "1"
    if has_video:
        return "video"
    if has_gif:
        return "gif"
    if has_image:
        return "image"
    return "other"


def age_bucket(row: Dict[str, str], quartiles: Tuple[float, float, float]) -> str:
    age = to_float(row.get("age_hours"))
    if age is None:
        return "unknown"
    q1, q2, q3 = quartiles
    if age <= q1:
        return "Q1"
    if age <= q2:
        return "Q2"
    if age <= q3:
        return "Q3"
    return "Q4"


def compute_quartiles(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    sorted_vals = sorted(values)
    q1 = np.quantile(sorted_vals, 0.25)
    q2 = np.quantile(sorted_vals, 0.50)
    q3 = np.quantile(sorted_vals, 0.75)
    return float(q1), float(q2), float(q3)


def fit_regression(rows: List[Dict[str, str]], seed: int) -> Dict[str, float]:
    data = []
    for row in rows:
        y = to_float(row.get("dfbench_prob_mean_std"))
        if y is None:
            continue
        features = []
        valid = True
        for feature in REGRESSION_FEATURES:
            val = to_float(row.get(feature))
            if val is None:
                valid = False
                break
            features.append(val)
        if not valid:
            continue
        data.append((features, y))
    if len(data) < 100:
        return {"train_r2": float("nan"), "test_r2": float("nan"), "samples": len(data)}

    random.Random(seed).shuffle(data)
    split = int(len(data) * 0.8)
    train = data[:split]
    test = data[split:]

    X_train = np.array([row[0] for row in train], dtype=np.float64)
    y_train = np.array([row[1] for row in train], dtype=np.float64)
    X_test = np.array([row[0] for row in test], dtype=np.float64)
    y_test = np.array([row[1] for row in test], dtype=np.float64)

    # standardise
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    coef, _, _, _ = np.linalg.lstsq(X_train_std, y_train, rcond=None)
    y_pred_train = X_train_std @ coef
    y_pred_test = X_test_std @ coef

    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "train_r2": r2(y_train, y_pred_train),
        "test_r2": r2(y_test, y_pred_test),
        "samples": len(data),
        "coef": coef.tolist(),
        "feature_names": REGRESSION_FEATURES,
    }


def main() -> None:
    args = parse_args()
    features_path = Path(args.features_csv).resolve()
    rows = load_rows(features_path)
    out_dir = Path(args.out_dir or os.path.join(features_path.parent, "validation")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_metric = "dfbench_prob_mean_std"
    propagation_feature = "observed_score_24h"

    report = {"slices": {}, "regression": {}}

    # Subreddit slices
    subs = top_subreddits(rows, args.top_subreddits)
    sub_results = []
    for sub in subs:
        subset = subgroup_rows(rows, "subreddit", sub)
        stats = compute_correlations(subset, propagation_feature, target_metric)
        if stats["n"] >= args.min_samples:
            sub_results.append({"subreddit": sub, **stats})
    report["slices"]["subreddits"] = sub_results

    # Media-type slices
    media_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        media_groups[media_categories(row)].append(row)
    media_results = []
    for media, subset in media_groups.items():
        stats = compute_correlations(subset, propagation_feature, target_metric)
        if stats["n"] >= args.min_samples:
            media_results.append({"media": media, **stats})
    report["slices"]["media"] = media_results

    # Age quartiles
    ages = gather_numeric(rows, "age_hours")
    quartiles = compute_quartiles(ages)
    age_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        age_groups[age_bucket(row, quartiles)].append(row)
    age_results = []
    for bucket, subset in age_groups.items():
        stats = compute_correlations(subset, propagation_feature, target_metric)
        if stats["n"] >= args.min_samples:
            age_results.append({"age_bucket": bucket, **stats})
    report["slices"]["age"] = age_results

    # Regression
    report["regression"] = fit_regression(rows, args.random_seed)

    # Save JSON + text summary
    json_path = out_dir / "validation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    text_lines = ["Propagation validation report", f"rows_total={len(rows)}", ""]
    text_lines.append("Subreddit slices:")
    for item in sub_results:
        text_lines.append(
            f"  {item['subreddit']}: spearman={item['spearman']:.3f}, p={item['p_value']:.3g}, n={item['n']}"
        )
    text_lines.append("")
    text_lines.append("Media-type slices:")
    for item in media_results:
        text_lines.append(
            f"  {item['media']}: spearman={item['spearman']:.3f}, p={item['p_value']:.3g}, n={item['n']}"
        )
    text_lines.append("")
    text_lines.append("Age quartile slices:")
    for item in age_results:
        text_lines.append(
            f"  {item['age_bucket']}: spearman={item['spearman']:.3f}, p={item['p_value']:.3g}, n={item['n']}"
        )
    text_lines.append("")
    text_lines.append(
        "Regression (predict dfbench_prob_mean_std from propagation features): "
        f"train_r2={report['regression'].get('train_r2'):.3f}, "
        f"test_r2={report['regression'].get('test_r2'):.3f}, "
        f"samples={report['regression'].get('samples')}"
    )

    txt_path = out_dir / "validation_report.txt"
    txt_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    print("[validate] written report:")
    print("  ", json_path)
    print("  ", txt_path)


if __name__ == "__main__":
    main()
