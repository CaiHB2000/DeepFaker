# -*- coding: utf-8 -*-
"""
Estimate propagation-related features (early engagement buckets, logistic/Gompertz growth proxies,
and Hawkes-process style parameters) for Reddit posts harvested by run_reddit_seed.

Input: reddit_posts.jsonl
Output: propagation_features.csv (per posting_id)

Usage example:
    python -m p2p.tools.propagation_features \
        --posts-jsonl tmp/reddit_seed_run_mass/reddit_posts.jsonl \
        --out-csv tmp/reddit_seed_run_mass/propagation_features.csv
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

CSV_FIELDS = [
    "posting_id",
    "age_hours",
    "score",
    "num_comments",
    "upvote_ratio",
    "engagement_total",
    "score_rate_per_hour",
    "comment_rate_per_hour",
    "engagement_rate_per_hour",
    "logistic_carrying_capacity",
    "logistic_growth_rate",
    "logistic_pred_1h",
    "logistic_pred_6h",
    "logistic_pred_24h",
    "logistic_early_frac_1h",
    "logistic_early_frac_6h",
    "logistic_early_frac_24h",
    "gompertz_growth_rate",
    "gompertz_pred_1h",
    "gompertz_pred_6h",
    "gompertz_pred_24h",
    "gompertz_early_frac_1h",
    "gompertz_early_frac_6h",
    "gompertz_early_frac_24h",
    "hawkes_base_mu",
    "hawkes_branching_ratio",
    "hawkes_peak_intensity",
    "hawkes_expected_count_6h",
    "hawkes_expected_count_24h",
    "hawkes_early_frac_6h",
    "hawkes_early_frac_24h",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute propagation feature proxies from reddit_posts.jsonl.")
    ap.add_argument("--posts-jsonl", required=True, help="Path to reddit_posts.jsonl emitted by run_reddit_seed.")
    ap.add_argument(
        "--snapshot-utc",
        type=float,
        default=None,
        help="Override snapshot timestamp (UTC seconds). Defaults to max(created_utc)+3600.",
    )
    ap.add_argument(
        "--min-age-minutes",
        type=float,
        default=10.0,
        help="Minimum post age (minutes) to avoid division by zero when estimating rates.",
    )
    ap.add_argument(
        "--k-multiplier",
        type=float,
        default=1.5,
        help="Multiplier for logistic/Gompertz carrying capacity K relative to observed score.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output CSV path (default: same directory as posts_jsonl, file name propagation_features.csv).",
    )
    return ap.parse_args()


def read_posts(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def safe_posting_id(data: Dict[str, object]) -> Optional[str]:
    pid = data.get("posting_id")
    if isinstance(pid, str) and pid:
        return pid
    post_id = data.get("post_id")
    platform = data.get("platform", "reddit")
    if isinstance(post_id, str) and post_id:
        return f"{platform}:{post_id}"
    return None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ensure_positive(value: float, default: float) -> float:
    return value if value > 0.0 else default


def estimate_logistic(score: float, age_hours: float, k_multiplier: float, n0: float = 1.0) -> Dict[str, float]:
    """
    Estimate logistic growth rate r and carrying capacity K given observed score at age t.
    Uses analytic solution of logistic equation with initial seeds n0.
    """
    t = ensure_positive(age_hours, 1.0 / 12.0)
    n_obs = max(score, 0.0)
    k = max(n_obs * k_multiplier + 1.0, n_obs + 1.0, n0 + 1.0)
    a = (k - n0) / n0
    numerator = k - n_obs
    denom = max(n_obs * a, 1e-6)
    ratio = clamp(numerator / denom, 1e-6, 1e6)
    r = max(0.0, -(1.0 / t) * math.log(ratio))
    return {"r": r, "K": k, "A": a}


def logistic_predict(tau: float, params: Dict[str, float], n0: float = 1.0) -> float:
    r, k, a = params["r"], params["K"], params["A"]
    return k / (1.0 + a * math.exp(-r * tau))


def estimate_gompertz(score: float, age_hours: float, k_multiplier: float, n0: float = 1.0) -> Dict[str, float]:
    """
    Estimate Gompertz growth parameter g (using b = ln(K/N0)).
    """
    t = ensure_positive(age_hours, 1.0 / 12.0)
    n_obs = max(score, 0.0)
    k = max(n_obs * k_multiplier + 1.0, n_obs + 1.0, n0 + 1.0)
    b = math.log(k / n0)
    ratio_raw = -math.log(clamp(n_obs / k, 1e-9, 1.0 - 1e-6))
    ratio = clamp(ratio_raw / b, 1e-6, 0.999999)
    g = max(0.0, -(1.0 / t) * math.log(ratio))
    return {"g": g, "K": k, "b": b}


def gompertz_predict(tau: float, params: Dict[str, float], n0: float = 1.0) -> float:
    g, k, b = params["g"], params["K"], params["b"]
    return k * math.exp(-b * math.exp(-g * tau))


def estimate_hawkes(score: float, num_comments: float, age_hours: float) -> Dict[str, float]:
    """
    Simple Hawkes-process proxy:
      - branching ratio α approximated by share of comments in total engagement
      - base rate μ chosen so that expected events μ / (1 - α) ≈ score_rate_per_hour
    """
    t = ensure_positive(age_hours, 1.0 / 12.0)
    score_rate = score / t
    total_eng = max(score + num_comments, 0.0)
    alpha = clamp(num_comments / max(total_eng, 1e-6), 0.0, 0.95)
    mu = max(score_rate * (1.0 - alpha), 0.0)
    peak = mu / max(1.0 - alpha, 1e-6)
    return {"alpha": alpha, "mu": mu, "peak": peak}


def hawkes_expected(mu: float, alpha: float, tau_hours: float) -> float:
    """
    Expected cumulative count for a linear Hawkes process with exponential kernel
    approximated via mu * tau / (1 - alpha).
    """
    return mu * tau_hours / max(1.0 - alpha, 1e-6)


def compute_features(
    posts: Iterable[Dict[str, object]],
    snapshot_ts: float,
    min_age_minutes: float,
    k_multiplier: float,
) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    min_age_hours = ensure_positive(min_age_minutes / 60.0, 1.0 / 60.0)
    for post in posts:
        posting_id = safe_posting_id(post)
        if not posting_id:
            continue
        created = float(post.get("created_utc") or snapshot_ts)
        age_hours = max((snapshot_ts - created) / 3600.0, min_age_hours)
        score = float(post.get("score") or 0.0)
        num_comments = float(post.get("num_comments") or 0.0)
        upvote_ratio = float(post.get("upvote_ratio") or 0.0)
        engagement = score + num_comments
        score_rate = score / age_hours
        comment_rate = num_comments / age_hours
        engagement_rate = engagement / age_hours

        logistic_params = estimate_logistic(score, age_hours, k_multiplier)
        gompertz_params = estimate_gompertz(score, age_hours, k_multiplier)
        hawkes_params = estimate_hawkes(score, num_comments, age_hours)

        buckets = [1.0, 6.0, 24.0]
        logistic_preds = {}
        gompertz_preds = {}
        for tau in buckets:
            logistic_preds[tau] = logistic_predict(min(tau, age_hours), logistic_params)
            gompertz_preds[tau] = gompertz_predict(min(tau, age_hours), gompertz_params)

        hawkes_expected_6h = hawkes_expected(hawkes_params["mu"], hawkes_params["alpha"], min(6.0, age_hours))
        hawkes_expected_24h = hawkes_expected(hawkes_params["mu"], hawkes_params["alpha"], min(24.0, age_hours))

        def frac(val: float) -> float:
            if score <= 0.0:
                return 0.0
            return clamp(val / score, 0.0, 1.5)

        row = {
            "posting_id": posting_id,
            "age_hours": round(age_hours, 6),
            "score": score,
            "num_comments": num_comments,
            "upvote_ratio": clamp(upvote_ratio, 0.0, 1.0),
            "engagement_total": engagement,
            "score_rate_per_hour": score_rate,
            "comment_rate_per_hour": comment_rate,
            "engagement_rate_per_hour": engagement_rate,
            "logistic_carrying_capacity": logistic_params["K"],
            "logistic_growth_rate": logistic_params["r"],
            "logistic_pred_1h": logistic_preds[1.0],
            "logistic_pred_6h": logistic_preds[6.0],
            "logistic_pred_24h": logistic_preds[24.0],
            "logistic_early_frac_1h": frac(logistic_preds[1.0]),
            "logistic_early_frac_6h": frac(logistic_preds[6.0]),
            "logistic_early_frac_24h": frac(logistic_preds[24.0]),
            "gompertz_growth_rate": gompertz_params["g"],
            "gompertz_pred_1h": gompertz_preds[1.0],
            "gompertz_pred_6h": gompertz_preds[6.0],
            "gompertz_pred_24h": gompertz_preds[24.0],
            "gompertz_early_frac_1h": frac(gompertz_preds[1.0]),
            "gompertz_early_frac_6h": frac(gompertz_preds[6.0]),
            "gompertz_early_frac_24h": frac(gompertz_preds[24.0]),
            "hawkes_base_mu": hawkes_params["mu"],
            "hawkes_branching_ratio": hawkes_params["alpha"],
            "hawkes_peak_intensity": hawkes_params["peak"],
            "hawkes_expected_count_6h": hawkes_expected_6h,
            "hawkes_expected_count_24h": hawkes_expected_24h,
            "hawkes_early_frac_6h": frac(hawkes_expected_6h),
            "hawkes_early_frac_24h": frac(hawkes_expected_24h),
        }
        out.append(row)
    return out


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(CSV_FIELDS) + "\n")
        for row in rows:
            values: List[str] = []
            for field in CSV_FIELDS:
                value = row.get(field, "")
                if isinstance(value, float):
                    values.append(f"{value:.6f}".rstrip("0").rstrip("."))
                else:
                    values.append(str(value))
            f.write(",".join(values) + "\n")


def main() -> None:
    args = parse_args()
    posts_path = Path(args.posts_jsonl).resolve()
    posts = read_posts(posts_path)
    if not posts:
        raise RuntimeError(f"No posts loaded from {posts_path}")

    if args.snapshot_utc is not None:
        snapshot_ts = float(args.snapshot_utc)
    else:
        max_created = max(float(p.get("created_utc") or time.time()) for p in posts)
        snapshot_ts = max_created + 3600.0  # assume scrape happens within ~1 hour of last post

    features = compute_features(
        posts=posts,
        snapshot_ts=snapshot_ts,
        min_age_minutes=args.min_age_minutes,
        k_multiplier=args.k_multiplier,
    )

    out_csv = Path(args.out_csv) if args.out_csv else posts_path.with_name("propagation_features.csv")
    write_csv(out_csv, features)
    print(f"[propagation] wrote {len(features)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
