# -*- coding: utf-8 -*-
"""
Build consolidated feature table (posts + provenance + propagation) and compute
correlations against DeepFakeBench detector outputs.

Example:
    python -m p2p.tools.analyze_correlations \
        --posts-summary tmp/reddit_seed_run_mass/posts_summary_with_dfbench_multi.csv \
        --provenance tmp/reddit_seed_run_mass/provenance_media.csv \
        --inventory tmp/reddit_seed_run_mass/media_inventory.csv \
        --propagation tmp/reddit_seed_run_mass/propagation_features.csv \
        --out-dir tmp/reddit_seed_run_mass/analysis
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DFBENCH_METRICS = [
    "dfbench_prob_mean_mean",
    "dfbench_prob_mean_max",
    "dfbench_prob_mean_min",
    "dfbench_prob_mean_std",
    "dfbench_prob_max",
    "dfbench_any_pred_fake",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Correlate provenance/propagation features with detector outputs.")
    ap.add_argument("--posts-summary", required=True, help="posts_summary_with_dfbench*.csv path.")
    ap.add_argument("--provenance", required=True, help="provenance_media.csv path.")
    ap.add_argument("--inventory", required=True, help="media_inventory.csv path.")
    ap.add_argument("--propagation", default=None, help="Optional propagation_features.csv path.")
    ap.add_argument("--out-dir", required=True, help="Directory to save analysis CSV/plots.")
    ap.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum paired samples required to compute correlation.",
    )
    return ap.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_int(value: Optional[str]) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def aggregate_provenance(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "total": 0,
            "c2pa_present_sum": 0,
            "c2pa_valid_sum": 0,
            "pdq_present_sum": 0,
            "vpdq_frames": [],
        }
    )
    for row in rows:
        pid = row.get("posting_id", "")
        if not pid:
            continue
        entry = stats[pid]
        entry["total"] += 1
        if row.get("c2pa_present") == "1":
            entry["c2pa_present_sum"] += 1
        if row.get("c2pa_valid") == "1":
            entry["c2pa_valid_sum"] += 1
        if row.get("pdq_available") == "1":
            entry["pdq_present_sum"] += 1
        vpdq = parse_float(row.get("vpdq_frames"))
        if vpdq is not None:
            entry["vpdq_frames"].append(vpdq)
    return stats


def aggregate_inventory(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "total": 0,
            "sizes": [],
            "kind_counts": defaultdict(int),
            "errors": 0,
        }
    )
    for row in rows:
        pid = row.get("posting_id", "")
        if not pid:
            continue
        entry = stats[pid]
        entry["total"] += 1
        size_val = parse_float(row.get("size_bytes"))
        if size_val is not None:
            entry["sizes"].append(size_val)
        kind = (row.get("kind") or "").strip().lower()
        if kind:
            entry["kind_counts"][kind] += 1
        if row.get("errors"):
            entry["errors"] += 1
    return stats


def load_propagation(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}
    mapping: Dict[str, Dict[str, str]] = {}
    rows = read_csv(path)
    for row in rows:
        pid = row.get("posting_id")
        if pid:
            mapping[pid] = row
    return mapping


def fmt_float(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_feature_rows(
    posts: List[Dict[str, str]],
    prov_stats: Dict[str, Dict[str, object]],
    inv_stats: Dict[str, Dict[str, object]],
    prop_rows: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    out_rows: List[Dict[str, str]] = []
    for post in posts:
        pid = post.get("posting_id")
        if not pid:
            continue
        row = dict(post)

        # Provenance aggregates
        pstats = prov_stats.get(pid, {})
        total_prov = pstats.get("total") or 0
        c2pa_present_sum = pstats.get("c2pa_present_sum", 0)
        c2pa_valid_sum = pstats.get("c2pa_valid_sum", 0)
        pdq_present_sum = pstats.get("pdq_present_sum", 0)
        vpdq_frames = pstats.get("vpdq_frames", [])
        row.update(
            {
                "c2pa_present_ratio": fmt_float(c2pa_present_sum / total_prov) if total_prov else "",
                "c2pa_present_any": "1" if c2pa_present_sum else ("0" if total_prov else ""),
                "c2pa_present_sum": str(c2pa_present_sum if total_prov else 0),
                "c2pa_valid_ratio": fmt_float(c2pa_valid_sum / total_prov) if total_prov else "",
                "c2pa_valid_any": "1" if c2pa_valid_sum else ("0" if total_prov else ""),
                "c2pa_valid_sum": str(c2pa_valid_sum if total_prov else 0),
                "pdq_present_ratio": fmt_float(pdq_present_sum / total_prov) if total_prov else "",
                "pdq_present_sum": str(pdq_present_sum if total_prov else 0),
                "vpdq_frames_mean": fmt_float(sum(vpdq_frames) / len(vpdq_frames)) if vpdq_frames else "",
                "vpdq_frames_max": fmt_float(max(vpdq_frames)) if vpdq_frames else "",
            }
        )

        # Inventory aggregates
        istats = inv_stats.get(pid, {})
        total_inv = istats.get("total") or 0
        sizes = istats.get("sizes", [])
        kind_counts = istats.get("kind_counts", {})
        errors_sum = istats.get("errors", 0)
        if sizes:
            sorted_sizes = sorted(sizes)
            median_size = sorted_sizes[len(sorted_sizes) // 2]
        else:
            median_size = None
        row.update(
            {
                "size_bytes_mean": fmt_float(sum(sizes) / len(sizes)) if sizes else "",
                "size_bytes_median": fmt_float(median_size) if median_size is not None else "",
                "size_bytes_max": fmt_float(max(sizes)) if sizes else "",
                "size_bytes_min": fmt_float(min(sizes)) if sizes else "",
                "media_item_count": str(total_inv),
                "share_video": fmt_float(kind_counts.get("video_mp4", 0) / total_inv) if total_inv else "",
                "share_gif": fmt_float(kind_counts.get("gif", 0) / total_inv) if total_inv else "",
                "share_image": fmt_float(kind_counts.get("image", 0) / total_inv) if total_inv else "",
                "has_errors_sum": str(errors_sum if total_inv else 0),
                "error_rate": fmt_float(errors_sum / total_inv) if total_inv else "",
            }
        )

        # Propagation features (direct copy of floats as strings)
        if pid in prop_rows:
            for k, v in prop_rows[pid].items():
                if k == "posting_id":
                    continue
                row[k] = v

        out_rows.append(row)
    return out_rows


def write_feature_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def paired_arrays(
    rows: List[Dict[str, str]],
    feature_key: str,
    dfbench_key: str,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        x = parse_float(row.get(feature_key))
        y = parse_float(row.get(dfbench_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def compute_correlations(
    rows: List[Dict[str, str]],
    dfbench_metrics: List[str],
    min_samples: int,
) -> List[Dict[str, object]]:
    from scipy.stats import pearsonr, spearmanr

    results: List[Dict[str, object]] = []
    for feature in rows[0].keys():
        if feature in dfbench_metrics or feature == "posting_id":
            continue
        if feature.startswith("dfbench_"):
            continue
        # skip non-numeric quick check
        if parse_float(rows[0].get(feature)) is None:
            numeric_candidate = any(parse_float(row.get(feature)) is not None for row in rows)
            if not numeric_candidate:
                continue
        for metric in dfbench_metrics:
            xs, ys = paired_arrays(rows, feature, metric)
            if len(xs) < min_samples:
                continue
            try:
                pr, pp = pearsonr(xs, ys)
            except Exception:
                pr, pp = float("nan"), float("nan")
            try:
                sr, sp = spearmanr(xs, ys)
            except Exception:
                sr, sp = float("nan"), float("nan")
            results.append(
                {
                    "feature": feature,
                    "detector_metric": metric,
                    "n": len(xs),
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_r": sr,
                    "spearman_p": sp,
                    "abs_pearson": abs(pr) if not math.isnan(pr) else float("nan"),
                    "abs_spearman": abs(sr) if not math.isnan(sr) else float("nan"),
                }
            )
    return results


def write_correlations(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "feature",
        "detector_metric",
        "n",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
        "abs_pearson",
        "abs_spearman",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row.get(k)
                if isinstance(v, float):
                    out[k] = fmt_float(v)
                else:
                    out[k] = v
            writer.writerow(out)


def render_top_plot(results: List[Dict[str, object]], out_path: Path, title: str, limit: int = 10) -> None:
    top = sorted(results, key=lambda r: r["abs_spearman"], reverse=True)[:limit]
    if not top:
        return
    labels = [f"{r['feature']} → {r['detector_metric']}" for r in top]
    values = [r["spearman_r"] for r in top]
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in values]
    plt.figure(figsize=(10, max(4, limit / 1.2)))
    positions = range(len(top) - 1, -1, -1)
    plt.barh(list(positions), list(reversed(values)), color=list(reversed(colors)))
    plt.yticks(list(positions), list(reversed(labels)))
    plt.axvline(0.0, color="#444", linewidth=0.8)
    plt.xlabel("Spearman correlation")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def render_scatter(
    rows: List[Dict[str, str]],
    feature: str,
    detector_metric: str,
    out_path: Path,
    spearman: float,
) -> None:
    xs, ys = paired_arrays(rows, feature, detector_metric)
    if len(xs) < 5:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, s=8, alpha=0.4)
    plt.xlabel(feature)
    plt.ylabel(detector_metric)
    plt.title(f"{feature} vs {detector_metric}\nSpearman={spearman:.3f}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = parse_args()
    posts_path = Path(args.posts_summary).resolve()
    prov_path = Path(args.provenance).resolve()
    inv_path = Path(args.inventory).resolve()
    prop_path = Path(args.propagation).resolve() if args.propagation else None
    out_dir = Path(args.out_dir).resolve()

    posts = read_csv(posts_path)
    prov_stats = aggregate_provenance(read_csv(prov_path))
    inv_stats = aggregate_inventory(read_csv(inv_path))
    prop_rows = load_propagation(prop_path)

    feature_rows = build_feature_rows(posts, prov_stats, inv_stats, prop_rows)

    features_csv = out_dir / "posts_with_features.csv"
    write_feature_csv(features_csv, feature_rows)

    correlations = compute_correlations(feature_rows, DFBENCH_METRICS, args.min_samples)
    correlations_sorted = sorted(correlations, key=lambda r: (-r["abs_spearman"], -r["abs_pearson"]))

    corr_csv = out_dir / "provenance_dfbench_correlations.csv"
    write_correlations(corr_csv, correlations_sorted)

    top20 = correlations_sorted[:20]
    top_csv = out_dir / "provenance_dfbench_correlations_top20.csv"
    write_correlations(top_csv, top20)

    render_top_plot(correlations_sorted, out_dir / "fig_top10_spearman.png", "Top Spearman correlations")

    # Highlight strongest propagation-specific correlation if available
    prop_candidates = [r for r in correlations_sorted if r["feature"].startswith(("logistic_", "gompertz_", "hawkes_", "age_hours", "score_rate_per_hour", "comment_rate_per_hour"))]
    if prop_candidates:
        best_prop = prop_candidates[0]
        render_scatter(
            feature_rows,
            best_prop["feature"],
            best_prop["detector_metric"],
            out_dir / "fig_scatter_top_propagation.png",
            best_prop["spearman_r"],
        )

    # General best scatter
    if correlations_sorted:
        render_scatter(
            feature_rows,
            correlations_sorted[0]["feature"],
            correlations_sorted[0]["detector_metric"],
            out_dir / "fig_scatter_top_overall.png",
            correlations_sorted[0]["spearman_r"],
        )

    print(f"[analysis] wrote features -> {features_csv}")
    print(f"[analysis] correlations -> {corr_csv}")
    print(f"[analysis] top20 -> {top_csv}")


if __name__ == "__main__":
    main()
