# -*- coding: utf-8 -*-
"""
Join DeepFakeBench content scores back into the post-level summary table.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Join dfbench_content_scores.csv into posts_summary.csv.")
    ap.add_argument("--posts-summary", required=True, help="Path to posts_summary.csv.")
    ap.add_argument("--content-map", required=True, help="Path to content_id_map.csv (maps content_id -> posting_id).")
    ap.add_argument("--content-scores", required=True, help="Path to dfbench_content_scores.csv.")
    ap.add_argument("--out-path", type=str, default=None, help="Output CSV path (default: <posts_summary> with _with_dfbench suffix).")
    ap.add_argument("--prob-threshold", type=float, default=0.5, help="Probability threshold to flag a post as suspicious.")
    return ap


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def read_content_scores(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv(path)
    return {row.get("content_id", ""): row for row in rows if row.get("content_id")}


def build_post_to_content_map(path: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    rows = read_csv(path)
    for row in rows:
        cid = row.get("content_id", "")
        pid = row.get("posting_id", "")
        if not cid or not pid:
            continue
        mapping.setdefault(pid, []).append(cid)
    return mapping


def compute_post_metrics(
    content_ids: List[str],
    content_scores: Dict[str, Dict[str, str]],
    prob_threshold: float,
) -> Tuple[Dict[str, str], List[str]]:
    scored = []
    missing = []
    for cid in content_ids:
        if cid in content_scores:
            scored.append(content_scores[cid])
        else:
            missing.append(cid)

    if not scored:
        return {
            "dfbench_scored_contents": "0",
            "dfbench_prob_mean_mean": "",
            "dfbench_prob_mean_max": "",
            "dfbench_prob_mean_min": "",
            "dfbench_prob_mean_std": "",
            "dfbench_prob_max": "",
            "dfbench_any_pred_fake": "0",
            "dfbench_pred_label_detail": "",
            "dfbench_flag": "0",
        }, missing

    mean_vals = [parse_float(row.get("prob_mean")) for row in scored]
    max_vals = [parse_float(row.get("prob_max")) for row in scored]
    std_vals = [parse_float(row.get("prob_std")) for row in scored]
    pred_labels = [row.get("pred_label", "0") for row in scored]

    any_fake = any(label == "1" for label in pred_labels)

    detail_parts = []
    detector_counts = []
    detector_detail_parts = []
    for row in scored:
        cid = row.get("content_id", "")
        prob = parse_float(row.get("prob_mean"))
        label = row.get("pred_label", "0")
        detail_parts.append(f"{cid}:{prob:.6f}:{label}")
        detector_counts.append(int(parse_float(row.get("num_detectors"), 0)))
        det_detail = row.get("detector_detail")
        if det_detail:
            detector_detail_parts.append(f"{cid}=>{det_detail}")

    metrics = {
        "dfbench_scored_contents": str(len(scored)),
        "dfbench_prob_mean_mean": f"{mean(mean_vals):.6f}",
        "dfbench_prob_mean_max": f"{max(mean_vals):.6f}",
        "dfbench_prob_mean_min": f"{min(mean_vals):.6f}",
        "dfbench_prob_mean_std": f"{mean(std_vals):.6f}",
        "dfbench_prob_max": f"{max(max_vals):.6f}",
        "dfbench_any_pred_fake": "1" if any_fake else "0",
        "dfbench_pred_label_detail": "|".join(detail_parts),
        "dfbench_flag": "1" if any_fake or max(mean_vals) >= prob_threshold else "0",
        "dfbench_detector_detail": " || ".join(detector_detail_parts),
        "dfbench_max_detectors": str(max(detector_counts) if detector_counts else 0),
    }
    return metrics, missing


def join_scores(
    posts_rows: List[Dict[str, str]],
    content_scores: Dict[str, Dict[str, str]],
    post_to_content: Dict[str, List[str]],
    prob_threshold: float,
) -> Tuple[List[Dict[str, str]], int, int]:
    updated_rows: List[Dict[str, str]] = []
    total_missing = 0
    posts_with_scores = 0

    for row in posts_rows:
        posting_id = row.get("posting_id", "")
        content_ids_field = row.get("content_ids", "")

        content_ids = [cid for cid in content_ids_field.split("|") if cid]
        if not content_ids and posting_id in post_to_content:
            content_ids = post_to_content.get(posting_id, [])

        metrics, missing = compute_post_metrics(content_ids, content_scores, prob_threshold=prob_threshold)
        if metrics["dfbench_scored_contents"] != "0":
            posts_with_scores += 1

        total_missing += len(missing)

        row_with_metrics = dict(row)
        row_with_metrics.update(metrics)
        row_with_metrics["dfbench_missing_content_ids"] = "|".join(missing)
        updated_rows.append(row_with_metrics)

    return updated_rows, posts_with_scores, total_missing


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    ap = build_argparser()
    args = ap.parse_args()

    posts_path = Path(args.posts_summary).resolve()
    content_scores_path = Path(args.content_scores).resolve()
    content_map_path = Path(args.content_map).resolve()

    if args.out_path:
        out_path = Path(args.out_path).resolve()
    else:
        stem = posts_path.stem + "_with_dfbench"
        out_path = posts_path.with_name(f"{stem}{posts_path.suffix}")

    posts_rows = read_csv(posts_path)
    content_scores = read_content_scores(content_scores_path)
    post_to_content = build_post_to_content_map(content_map_path)

    updated_rows, posts_with_scores, total_missing = join_scores(
        posts_rows,
        content_scores,
        post_to_content,
        prob_threshold=args.prob_threshold,
    )

    all_fields = list(posts_rows[0].keys()) if posts_rows else []
    new_fields = [
        "dfbench_scored_contents",
        "dfbench_prob_mean_mean",
        "dfbench_prob_mean_max",
        "dfbench_prob_mean_min",
        "dfbench_prob_mean_std",
        "dfbench_prob_max",
        "dfbench_any_pred_fake",
        "dfbench_pred_label_detail",
        "dfbench_flag",
        "dfbench_detector_detail",
        "dfbench_max_detectors",
        "dfbench_missing_content_ids",
    ]
    fieldnames = all_fields + [f for f in new_fields if f not in all_fields]

    write_csv(out_path, updated_rows, fieldnames)

    print(f"[join] posts updated → {out_path}")
    print(f"[join] posts with DFBench scores: {posts_with_scores}/{len(posts_rows)}")
    print(f"[join] missing content IDs (no score): {total_missing}")


if __name__ == "__main__":
    main()
