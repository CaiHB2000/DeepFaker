# -*- coding: utf-8 -*-
"""
Run multiple DeepFakeBench detectors on the Reddit dataset and aggregate scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import yaml

from p2p.runners.run_dfbench_capsule import (
    prepare_dataset,
    read_content_csv,
    run_dfbench_test,
    parse_predictions,
    aggregate_scores,
    write_content_scores,
    write_post_scores,
)
# Default detector configs tailored for the Reddit dataset
SAFE_DETECTOR_SEQUENCE = [
    ("capsule", "DeepFakeBench/training/config/detector/capsule_net_reddit.yaml"),
    ("xception", "DeepFakeBench/training/config/detector/xception_reddit.yaml"),
    ("core", "DeepFakeBench/training/config/detector/core_reddit.yaml"),
    ("srm", "DeepFakeBench/training/config/detector/srm_reddit.yaml"),
    ("recce", "DeepFakeBench/training/config/detector/recce_reddit.yaml"),
    ("meso4", "DeepFakeBench/training/config/detector/meso4_reddit.yaml"),
    ("meso4inception", "DeepFakeBench/training/config/detector/meso4Inception_reddit.yaml"),
    ("ffd", "DeepFakeBench/training/config/detector/ffd_reddit.yaml"),
    ("efficientnetb4", "DeepFakeBench/training/config/detector/efficientnetb4_reddit.yaml"),
]
OPTIONAL_DETECTORS = {
    "spsl": "DeepFakeBench/training/config/detector/spsl_reddit.yaml",
    "f3net": "DeepFakeBench/training/config/detector/f3net_reddit.yaml",
}
DEFAULT_DETECTORS = {name: path for name, path in SAFE_DETECTOR_SEQUENCE}
DEFAULT_DETECTORS.update(OPTIONAL_DETECTORS)
SAFE_DETECTOR_NAMES = [name for name, _ in SAFE_DETECTOR_SEQUENCE]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run multiple DeepFakeBench detectors and aggregate results.")
    ap.add_argument("--work-dir", type=str, default="tmp/reddit_seed_run", help="Workspace root containing media and manifests.")
    ap.add_argument("--media-root", type=str, default=None, help="Media root (defaults to work dir).")
    ap.add_argument("--kept-csv", type=str, default=None, help="kept_content.csv path (defaults to <work-dir>/kept_content.csv).")
    ap.add_argument("--content-map", type=str, default=None, help="content_id_map.csv path (defaults to <work-dir>/content_id_map.csv).")
    ap.add_argument("--dfb-root", type=str, default="DeepFakeBench", help="DeepFakeBench repository root.")
    ap.add_argument(
        "--detectors",
        type=str,
        default=",".join(SAFE_DETECTOR_NAMES),
        help="Comma-separated list of detector names to run (use 'all' for the default safe set).",
    )
    ap.add_argument(
        "--detector-config",
        action="append",
        default=[],
        help="Additional detector config YAML paths to include.",
    )
    ap.add_argument("--dataset-name", type=str, default="reddit_canonical", help="Dataset name inside DeepFakeBench.")
    ap.add_argument("--label-name", type=str, default="reddit_unknown", help="Label name used in dataset JSON.")
    ap.add_argument("--max-frames", type=int, default=32, help="Max frames to sample per video.")
    ap.add_argument("--image-size", type=int, default=256, help="Frame resize dimension.")
    ap.add_argument("--prob-threshold", type=float, default=0.5, help="Probability threshold for detector-level flagging.")
    ap.add_argument(
        "--min-positive-detectors",
        type=int,
        default=1,
        help="Minimum number of detectors that must predict fake for the content/post to be flagged.",
    )
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory (defaults to <work-dir>/dfbench).")
    ap.add_argument("--skip-join", action="store_true", help="Skip writing per-detector content/post CSVs.")
    ap.add_argument("--join-posts-summary", type=str, default=None, help="Optional posts_summary.csv path to join aggregated scores into.")
    ap.add_argument("--join-output", type=str, default=None, help="Optional output path for joined posts summary.")
    return ap


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sanitize_name(config_path: Path, cfg: Dict) -> str:
    name = cfg.get("model_name") or config_path.stem
    # normalize (remove suffixes)
    name = name.replace("_detector", "")
    name = name.replace("_net", "")
    name = name.replace("_reddit", "")
    return name


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_overall_content_metrics(
    detectors: List[str],
    per_detector: Dict[str, Dict[str, Dict[str, object]]],
    stats: Dict[str, Dict[str, object]],
    prob_threshold: float,
    min_positive_detectors: int,
) -> Dict[str, Dict[str, object]]:
    content_ids = set()
    for det_results in per_detector.values():
        content_ids.update(det_results.keys())

    aggregated: Dict[str, Dict[str, object]] = {}
    for cid in sorted(content_ids):
        per_det_probs = []
        per_det_max = []
        per_det_pred = []
        per_det_detail = []
        for det in detectors:
            metrics = per_detector.get(det, {}).get(cid)
            if not metrics:
                continue
            prob_mean = float(metrics.get("prob_mean", 0.0))
            prob_max = float(metrics.get("prob_max", 0.0))
            pred_label = int(metrics.get("pred_label", 0))
            per_det_probs.append(prob_mean)
            per_det_max.append(prob_max)
            per_det_pred.append(pred_label)
            per_det_detail.append((det, prob_mean, pred_label))

        stats_entry = {
            "source": "",
            "frame_dir": "",
            "frames": "",
            "issues": [],
        }
        if cid in stats:
            stats_entry.update(stats[cid])

        positive_count = sum(per_det_pred)
        overall_flag = 1 if (per_det_pred and positive_count >= min_positive_detectors) else 0
        if not overall_flag and per_det_probs:
            overall_flag = 1 if mean(per_det_probs) >= prob_threshold else 0

        row = {
            "prob_mean": f"{mean(per_det_probs):.6f}" if per_det_probs else "",
            "prob_max": f"{max(per_det_max):.6f}" if per_det_max else "",
            "prob_std": f"{pstdev(per_det_probs):.6f}" if len(per_det_probs) > 1 else "0.0" if per_det_probs else "",
            "pred_label": overall_flag,
            "num_detectors": len(per_det_probs),
            "positive_detectors": positive_count,
            "detail": "|".join(f"{det}:{prob:.6f}:{pred}" for det, prob, pred in per_det_detail),
            "stats": stats_entry,
        }
        aggregated[cid] = row
    return aggregated


def write_multi_content_scores(
    out_path: Path,
    detectors: List[str],
    per_detector: Dict[str, Dict[str, Dict[str, object]]],
    overall: Dict[str, Dict[str, object]],
    content_map: Dict[str, List[Dict[str, str]]],
) -> None:
    base_fields = [
        "content_id",
        "canonical_local_path",
        "source_path",
        "frame_dir",
        "num_frames",
        "prob_mean",
        "prob_max",
        "prob_std",
        "pred_label",
        "num_detectors",
        "positive_detectors",
        "detector_detail",
        "posting_ids",
        "issues",
    ]
    det_fields = []
    for det in detectors:
        det_fields.extend([
            f"{det}_prob_mean",
            f"{det}_prob_max",
            f"{det}_prob_std",
            f"{det}_pred_label",
        ])
    fieldnames = base_fields + det_fields

    ensure_directory(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid, data in sorted(overall.items()):
            stats_entry = data.get("stats", {})
            postings = sorted({row.get("posting_id", "") for row in content_map.get(cid, []) if row.get("posting_id")})
            canonical_path = ""
            if content_map.get(cid):
                canonical_rows = [r for r in content_map[cid] if r.get("local_path")]
                if canonical_rows:
                    canonical_path = canonical_rows[0].get("local_path", "")
            row = {
                "content_id": cid,
                "canonical_local_path": canonical_path,
                "source_path": stats_entry.get("source", ""),
                "frame_dir": stats_entry.get("frame_dir", ""),
                "num_frames": stats_entry.get("frames", ""),
                "prob_mean": data.get("prob_mean", ""),
                "prob_max": data.get("prob_max", ""),
                "prob_std": data.get("prob_std", ""),
                "pred_label": data.get("pred_label", 0),
                "num_detectors": data.get("num_detectors", 0),
                "positive_detectors": data.get("positive_detectors", 0),
                "detector_detail": data.get("detail", ""),
                "posting_ids": "|".join(postings),
                "issues": ";".join(stats_entry.get("issues", [])),
            }
            for det in detectors:
                metrics = per_detector.get(det, {}).get(cid)
                if metrics:
                    row[f"{det}_prob_mean"] = f"{float(metrics.get('prob_mean', 0.0)):.6f}"
                    row[f"{det}_prob_max"] = f"{float(metrics.get('prob_max', 0.0)):.6f}"
                    row[f"{det}_prob_std"] = f"{float(metrics.get('prob_std', 0.0)):.6f}"
                    row[f"{det}_pred_label"] = metrics.get("pred_label", "")
                else:
                    row[f"{det}_prob_mean"] = ""
                    row[f"{det}_prob_max"] = ""
                    row[f"{det}_prob_std"] = ""
                    row[f"{det}_pred_label"] = ""
            writer.writerow(row)


def compute_multi_post_scores(
    detectors: List[str],
    per_detector: Dict[str, Dict[str, Dict[str, object]]],
    content_map: Dict[str, List[Dict[str, str]]],
) -> Dict[str, Dict[str, object]]:
    post_scores: Dict[str, Dict[str, object]] = {}
    # Build mapping from posting to content IDs (dedup)
    post_to_contents: Dict[str, List[str]] = {}
    for cid, postings in content_map.items():
        for row in postings:
            pid = row.get("posting_id", "")
            if pid:
                post_to_contents.setdefault(pid, []).append(cid)

    for pid, cids in post_to_contents.items():
        per_det_post = {}
        for det in detectors:
            probs = [per_detector.get(det, {}).get(cid, {}).get("prob_mean") for cid in cids]
            probs = [float(p) for p in probs if p is not None]
            if probs:
                per_det_post[det] = {
                    "prob_mean": mean(probs),
                    "prob_max": max(probs),
                }
        overall_probs = [vals["prob_mean"] for vals in per_det_post.values()]
        row = {
            "posting_id": pid,
            "prob_mean": f"{mean(overall_probs):.6f}" if overall_probs else "",
            "prob_max": f"{max(vals['prob_max'] for vals in per_det_post.values()):.6f}" if per_det_post else "",
            "num_contents": len(cids),
            "content_ids": "|".join(sorted(set(cids))),
            "num_detectors": len(per_det_post),
            "detector_detail": "|".join(
                f"{det}:{vals['prob_mean']:.6f}" for det, vals in per_det_post.items()
            ),
        }
        for det in detectors:
            vals = per_det_post.get(det)
            row[f"{det}_prob_mean"] = f"{vals['prob_mean']:.6f}" if vals else ""
            row[f"{det}_prob_max"] = f"{vals['prob_max']:.6f}" if vals else ""
        post_scores[pid] = row
    return post_scores


def write_multi_post_scores(out_path: Path, posts: Dict[str, Dict[str, object]], detectors: List[str]) -> None:
    base_fields = [
        "posting_id",
        "prob_mean",
        "prob_max",
        "num_detectors",
        "num_contents",
        "content_ids",
        "detector_detail",
    ]
    det_fields = []
    for det in detectors:
        det_fields.extend([f"{det}_prob_mean", f"{det}_prob_max"])
    fieldnames = base_fields + det_fields
    ensure_directory(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pid, row in sorted(posts.items()):
            writer.writerow(row)


def read_content_map(csv_path: Path) -> Dict[str, List[Dict[str, str]]]:
    mapping: Dict[str, List[Dict[str, str]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cid = row.get("content_id", "")
            if not cid:
                continue
            mapping.setdefault(cid, []).append(row)
    return mapping


def main():
    ap = build_argparser()
    args = ap.parse_args()

    work_dir = Path(args.work_dir).resolve()
    media_root = Path(args.media_root).resolve() if args.media_root else work_dir
    kept_content_path = Path(args.kept_csv).resolve() if args.kept_csv else work_dir / "kept_content.csv"
    if not kept_content_path.exists():
        raise FileNotFoundError(f"kept_content.csv missing: {kept_content_path}")

    content_map_path = Path(args.content_map).resolve() if args.content_map else work_dir / "content_id_map.csv"
    if not content_map_path.exists():
        raise FileNotFoundError(f"content_id_map.csv missing: {content_map_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else work_dir / "dfbench"
    ensure_directory(out_dir)
    detectors_dir = out_dir / "detectors"
    ensure_directory(detectors_dir)

    dfb_root = Path(args.dfb_root).resolve()
    dataset_root = dfb_root / "datasets" / "rgb" / args.dataset_name
    dataset_json_path = dfb_root / "preprocessing" / "dataset_json" / f"{args.dataset_name}.json"
    ensure_directory(dataset_json_path.parent)

    detector_names_input = [name.strip() for name in args.detectors.split(",") if name.strip()]
    if len(detector_names_input) == 1 and detector_names_input[0].lower() == "all":
        detector_names_input = SAFE_DETECTOR_NAMES.copy()

    config_paths = []
    for name in detector_names_input:
        key = name.lower()
        if key not in DEFAULT_DETECTORS:
            raise ValueError(f"Unknown detector '{name}'. Available: {sorted(DEFAULT_DETECTORS.keys())}")
        config_paths.append(Path(DEFAULT_DETECTORS[key]).resolve())
    for extra in args.detector_config:
        config_paths.append(Path(extra).resolve())

    content_rows = read_content_csv(kept_content_path)
    dataset_json, stats = prepare_dataset(
        content_rows,
        media_root=media_root,
        dataset_root=dataset_root,
        dataset_name=args.dataset_name,
        label_name=args.label_name,
        max_frames=args.max_frames,
        image_size=args.image_size,
    )

    with dataset_json_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=2)

    per_detector_results: Dict[str, Dict[str, Dict[str, object]]] = {}
    content_map = read_content_map(content_map_path)

    ordered_detectors: List[str] = []
    for cfg_path in config_paths:
        cfg = load_yaml(cfg_path)
        det_name = sanitize_name(cfg_path, cfg)
        weights_path = cfg.get("pretrained")
        if weights_path:
            weights_path_path = Path(weights_path)
            weights_path_resolved = weights_path_path if weights_path_path.is_absolute() else (dfb_root / weights_path_path).resolve()
            if not weights_path_resolved.exists():
                print(f"[multi] warning: weights file not found for {det_name}: {weights_path_resolved}")
                print(f"[multi] skipping detector {det_name} due to missing weights.")
                continue
        else:
            print(f"[multi] warning: detector {det_name} does not specify pretrained weights; skipping.")
            continue

        ordered_detectors.append(det_name)
        det_dir = detectors_dir / det_name
        ensure_directory(det_dir)

        predictions_csv = det_dir / "predictions.csv"
        content_scores_csv = det_dir / "content_scores.csv"
        post_scores_csv = det_dir / "post_scores.csv"

        try:
            run_dfbench_test(
                dfb_root=dfb_root,
                detector_config=cfg_path,
                dataset_name=args.dataset_name,
                save_csv=predictions_csv,
                weights_path=weights_path_resolved,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[multi] detector {det_name} failed (exit {exc.returncode}); skipping. {exc}")
            predictions_csv.unlink(missing_ok=True)
            continue

        content_probs = parse_predictions(predictions_csv, dataset_name=args.dataset_name)
        aggregated = aggregate_scores(content_probs, stats, prob_threshold=args.prob_threshold)
        per_detector_results[det_name] = aggregated

        if not args.skip_join:
            write_content_scores(content_scores_csv, aggregated, content_map)
            write_post_scores(post_scores_csv, aggregated, content_map)

    if not ordered_detectors:
        print("[multi] no detectors executed; check weight paths or configuration.")
        return

    overall = compute_overall_content_metrics(
        ordered_detectors,
        per_detector_results,
        stats,
        prob_threshold=args.prob_threshold,
        min_positive_detectors=args.min_positive_detectors,
    )

    multi_content_path = out_dir / "dfbench_multi_content_scores.csv"
    write_multi_content_scores(
        multi_content_path,
        ordered_detectors,
        per_detector_results,
        overall,
        content_map,
    )
    print(f"[multi] aggregated content scores → {multi_content_path}")

    multi_post_scores = compute_multi_post_scores(
        ordered_detectors,
        per_detector_results,
        content_map,
    )
    multi_post_path = out_dir / "dfbench_multi_post_scores.csv"
    write_multi_post_scores(multi_post_path, multi_post_scores, ordered_detectors)
    print(f"[multi] aggregated post scores → {multi_post_path}")

    if args.join_posts_summary:
        join_cmd = [
            "python",
            "-m",
            "p2p.tools.join_dfbench_scores",
            "--posts-summary",
            args.join_posts_summary,
            "--content-map",
            str(content_map_path),
            "--content-scores",
            str(multi_content_path),
        ]
        if args.join_output:
            join_cmd.extend(["--out-path", args.join_output])
        print("[multi] running join:", " ".join(join_cmd))
        subprocess.run(join_cmd, check=True)


if __name__ == "__main__":
    main()
