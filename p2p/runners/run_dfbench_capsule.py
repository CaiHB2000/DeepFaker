# -*- coding: utf-8 -*-
"""
Full pipeline to run DeepFakeBench CapsuleNet inference on scraped Reddit media.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import yaml

from p2p.tools.prepare_dfbench_dataset import (
    prepare_dataset,
    read_content_csv,
    ensure_dir,
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run DeepFakeBench CapsuleNet on Reddit media.")
    ap.add_argument("--work-dir", type=str, default="tmp/reddit_seed_run", help="Base directory for intermediate outputs.")
    ap.add_argument("--kept-csv", type=str, default=None, help="Path to kept_content.csv (defaults to <work-dir>/kept_content.csv).")
    ap.add_argument("--media-root", type=str, default=None, help="Media root containing local_path files (defaults to work-dir).")
    ap.add_argument("--content-map", type=str, default=None, help="Path to content_id_map.csv (defaults to <work-dir>/content_id_map.csv).")
    ap.add_argument("--dfb-root", type=str, default="DeepFakeBench", help="Root directory of DeepFakeBench.")
    ap.add_argument("--dataset-name", type=str, default="reddit_canonical", help="Dataset name used for DeepFakeBench.")
    ap.add_argument("--label-name", type=str, default="reddit_unknown", help="Label name registered in test_config.yaml.")
    ap.add_argument("--detector-config", type=str, default="DeepFakeBench/training/config/detector/capsule_net_reddit.yaml", help="Detector YAML to use.")
    ap.add_argument("--weights-path", type=str, default=None, help="Optional weights override for detector.")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory to write inference outputs (defaults to <work-dir>/dfbench).")
    ap.add_argument("--max-frames", type=int, default=32, help="Max frames sampled per video.")
    ap.add_argument("--image-size", type=int, default=256, help="Frame size for stored crops.")
    ap.add_argument("--prob-threshold", type=float, default=0.5, help="Threshold to convert probability into binary label.")
    ap.add_argument("--skip-inference", action="store_true", help="Only rebuild dataset/JSON without running detector.")
    return ap


def read_csv_dict(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_content_map(path: Path) -> Dict[str, List[Dict[str, str]]]:
    mapping: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("content_id")
            if cid:
                mapping[cid].append(row)
    return mapping


def run_dfbench_test(
    dfb_root: Path,
    detector_config: Path,
    dataset_name: str,
    save_csv: Path,
    weights_path: Path | None = None,
) -> None:
    cmd = [
        "python",
        "training/test.py",
        "--detector_path",
        str(detector_config),
        "--test_dataset",
        dataset_name,
        "--save_csv",
        str(save_csv),
    ]
    if weights_path:
        cmd.extend(["--weights_path", str(weights_path)])

    print(f"[run] {' '.join(cmd)}")
    env = os.environ.copy()
    extra_paths = [
        str((dfb_root / "training").resolve()),
        str(dfb_root.resolve()),
    ]
    current_py_path = env.get("PYTHONPATH", "")
    combined = os.pathsep.join(extra_paths + ([current_py_path] if current_py_path else []))
    env["PYTHONPATH"] = combined
    env.setdefault("TQDM_DISABLE", "1")
    subprocess.run(cmd, check=True, env=env, cwd=str(dfb_root))


def parse_predictions(csv_path: Path, dataset_name: str) -> Dict[str, Dict[str, List[float]]]:
    content_probs: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"prob": [], "label": []})
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row.get("dataset")
            if ds != dataset_name:
                continue
            image_path = Path(row.get("image", ""))
            if "frames" in image_path.parts:
                idx = image_path.parts.index("frames")
                if idx + 1 < len(image_path.parts):
                    content_id = image_path.parts[idx + 1]
                else:
                    continue
            else:
                content_id = image_path.stem
            prob = float(row.get("prob", 0.0))
            label = int(float(row.get("label", 0)))
            entry = content_probs[content_id]
            entry["prob"].append(prob)
            entry["label"].append(label)
    return content_probs


def aggregate_scores(
    content_probs: Dict[str, Dict[str, List[float]]],
    stats: Dict[str, Dict[str, object]],
    prob_threshold: float,
) -> Dict[str, Dict[str, object]]:
    aggregated: Dict[str, Dict[str, object]] = {}
    for content_id, values in content_probs.items():
        probs = values.get("prob", [])
        labels = values.get("label", [])
        if not probs:
            continue
        mean_prob = float(mean(probs))
        max_prob = float(max(probs))
        std_prob = float(pstdev(probs)) if len(probs) > 1 else 0.0
        pred_label = 1 if mean_prob >= prob_threshold else 0
        aggregated[content_id] = {
            "prob_mean": mean_prob,
            "prob_max": max_prob,
            "prob_std": std_prob,
            "pred_label": pred_label,
            "num_samples": len(probs),
            "raw_labels": labels,
            "stats": stats.get(content_id, {}),
        }
    return aggregated


def write_content_scores(
    out_path: Path,
    aggregated: Dict[str, Dict[str, object]],
    content_map: Dict[str, List[Dict[str, str]]],
) -> None:
    fieldnames = [
        "content_id",
        "canonical_local_path",
        "source_path",
        "frame_dir",
        "num_frames",
        "prob_mean",
        "prob_max",
        "prob_std",
        "pred_label",
        "num_samples",
        "posting_ids",
        "issues",
    ]
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for content_id, data in sorted(aggregated.items()):
            stats = data.get("stats", {})
            postings = sorted({row.get("posting_id", "") for row in content_map.get(content_id, []) if row.get("posting_id")})
            canonical_path = None
            if content_map.get(content_id):
                canonical_rows = [r for r in content_map[content_id] if r.get("local_path")]
                if canonical_rows:
                    canonical_path = canonical_rows[0].get("local_path")
            writer.writerow({
                "content_id": content_id,
                "canonical_local_path": canonical_path or "",
                "source_path": stats.get("source", ""),
                "frame_dir": stats.get("frame_dir", ""),
                "num_frames": stats.get("frames", ""),
                "prob_mean": f"{data['prob_mean']:.6f}",
                "prob_max": f"{data['prob_max']:.6f}",
                "prob_std": f"{data['prob_std']:.6f}",
                "pred_label": data["pred_label"],
                "num_samples": data["num_samples"],
                "posting_ids": "|".join(postings),
                "issues": ";".join(stats.get("issues", [])),
            })


def write_post_scores(
    out_path: Path,
    aggregated: Dict[str, Dict[str, object]],
    content_map: Dict[str, List[Dict[str, str]]],
) -> None:
    fieldnames = [
        "posting_id",
        "prob_mean",
        "prob_max",
        "num_contents",
        "content_ids",
    ]
    ensure_dir(out_path.parent)
    post_scores: Dict[str, Dict[str, object]] = defaultdict(lambda: {"probs": [], "content_ids": []})
    for content_id, data in aggregated.items():
        postings = [row.get("posting_id") for row in content_map.get(content_id, []) if row.get("posting_id")]
        for pid in postings:
            ps = post_scores[pid]
            ps["probs"].append(data["prob_mean"])
            ps["content_ids"].append(content_id)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for posting_id, data in sorted(post_scores.items()):
            probs = data["probs"]
            writer.writerow({
                "posting_id": posting_id,
                "prob_mean": f"{mean(probs):.6f}" if probs else "",
                "prob_max": f"{max(probs):.6f}" if probs else "",
                "num_contents": len(data["content_ids"]),
                "content_ids": "|".join(sorted(set(data["content_ids"]))),
            })


def main():
    ap = build_argparser()
    args = ap.parse_args()

    work_dir = Path(args.work_dir).resolve()
    kept_csv_path = Path(args.kept_csv).resolve() if args.kept_csv else work_dir / "kept_content.csv"
    if not kept_csv_path.exists():
        raise FileNotFoundError(f"kept content CSV not found: {kept_csv_path}")

    media_root = Path(args.media_root).resolve() if args.media_root else work_dir
    content_map_path = Path(args.content_map).resolve() if args.content_map else work_dir / "content_id_map.csv"
    if not content_map_path.exists():
        raise FileNotFoundError(f"content_id_map.csv not found: {content_map_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else work_dir / "dfbench"
    ensure_dir(out_dir)

    dfb_root_path = Path(args.dfb_root).resolve()
    dataset_root = dfb_root_path / "datasets" / "rgb" / args.dataset_name
    dataset_json_path = dfb_root_path / "preprocessing" / "dataset_json" / f"{args.dataset_name}.json"

    detector_config_path = Path(args.detector_config).resolve()
    with detector_config_path.open("r", encoding="utf-8") as f:
        detector_cfg = yaml.safe_load(f) or {}

    content_rows = read_content_csv(kept_csv_path)
    dataset_json, stats = prepare_dataset(
        content_rows,
        media_root=media_root,
        dataset_root=dataset_root,
        dataset_name=args.dataset_name,
        label_name=args.label_name,
        max_frames=args.max_frames,
        image_size=args.image_size,
    )

    ensure_dir(dataset_json_path.parent)
    with dataset_json_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=2)
    print(f"[dfbench] dataset json written → {dataset_json_path}")

    if args.skip_inference:
        print("[dfbench] --skip-inference set; exiting after dataset preparation.")
        return

    predictions_csv = out_dir / "dfbench_predictions.csv"
    if args.weights_path:
        weights_path = Path(args.weights_path).resolve()
    else:
        pretrained = detector_cfg.get("pretrained")
        weights_path = (dfb_root_path / pretrained) if pretrained else None
        if weights_path and not weights_path.exists():
            print(f"[warn] pretrained weights not found at {weights_path}, using detector defaults")
            weights_path = None

    run_dfbench_test(
        dfb_root=dfb_root_path,
        detector_config=detector_config_path,
        dataset_name=args.dataset_name,
        save_csv=predictions_csv,
        weights_path=weights_path,
    )

    content_probs = parse_predictions(predictions_csv, dataset_name=args.dataset_name)
    aggregated = aggregate_scores(content_probs, stats, prob_threshold=args.prob_threshold)
    content_map = read_content_map(content_map_path)

    content_scores_path = out_dir / "dfbench_content_scores.csv"
    write_content_scores(content_scores_path, aggregated, content_map)
    print(f"[dfbench] content scores → {content_scores_path}")

    post_scores_path = out_dir / "dfbench_post_scores.csv"
    write_post_scores(post_scores_path, aggregated, content_map)
    print(f"[dfbench] post scores → {post_scores_path}")


if __name__ == "__main__":
    main()
