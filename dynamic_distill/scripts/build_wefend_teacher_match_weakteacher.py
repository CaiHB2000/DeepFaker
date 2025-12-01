#!/usr/bin/env python
"""Generate weak-teacher configs for WeFEND."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

Strategy = Tuple[str, Dict[Tuple[str, ...], Any]]

STRATEGIES: List[Strategy] = [
    (
        "weak_teacher_standard",
        {
            ("distillation", "event_layer", "enabled"): True,
            ("distillation", "event_layer", "metric"): "min",
            ("distillation", "event_layer", "min_seen"): 5,
            ("distillation", "event_layer", "high_threshold"): 0.93,
            ("distillation", "event_layer", "mid_threshold"): 0.87,
            ("distillation", "event_layer", "weight_high"): 1.05,
            ("distillation", "event_layer", "weight_mid"): 0.74,
            ("distillation", "event_layer", "weight_low"): 0.45,
            ("distillation", "weak_teacher", "enabled"): True,
            ("distillation", "weak_teacher", "lambda"): 0.12,
            ("distillation", "weak_teacher", "metric"): "min",
            ("distillation", "weak_teacher", "min_seen"): 4,
            ("distillation", "weak_teacher", "threshold"): 0.85,
            ("distillation", "weak_teacher", "apply_positive"): True,
            ("distillation", "weak_teacher", "apply_negative"): False,
            ("distillation", "weak_teacher", "temperature"): 2.25,
        },
    ),
    (
        "weak_teacher_dual",
        {
            ("distillation", "event_layer", "enabled"): True,
            ("distillation", "event_layer", "metric"): "avg",
            ("distillation", "event_layer", "min_seen"): 6,
            ("distillation", "event_layer", "high_threshold"): 0.92,
            ("distillation", "event_layer", "mid_threshold"): 0.86,
            ("distillation", "event_layer", "weight_high"): 1.06,
            ("distillation", "event_layer", "weight_mid"): 0.76,
            ("distillation", "event_layer", "weight_low"): 0.48,
            ("distillation", "weak_teacher", "enabled"): True,
            ("distillation", "weak_teacher", "lambda"): 0.15,
            ("distillation", "weak_teacher", "metric"): "avg",
            ("distillation", "weak_teacher", "min_seen"): 5,
            ("distillation", "weak_teacher", "threshold"): 0.88,
            ("distillation", "weak_teacher", "apply_positive"): True,
            ("distillation", "weak_teacher", "apply_negative"): True,
            ("distillation", "weak_teacher", "temperature"): 2.20,
        },
    ),
    (
        "weak_teacher_focus",
        {
            ("distillation", "event_layer", "enabled"): True,
            ("distillation", "event_layer", "metric"): "min",
            ("distillation", "event_layer", "min_seen"): 4,
            ("distillation", "event_layer", "high_threshold"): 0.91,
            ("distillation", "event_layer", "mid_threshold"): 0.84,
            ("distillation", "event_layer", "weight_high"): 1.02,
            ("distillation", "event_layer", "weight_mid"): 0.72,
            ("distillation", "event_layer", "weight_low"): 0.5,
            ("distillation", "weak_teacher", "enabled"): True,
            ("distillation", "weak_teacher", "lambda"): 0.1,
            ("distillation", "weak_teacher", "metric"): "min",
            ("distillation", "weak_teacher", "min_seen"): 4,
            ("distillation", "weak_teacher", "threshold"): 0.83,
            ("distillation", "weak_teacher", "apply_positive"): True,
            ("distillation", "weak_teacher", "apply_negative"): False,
            ("distillation", "weak_teacher", "temperature"): 2.18,
            ("distillation", "student_focus", "enabled"): True,
            ("distillation", "student_focus", "warmup"): 6,
            ("distillation", "student_focus", "threshold"): 0.9,
            ("distillation", "student_focus", "mode"): "pos_recall",
        },
    ),
    (
        "weak_teacher_fallback",
        {
            ("distillation", "event_layer", "enabled"): True,
            ("distillation", "event_layer", "metric"): "conf",
            ("distillation", "event_layer", "min_seen"): 5,
            ("distillation", "event_layer", "high_threshold"): 0.93,
            ("distillation", "event_layer", "mid_threshold"): 0.88,
            ("distillation", "event_layer", "weight_high"): 1.04,
            ("distillation", "event_layer", "weight_mid"): 0.75,
            ("distillation", "event_layer", "weight_low"): 0.5,
            ("distillation", "weak_teacher", "enabled"): True,
            ("distillation", "weak_teacher", "lambda"): 0.14,
            ("distillation", "weak_teacher", "metric"): "conf",
            ("distillation", "weak_teacher", "min_seen"): 5,
            ("distillation", "weak_teacher", "threshold"): 0.87,
            ("distillation", "weak_teacher", "apply_positive"): True,
            ("distillation", "weak_teacher", "apply_negative"): False,
            ("distillation", "weak_teacher", "temperature"): 2.23,
            ("distillation", "fallback", "enabled"): True,
            ("distillation", "fallback", "min_pairs_fraction"): 0.04,
            ("distillation", "fallback", "positive_only"): True,
            ("distillation", "fallback", "teacher_conf"): 0.94,
            ("distillation", "fallback", "student_conf"): 0.92,
            ("distillation", "fallback", "temperature"): 2.20,
            ("distillation", "fallback", "lambda_scale"): 0.15,
            ("distillation", "fallback", "require_student_mistake"): False,
            ("distillation", "fallback", "start_fraction"): 0.34,
        },
    ),
]

GPU_LIST = [2, 3, 4, 5, 6]


def apply_strategy(base_cfg: dict, strategy: Strategy) -> dict:
    name, modifications = strategy
    cfg = copy.deepcopy(base_cfg)
    for keys, value in modifications.items():
        target = cfg
        for key in keys[:-1]:
            if key not in target or target[key] is None:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
    cfg.setdefault("notes", {})["strategy_name"] = name
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Create weak-teacher configs for WeFEND.")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_pos_balanced.yaml",
    )
    parser.add_argument("--out-dir", default="dynamic_distill/configs/wefend_teacher_match_weakteacher")
    parser.add_argument("--run-script", default="scripts/run_wefend_teacher_match_weakteacher.sh")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_teacher_match_weakteacher"]

    job_idx = 0
    total_jobs = len(STRATEGIES) * len(args.seeds)
    generated_paths: List[Path] = []
    for name, modifications in STRATEGIES:
        cfg = apply_strategy(base_cfg, (name, modifications))
        cfg.setdefault("logging", {})["output_dir"] = "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
        cfg.setdefault("logging", {})["run_name"] = f"wefend_{name}"
        cfg_path = out_dir / f"wefend_{name}.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        generated_paths.append(cfg_path)

    for cfg_path in generated_paths:
        stem = cfg_path.stem
        for seed in args.seeds:
            gpu = GPU_LIST[job_idx % len(GPU_LIST)]
            log_path = f"logs/wefend_teacher_match_weakteacher/{stem}_seed{seed}.log"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {cfg_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1
            if job_idx % len(GPU_LIST) == 0 and job_idx < total_jobs:
                run_lines.append("wait")
                run_lines.append("echo '--- weak-teacher batch slice complete ---'")
    run_lines.append("wait")
    run_lines.append("echo 'All weak-teacher jobs finished.'")

    run_script = Path(args.run_script)
    with run_script.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_script.chmod(0o755)

    print(f"Generated {len(generated_paths)} configs to {out_dir}")
    print(f"Run `bash {run_script}` to launch jobs on GPUs {GPU_LIST}.")


if __name__ == "__main__":
    main()
