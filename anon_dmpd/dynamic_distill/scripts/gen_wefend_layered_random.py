#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import yaml

GPU_LIST = [2, 3, 4, 5, 6]


PARAM_RANGES = {
    ("distillation", "event_layer", "metric"): ["min", "acc", "conf", "avg"],
    ("distillation", "event_layer", "min_seen"): [3, 4, 5, 6],
    ("distillation", "event_layer", "high_threshold"): (0.90, 0.95),
    ("distillation", "event_layer", "mid_threshold"): (0.80, 0.90),
    ("distillation", "event_layer", "weight_high"): (1.02, 1.12),
    ("distillation", "event_layer", "weight_mid"): (0.68, 0.82),
    ("distillation", "event_layer", "weight_low"): (0.40, 0.55),
    ("distillation", "event_layer", "default_weight"): (0.5, 0.9),
    ("distillation", "fallback", "enabled"): [True, False],
    ("distillation", "fallback", "min_pairs_fraction"): (0.0, 0.08),
    ("distillation", "fallback", "positive_only"): [True, False],
    ("distillation", "fallback", "teacher_conf"): (0.92, 0.96),
    ("distillation", "fallback", "student_conf"): (0.90, 0.94),
    ("distillation", "fallback", "temperature"): (2.15, 2.35),
    ("distillation", "fallback", "lambda_scale"): (0.1, 0.25),
    ("distillation", "fallback", "start_fraction"): (0.28, 0.42),
    ("distillation", "fallback", "require_student_mistake"): [False],
    ("distillation", "student_focus", "enabled"): [True, False],
    ("distillation", "student_focus", "warmup"): [4, 6, 8],
    ("distillation", "student_focus", "threshold"): (0.88, 0.93),
    ("distillation", "student_focus", "mode"): ["pos_recall"],
    ("distillation", "temperature"): (2.30, 2.36),
    ("distillation", "adaptive_temperature", "base"): (2.30, 2.36),
    ("distillation", "adaptive_temperature", "coeff"): (1.0, 1.3),
    ("distillation", "delta"): (0.058, 0.072),
    ("distillation", "positive_distill_boost"): (1.6, 1.8),
    ("distillation", "positive_stage_boost"): (2.1, 2.4),
    ("distillation", "positive_stage_start_fraction"): (0.42, 0.48),
}


def random_choice(values):
    if isinstance(values, list):
        return random.choice(values)
    low, high = values
    return round(random.uniform(low, high), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random layered config generator for WeFEND.")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_pos_balanced.yaml",
    )
    parser.add_argument("--out-dir", default="dynamic_distill/configs/wefend_layered_random")
    parser.add_argument("--run-script", default="scripts/run_wefend_layered_random.sh")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--random-seed", type=int, default=2025)
    args = parser.parse_args()

    random.seed(args.random_seed)

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    generated = []
    for idx in range(args.count):
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("distillation", {})["event_layer"] = cfg["distillation"].get("event_layer", {})
        cfg["distillation"]["event_layer"]["enabled"] = True
        for keys, values in PARAM_RANGES.items():
            value = random_choice(values)
            target = cfg
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = value
        name = f"wefend_layered_rand_{timestamp}_{idx:02d}"
        cfg.setdefault("notes", {})["strategy_name"] = name
        cfg.setdefault("logging", {})["output_dir"] = "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
        cfg.setdefault("logging", {})["run_name"] = name
        cfg_path = out_dir / f"{name}.yaml"
        yaml.safe_dump(cfg, cfg_path.open("w"), sort_keys=False)
        generated.append(cfg_path)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_layered_random"]
    job_idx = 0
    total_jobs = len(generated) * len(args.seeds)
    for cfg_path in generated:
        stem = cfg_path.stem
        for seed in args.seeds:
            gpu = GPU_LIST[job_idx % len(GPU_LIST)]
            log_path = f"logs/wefend_layered_random/{stem}_seed{seed}.log"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {cfg_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1
            if job_idx % len(GPU_LIST) == 0 and job_idx < total_jobs:
                run_lines.append("wait")
                run_lines.append("echo '--- layered random slice complete ---'")
    run_lines.append("wait")
    run_lines.append("echo 'All layered random jobs finished.'")

    run_path = Path(args.run_script)
    with run_path.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_path.chmod(0o755)

    print(f"Generated {len(generated)} configs in {out_dir}")
    print(f"Run `bash {run_path}` to start jobs on GPUs {GPU_LIST}.")


if __name__ == "__main__":
    main()
