#!/usr/bin/env python
"""Random search generator for weak-teacher WeFEND configs."""
from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import yaml

GPU_LIST = [2, 3, 4, 5, 6]

EVENT_LAYER_PARAMS = {
    ("distillation", "event_layer", "enabled"): [True],
    ("distillation", "event_layer", "metric"): ["min", "acc", "conf", "avg"],
    ("distillation", "event_layer", "min_seen"): [4, 5, 6],
    ("distillation", "event_layer", "high_threshold"): (0.91, 0.94),
    ("distillation", "event_layer", "mid_threshold"): (0.82, 0.89),
    ("distillation", "event_layer", "weight_high"): (1.02, 1.08),
    ("distillation", "event_layer", "weight_mid"): (0.7, 0.8),
    ("distillation", "event_layer", "weight_low"): (0.42, 0.55),
}

WEAK_TEACHER_PARAMS = {
    ("distillation", "weak_teacher", "enabled"): [True],
    ("distillation", "weak_teacher", "lambda"): (0.08, 0.18),
    ("distillation", "weak_teacher", "metric"): ["min", "avg", "conf"],
    ("distillation", "weak_teacher", "min_seen"): [3, 4, 5],
    ("distillation", "weak_teacher", "threshold"): (0.82, 0.88),
    ("distillation", "weak_teacher", "apply_positive"): [True],
    ("distillation", "weak_teacher", "apply_negative"): [False, True],
    ("distillation", "weak_teacher", "temperature"): (2.18, 2.30),
}

OTHER_PARAMS = {
    ("distillation", "student_focus", "enabled"): [True, False],
    ("distillation", "student_focus", "warmup"): [4, 6],
    ("distillation", "student_focus", "threshold"): (0.88, 0.92),
    ("distillation", "student_focus", "mode"): ["pos_recall"],
    ("distillation", "temperature"): (2.30, 2.36),
    ("distillation", "adaptive_temperature", "base"): (2.30, 2.36),
    ("distillation", "adaptive_temperature", "coeff"): (1.0, 1.2),
    ("distillation", "delta"): (0.058, 0.07),
    ("distillation", "positive_distill_boost"): (1.6, 1.8),
    ("distillation", "positive_stage_boost"): (2.2, 2.45),
    ("distillation", "positive_stage_start_fraction"): (0.42, 0.48),
}


def random_value(value_range):
    if isinstance(value_range, list):
        return random.choice(value_range)
    low, high = value_range
    return round(random.uniform(low, high), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random weak-teacher config generator.")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_pos_balanced.yaml",
    )
    parser.add_argument("--out-dir", default="dynamic_distill/configs/wefend_weakteacher_random")
    parser.add_argument("--run-script", default="scripts/run_wefend_weakteacher_random.sh")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--random-seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.random_seed)

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    generated_paths = []

    for idx in range(args.count):
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("distillation", {})["event_layer"] = cfg["distillation"].get("event_layer", {})
        cfg.setdefault("distillation", {})["weak_teacher"] = cfg["distillation"].get("weak_teacher", {})
        cfg["distillation"]["event_layer"]["enabled"] = True
        cfg["distillation"]["weak_teacher"]["enabled"] = True

        for param_dict in (EVENT_LAYER_PARAMS, WEAK_TEACHER_PARAMS, OTHER_PARAMS):
            for keys, value_range in param_dict.items():
                value = random_value(value_range)
                target = cfg
                for key in keys[:-1]:
                    target = target.setdefault(key, {})
                target[keys[-1]] = value

        name = f"wefend_weakteacher_rand_{timestamp}_{idx:02d}"
        cfg.setdefault("notes", {})["strategy_name"] = name
        cfg.setdefault("logging", {})["output_dir"] = "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
        cfg.setdefault("logging", {})["run_name"] = name
        cfg_path = out_dir / f"{name}.yaml"
        yaml.safe_dump(cfg, cfg_path.open("w"), sort_keys=False)
        generated_paths.append(cfg_path)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_weakteacher_random"]
    job_idx = 0
    total_jobs = len(generated_paths) * len(args.seeds)
    for cfg_path in generated_paths:
        stem = cfg_path.stem
        for seed in args.seeds:
            gpu = GPU_LIST[job_idx % len(GPU_LIST)]
            log_path = f"logs/wefend_weakteacher_random/{stem}_seed{seed}.log"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {cfg_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1
            if job_idx % len(GPU_LIST) == 0 and job_idx < total_jobs:
                run_lines.append("wait")
                run_lines.append("echo '--- weakteacher random slice complete ---'")
    run_lines.append("wait")
    run_lines.append("echo 'All weakteacher random jobs finished.'")

    run_path = Path(args.run_script)
    with run_path.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_path.chmod(0o755)

    print(f"Generated {len(generated_paths)} configs in {out_dir}")
    print(f"Run `bash {run_path}` to launch jobs on GPUs {GPU_LIST}.")


if __name__ == "__main__":
    main()
