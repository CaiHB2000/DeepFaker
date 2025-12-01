#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import os
import random
import time
from pathlib import Path

import yaml

# Parameter grids focusing on positive-class improvements and gating variations.
PARAM_GRID = {
    ("distillation", "temperature"): [2.30, 2.35, 2.40, 2.45],
    ("distillation", "lambda_kl"): [1.05, 1.10, 1.15, 1.20],
    ("distillation", "delta"): [0.060, 0.065, 0.070],
    ("distillation", "confidence_gate", "margin"): [0.12, 0.14, 0.16],
    ("distillation", "positive_distill_boost"): [1.7, 1.9, 2.1],
    ("distillation", "positive_stage_start_fraction"): [0.38, 0.42, 0.46],
    ("distillation", "positive_stage_boost"): [2.2, 2.4, 2.6],
    ("distillation", "positive_student_conf_margin"): [0.92, 0.94, 0.96],
    ("distillation", "agreement_confidence_gap"): [0.03, 0.04, 0.05],
    ("distillation", "fusion_confidence"): [0.18, 0.20, 0.22],
    ("distillation", "start_fraction"): [0.16, 0.18, 0.20],
    ("distillation", "end_fraction"): [0.60, 0.64, 0.68],
    ("distillation", "delta_schedule", "end_value"): [0.035, 0.040, 0.045],
    ("distillation", "uncertainty_weight", "scale"): [1.0, 1.2, 1.4],
    ("distillation", "uncertainty_weight", "power"): [1.0, 1.1, 1.2],
}

BOOL_CHOICES = {
    ("distillation", "require_student_mistake"): [True, False],
    ("distillation", "positive_event_gate", "only"): [True, False],
}

POS_EVENT_CHOICES = {
    ("distillation", "positive_event_gate", "teacher_conf"): [0.94, 0.95, 0.96],
    ("distillation", "positive_event_gate", "student_conf"): [0.84, 0.86, 0.88],
}

EVENT_FILTER_CHOICES = {
    ("distillation", "event_filter", "teacher_min_acc"): [0.86, 0.88, 0.90],
    ("distillation", "event_filter", "teacher_min_conf"): [0.90, 0.91, 0.92],
    ("distillation", "event_filter", "warmup_steps"): [3, 4, 5],
}

CLASS_WEIGHT_CHOICES = [2.1, 2.3, 2.5]

TRAINING_CHOICES = {
    ("training", "max_steps_per_epoch"): [220, 230, 240],
}


def set_nested(config: dict, keys: tuple[str, ...], value):
    target = config
    for key in keys[:-1]:
        if key not in target or target[key] is None:
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def main():
    parser = argparse.ArgumentParser(description="Generate batched Wefend configs.")
    parser.add_argument("--base", default="dynamic_distill/configs/wefend_dynamic_distill_teacher_wcls_event_reliable.yaml",
                        help="Base YAML config to copy from.")
    parser.add_argument("--out-dir", default="dynamic_distill/configs/auto_wefend",
                        help="Directory to store generated configs.")
    parser.add_argument("--count", type=int, default=12, help="Number of random configs to generate.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0], help="Seeds to evaluate per config.")
    parser.add_argument("--run-script", default="scripts/run_wefend_auto.sh",
                        help="Output script with run commands.")
    parser.add_argument("--random-seed", type=int, default=2025, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.random_seed)

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    generated = []

    for idx in range(args.count):
        cfg = copy.deepcopy(base_cfg)

        # Apply random draws from grids
        for keys, choices in PARAM_GRID.items():
            set_nested(cfg, keys, random.choice(choices))
        for keys, choices in BOOL_CHOICES.items():
            set_nested(cfg, keys, random.choice(choices))
        for keys, choices in POS_EVENT_CHOICES.items():
            set_nested(cfg, keys, random.choice(choices))
        for keys, choices in EVENT_FILTER_CHOICES.items():
            set_nested(cfg, keys, random.choice(choices))
        for keys, choices in TRAINING_CHOICES.items():
            set_nested(cfg, keys, random.choice(choices))

        # Adjust class weights (positive class index 1)
        class_weights = cfg.get("loss", {}).get("class_weights", [1.0, 1.0])
        if len(class_weights) < 2:
            class_weights = [1.0, 1.0]
        class_weights[1] = random.choice(CLASS_WEIGHT_CHOICES)
        cfg.setdefault("loss", {})["class_weights"] = class_weights

        cfg.setdefault("notes", {})["auto_batch"] = {
            "timestamp": timestamp,
            "index": idx,
            "random_seed": args.random_seed,
        }

        cfg_name = f"wefend_auto_{timestamp}_{idx:02d}.yaml"
        cfg_path = out_dir / cfg_name
        yaml.safe_dump(cfg, cfg_path.open("w"), sort_keys=False)
        generated.append(cfg_path)

    # Compose run script
    run_path = Path(args.run_script)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    gpus = [2, 3, 4, 5, 6]
    with run_path.open("w") as fh:
        fh.write("#!/usr/bin/env bash\n")
        fh.write("set -e\n")
        fh.write("mkdir -p logs/wefend_batch\n")
        job_idx = 0
        for cfg_path in generated:
            for seed in args.seeds:
                gpu = gpus[job_idx % len(gpus)]
                log_name = cfg_path.stem + f"_seed{seed}.log"
                command = (
                    f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                    f"--config {cfg_path} --seed {seed} --progress > logs/wefend_batch/{log_name} 2>&1 &"
                )
                fh.write(command + "\n")
                job_idx += 1
        fh.write("wait\n")
        fh.write("echo 'All jobs finished.'\n")

    os.chmod(run_path, 0o755)

    print(f"Generated {len(generated)} configs under {out_dir}")
    print(f"Run `bash {run_path}` to launch batch experiments (will occupy GPUs 2-6).")


if __name__ == "__main__":
    main()
