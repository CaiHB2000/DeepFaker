#!/usr/bin/env python
"""Builder for event-dynamic-lambda WeFEND configs."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

Strategy = Tuple[str, Dict[Tuple[str, ...], Any]]

STRATEGIES: List[Strategy] = [
    (
        "dynlambda_balanced",
        {
            ("loss", "class_weights", "index1"): 2.85,
            ("distillation", "weak_teacher", "lambda"): 0.2,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.78,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.92,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.2,
            ("distillation", "event_layer", "weight_mid"): 0.82,
            ("distillation", "event_layer", "weight_low"): 0.5,
        },
    ),
    (
        "dynlambda_balanced_stage",
        {
            ("training", "epochs"): 18,
            ("distillation", "start_fraction"): 0.2,
            ("distillation", "end_fraction"): 0.72,
            ("distillation", "weak_teacher", "lambda"): 0.21,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.76,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.9,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.4,
            ("distillation", "positive_stage_start_fraction"): 0.38,
            ("distillation", "positive_stage_boost"): 2.4,
        },
    ),
    (
        "dynlambda_dualgate",
        {
            ("distillation", "start_fraction"): 0.18,
            ("distillation", "end_fraction"): 0.68,
            ("distillation", "delta_schedule", "start_fraction"): 0.18,
            ("distillation", "delta_schedule", "end_fraction"): 0.68,
            ("distillation", "delta_schedule", "end_value"): 0.046,
            ("distillation", "weak_teacher", "lambda"): 0.2,
            ("distillation", "weak_teacher", "threshold"): 0.84,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.74,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.88,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.1,
        },
    ),
    (
        "dynlambda_posfocus",
        {
            ("distillation", "positive_event_gate", "only"): True,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.965,
            ("distillation", "positive_event_gate", "student_conf"): 0.82,
            ("distillation", "positive_distill_boost"): 2.1,
            ("distillation", "weak_teacher", "lambda"): 0.19,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.75,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.9,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.3,
        },
    ),
    (
        "dynlambda_budget",
        {
            ("distillation", "soft_quota", "enabled"): True,
            ("distillation", "soft_quota", "min_frac"): 0.04,
            ("distillation", "soft_quota", "max_frac"): 0.1,
            ("distillation", "soft_quota", "per_event_cap_frac"): 0.18,
            ("distillation", "soft_quota", "min_reliability"): 0.55,
            ("distillation", "weak_teacher", "lambda"): 0.19,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.72,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.88,
        },
    ),
    (
        "dynlambda_ema_mix",
        {
            ("distillation", "weak_teacher", "mix_ratio"): 0.4,
            ("distillation", "weak_teacher", "event_mix"): 0.78,
            ("distillation", "weak_teacher", "event_ema_decay"): 0.75,
            ("distillation", "weak_teacher", "lambda"): 0.2,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.74,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.9,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.25,
        },
    ),
]

GPU_LIST = [2, 3, 4, 5, 6]


def set_nested(cfg: dict, keys: Tuple[str, ...], value: Any) -> None:
    target = cfg
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def apply_strategy(cfg: dict, strategy: Strategy) -> dict:
    name, modifications = strategy
    new_cfg = copy.deepcopy(cfg)
    for keys, value in modifications.items():
        if keys == ("loss", "class_weights", "index1"):
            weights = new_cfg.setdefault("loss", {}).get("class_weights", [1.0, 1.0])
            if len(weights) < 2:
                weights = [1.0, 1.0]
            weights[1] = value
            new_cfg["loss"]["class_weights"] = weights
        else:
            set_nested(new_cfg, keys, value)
    new_cfg.setdefault("notes", {})["strategy_name"] = name
    new_cfg.setdefault("logging", {})["output_dir"] = (
        "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
    )
    new_cfg.setdefault("logging", {})["run_name"] = name
    return new_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dynamic-lambda configs.")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_weakteacher_eventema_stage.yaml",
    )
    parser.add_argument("--out-dir", default="dynamic_distill/configs/wefend_event_dynamic_lambda")
    parser.add_argument("--run-script", default="scripts/run_wefend_event_dynamic_lambda.sh")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_event_dynamic_lambda"]
    job_idx = 0
    total_jobs = len(STRATEGIES) * len(args.seeds)
    generated_paths: List[Path] = []
    for name, mods in STRATEGIES:
        cfg = apply_strategy(base_cfg, (name, mods))
        cfg_path = out_dir / f"{name}.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        generated_paths.append(cfg_path)

        for seed in args.seeds:
            gpu = GPU_LIST[job_idx % len(GPU_LIST)]
            log_path = f"logs/wefend_event_dynamic_lambda/{cfg_path.stem}_seed{seed:02d}.log"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {cfg_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1
            if job_idx % len(GPU_LIST) == 0 and job_idx < total_jobs:
                run_lines.append("wait")
                run_lines.append("echo '--- dynamic lambda slice complete ---'")

    run_lines.append("wait")
    run_lines.append("echo 'All dynamic-lambda jobs finished.'")

    run_script = Path(args.run_script)
    with run_script.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_script.chmod(0o755)

    print(f"Generated {len(generated_paths)} configs to {out_dir}")
    print(f"Run `bash {run_script}` to launch jobs on GPUs {GPU_LIST}.")


if __name__ == "__main__":
    main()
