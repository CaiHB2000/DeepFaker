#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

Strategy = Tuple[str, Dict[Tuple[str, ...], Any]]

STRATEGIES: List[Strategy] = [
    (
        "pos_balanced_eventreweight",
        {
            ("distillation", "event_filter", "teacher_min_acc"): 0.88,
            ("distillation", "event_filter", "teacher_min_conf"): 0.915,
            ("loss", "event_reweight", "enabled"): True,
            ("loss", "event_reweight", "warmup_steps"): 6,
            ("loss", "event_reweight", "min_size"): 2,
            ("loss", "event_reweight", "scale"): 1.4,
            ("loss", "event_reweight", "power"): 1.4,
            ("loss", "event_reweight", "focus_positive"): True,
            ("loss", "event_reweight", "clip"): 2.4,
        },
    ),
    (
        "pos_balanced_delta_tight",
        {
            ("distillation", "delta"): 0.060,
            ("distillation", "start_fraction"): 0.20,
            ("distillation", "end_fraction"): 0.68,
            ("distillation", "delta_schedule", "start_fraction"): 0.18,
            ("distillation", "delta_schedule", "end_fraction"): 0.72,
            ("distillation", "delta_schedule", "end_value"): 0.032,
            ("distillation", "positive_stage_start_fraction"): 0.45,
            ("distillation", "positive_stage_boost"): 2.20,
        },
    ),
    (
        "pos_balanced_temp_mix",
        {
            ("distillation", "temperature"): 2.32,
            ("distillation", "adaptive_temperature", "base"): 2.32,
            ("distillation", "adaptive_temperature", "coeff"): 1.10,
            ("distillation", "lambda_kl"): 1.02,
            ("distillation", "delta"): 0.068,
        },
    ),
    (
        "pos_balanced_gate_relaxed",
        {
            ("distillation", "event_filter", "teacher_min_acc"): 0.89,
            ("distillation", "event_filter", "teacher_min_conf"): 0.915,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.945,
            ("distillation", "positive_event_gate", "student_conf"): 0.87,
            ("distillation", "positive_student_conf_margin"): 0.88,
        },
    ),
    (
        "pos_balanced_student_focus",
        {
            ("distillation", "student_focus", "enabled"): True,
            ("distillation", "student_focus", "warmup"): 6,
            ("distillation", "student_focus", "threshold"): 0.90,
            ("distillation", "student_focus", "mode"): "pos_recall",
            ("distillation", "event_filter", "teacher_min_acc"): 0.89,
        },
    ),
    (
        "pos_balanced_soft_fallback",
        {
            ("distillation", "fallback", "enabled"): True,
            ("distillation", "fallback", "min_pairs_fraction"): 0.08,
            ("distillation", "fallback", "positive_only"): True,
            ("distillation", "fallback", "teacher_conf"): 0.94,
            ("distillation", "fallback", "student_conf"): 0.93,
            ("distillation", "fallback", "temperature"): 2.2,
            ("distillation", "fallback", "lambda_scale"): 0.25,
            ("distillation", "fallback", "require_student_mistake"): False,
        },
    ),
    (
        "pos_balanced_eventreweight_temp",
        {
            ("distillation", "event_filter", "teacher_min_acc"): 0.88,
            ("distillation", "event_filter", "teacher_min_conf"): 0.915,
            ("loss", "event_reweight", "enabled"): True,
            ("loss", "event_reweight", "warmup_steps"): 8,
            ("loss", "event_reweight", "min_size"): 2,
            ("loss", "event_reweight", "scale"): 1.3,
            ("loss", "event_reweight", "power"): 1.3,
            ("loss", "event_reweight", "focus_positive"): True,
            ("distillation", "temperature"): 2.33,
            ("distillation", "adaptive_temperature", "base"): 2.33,
            ("distillation", "adaptive_temperature", "coeff"): 1.08,
            ("distillation", "lambda_kl"): 1.04,
        },
    ),
]

GPU_ROUND_ROBIN = [0, 3, 4, 5, 6]


def set_nested(cfg: dict, keys: Tuple[str, ...], value: Any) -> None:
    target = cfg
    for key in keys[:-1]:
        if key not in target or target[key] is None:
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def apply_strategy(cfg: dict, strategy: Strategy) -> dict:
    name, modifications = strategy
    new_cfg = copy.deepcopy(cfg)
    for keys, value in modifications.items():
        if keys == ("loss", "event_reweight", "enabled") and not value:
            new_cfg.setdefault("loss", {})["event_reweight"] = {"enabled": False}
        elif keys == ("loss", "event_reweight", "enabled"):
            # ensure nested dict before setting other event_reweight keys
            new_cfg.setdefault("loss", {}).setdefault("event_reweight", {})
            new_cfg["loss"]["event_reweight"]["enabled"] = value
        elif keys[:2] == ("loss", "event_reweight") and keys[2] not in ("enabled",):
            new_cfg.setdefault("loss", {}).setdefault("event_reweight", {})
            new_cfg["loss"]["event_reweight"][keys[2]] = value
        elif keys[:2] == ("distillation", "fallback"):
            new_cfg.setdefault("distillation", {}).setdefault("fallback", {})
            new_cfg["distillation"]["fallback"][keys[2]] = value
        elif keys[:2] == ("distillation", "student_focus"):
            new_cfg.setdefault("distillation", {}).setdefault("student_focus", {})
            new_cfg["distillation"]["student_focus"][keys[2]] = value
        else:
            set_nested(new_cfg, keys, value)
    new_cfg.setdefault("notes", {})["strategy_name"] = name
    return new_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Create enhanced teacher-match configs for next iteration.")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_pos_balanced.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="dynamic_distill/configs/wefend_teacher_match_enhanced",
    )
    parser.add_argument(
        "--run-script",
        default="scripts/run_wefend_teacher_match_enhanced.sh",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_teacher_match_enhanced"]

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
        name = cfg_path.stem
        for seed in args.seeds:
            gpu = GPU_ROUND_ROBIN[job_idx % len(GPU_ROUND_ROBIN)]
            log_path = f"logs/wefend_teacher_match_enhanced/{name}_seed{seed}.log"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {cfg_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1
            if job_idx % len(GPU_ROUND_ROBIN) == 0 and job_idx < total_jobs:
                run_lines.append("wait")
                run_lines.append("echo '--- enhanced batch slice complete ---'")

    run_lines.append("wait")
    run_lines.append("echo 'All enhanced jobs finished.'")

    run_script = Path(args.run_script)
    with run_script.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_script.chmod(0o755)

    print(f"Generated {len(generated_paths)} configs to {out_dir}")
    print(f"Run `bash {run_script}` to launch enhanced batch (GPUs 0/3/4/5/6).")


if __name__ == "__main__":
    main()
