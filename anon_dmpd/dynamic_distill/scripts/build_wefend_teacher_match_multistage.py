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
        "pos_balanced_uncertainty",
        {
            ("distillation", "uncertainty_weight", "enabled"): True,
            ("distillation", "uncertainty_weight", "scale"): 1.3,
            ("distillation", "uncertainty_weight", "power"): 1.1,
            ("distillation", "uncertainty_weight", "clip"): 2.2,
            ("distillation", "delta"): 0.062,
        },
    ),
    (
        "pos_balanced_fallback_stage",
        {
            ("distillation", "fallback", "enabled"): True,
            ("distillation", "fallback", "min_pairs_fraction"): 0.06,
            ("distillation", "fallback", "positive_only"): True,
            ("distillation", "fallback", "teacher_conf"): 0.94,
            ("distillation", "fallback", "student_conf"): 0.92,
            ("distillation", "fallback", "temperature"): 2.25,
            ("distillation", "fallback", "lambda_scale"): 0.22,
            ("distillation", "fallback", "require_student_mistake"): False,
            ("distillation", "fallback", "start_fraction"): 0.35,
        },
    ),
    (
        "pos_balanced_two_stage",
        {
            ("distillation", "positive_stage_start_fraction"): 0.40,
            ("distillation", "positive_stage_boost"): 2.30,
            ("distillation", "positive_distill_boost"): 1.75,
            ("distillation", "delta_schedule", "start_fraction"): 0.20,
            ("distillation", "delta_schedule", "end_fraction"): 0.74,
            ("distillation", "delta_schedule", "end_value"): 0.030,
        },
    ),
    (
        "pos_balanced_reweight_light",
        {
            ("loss", "event_reweight", "enabled"): True,
            ("loss", "event_reweight", "warmup_steps"): 8,
            ("loss", "event_reweight", "min_size"): 2,
            ("loss", "event_reweight", "scale"): 1.1,
            ("loss", "event_reweight", "power"): 1.2,
            ("loss", "event_reweight", "focus_positive"): True,
            ("loss", "event_reweight", "clip"): 1.8,
            ("distillation", "event_filter", "teacher_min_acc"): 0.89,
            ("distillation", "event_filter", "teacher_min_conf"): 0.918,
        },
    ),
    (
        "pos_balanced_student_temp",
        {
            ("distillation", "student_focus", "enabled"): True,
            ("distillation", "student_focus", "warmup"): 6,
            ("distillation", "student_focus", "threshold"): 0.91,
            ("distillation", "student_focus", "mode"): "pos_recall",
            ("distillation", "temperature"): 2.32,
            ("distillation", "adaptive_temperature", "base"): 2.32,
            ("distillation", "adaptive_temperature", "coeff"): 1.08,
            ("distillation", "lambda_kl"): 1.02,
        },
    ),
    (
        "pos_balanced_gate_dual",
        {
            ("distillation", "event_filter", "teacher_min_acc"): 0.88,
            ("distillation", "event_filter", "teacher_min_conf"): 0.915,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.950,
            ("distillation", "positive_event_gate", "student_conf"): 0.85,
            ("distillation", "positive_event_gate", "only"): False,
            ("distillation", "positive_student_conf_margin"): 0.87,
        },
    ),
]

GPU_ROUND_ROBIN = [0, 3, 4, 5, 6]


def ensure_event_reweight(cfg: dict) -> None:
    cfg.setdefault("loss", {}).setdefault("event_reweight", {"enabled": False})


def ensure_fallback(cfg: dict) -> None:
    cfg.setdefault("distillation", {}).setdefault("fallback", {})


def ensure_student_focus(cfg: dict) -> None:
    cfg.setdefault("distillation", {}).setdefault("student_focus", {})


def ensure_uncertainty(cfg: dict) -> None:
    cfg.setdefault("distillation", {}).setdefault("uncertainty_weight", {})


def set_nested(cfg: dict, keys: Tuple[str, ...], value: Any) -> None:
    target = cfg
    for key in keys[:-1]:
        if key not in target or target[key] is None:
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def apply_strategy(base_cfg: dict, strategy: Strategy) -> dict:
    name, modifications = strategy
    cfg = copy.deepcopy(base_cfg)
    for keys, value in modifications.items():
        if keys[:2] == ("loss", "event_reweight"):
            ensure_event_reweight(cfg)
            if keys[2] == "enabled":
                cfg["loss"]["event_reweight"]["enabled"] = value
            else:
                cfg["loss"]["event_reweight"][keys[2]] = value
        elif keys[:2] == ("distillation", "fallback"):
            ensure_fallback(cfg)
            cfg["distillation"]["fallback"][keys[2]] = value
        elif keys[:2] == ("distillation", "student_focus"):
            ensure_student_focus(cfg)
            cfg["distillation"]["student_focus"][keys[2]] = value
        elif keys[:2] == ("distillation", "uncertainty_weight"):
            ensure_uncertainty(cfg)
            cfg["distillation"]["uncertainty_weight"][keys[2]] = value
        else:
            set_nested(cfg, keys, value)
    cfg.setdefault("notes", {})["strategy_name"] = name
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Create multi-stage teacher-match configs for next experiments.")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_pos_balanced.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="dynamic_distill/configs/wefend_teacher_match_multistage",
    )
    parser.add_argument(
        "--run-script",
        default="scripts/run_wefend_teacher_match_multistage.sh",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_teacher_match_multistage"]

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
            gpu = GPU_ROUND_ROBIN[job_idx % len(GPU_ROUND_ROBIN)]
            log_path = f"logs/wefend_teacher_match_multistage/{stem}_seed{seed}.log"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {cfg_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1
            if job_idx % len(GPU_ROUND_ROBIN) == 0 and job_idx < total_jobs:
                run_lines.append("wait")
                run_lines.append("echo '--- multistage batch slice complete ---'")

    run_lines.append("wait")
    run_lines.append("echo 'All multistage jobs finished.'")

    run_script = Path(args.run_script)
    with run_script.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_script.chmod(0o755)

    print(f"Generated {len(generated_paths)} configs to {out_dir}")
    print(f"Run `bash {run_script}` to launch multistage batch (GPUs 0/3/4/5/6).")


if __name__ == "__main__":
    main()
