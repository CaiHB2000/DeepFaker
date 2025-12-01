#!/usr/bin/env python
"""Generate pseudo-label focused WeFEND configs and job list."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

Strategy = Tuple[str, Dict[Tuple[str, ...], Any]]

STRATEGIES: List[Strategy] = [
    (
        "pseudolabel_strict",
        {
            ("distillation", "weak_teacher", "lambda"): 0.22,
            ("distillation", "weak_teacher", "event_ema_enabled"): True,
            ("distillation", "weak_teacher", "event_ema_decay"): 0.75,
            ("distillation", "weak_teacher", "event_mix"): 0.85,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.78,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.94,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.2,
            ("distillation", "soft_quota", "enabled"): True,
            ("distillation", "soft_quota", "min_frac"): 0.05,
            ("distillation", "soft_quota", "max_frac"): 0.12,
            ("distillation", "soft_quota", "per_event_cap_frac"): 0.2,
            ("distillation", "positive_event_gate", "only"): True,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.97,
            ("distillation", "positive_event_gate", "student_conf"): 0.83,
        },
    ),
    (
        "pseudolabel_relaxed",
        {
            ("distillation", "weak_teacher", "lambda"): 0.20,
            ("distillation", "weak_teacher", "event_ema_enabled"): True,
            ("distillation", "weak_teacher", "event_mix"): 0.7,
            ("distillation", "event_layer", "weight_mid"): 0.8,
            ("distillation", "event_layer", "weight_low"): 0.55,
            ("distillation", "event_filter", "teacher_min_acc"): 0.88,
            ("distillation", "event_filter", "teacher_min_conf"): 0.9,
            ("distillation", "event_calibration", "enabled"): True,
            ("distillation", "event_calibration", "min_seen"): 5,
            ("distillation", "event_calibration", "default_reliability"): 0.8,
            ("distillation", "event_calibration", "temp_scale"): 1.1,
        },
    ),
    (
        "pseudolabel_dualgate",
        {
            ("distillation", "start_fraction"): 0.18,
            ("distillation", "end_fraction"): 0.68,
            ("distillation", "delta_schedule", "start_fraction"): 0.18,
            ("distillation", "delta_schedule", "end_fraction"): 0.68,
            ("distillation", "delta_schedule", "end_value"): 0.045,
            ("distillation", "weak_teacher", "lambda"): 0.215,
            ("distillation", "weak_teacher", "threshold"): 0.83,
            ("distillation", "weak_teacher", "event_mix"): 0.8,
            ("distillation", "student_focus", "enabled"): True,
            ("distillation", "student_focus", "warmup"): 4,
            ("distillation", "student_focus", "threshold"): 0.9,
            ("distillation", "student_focus", "mode"): "pos_recall",
        },
    ),
    (
        "pseudolabel_budget",
        {
            ("distillation", "soft_quota", "enabled"): True,
            ("distillation", "soft_quota", "min_frac"): 0.04,
            ("distillation", "soft_quota", "max_frac"): 0.1,
            ("distillation", "soft_quota", "per_event_cap_frac"): 0.18,
            ("distillation", "soft_quota", "min_reliability"): 0.6,
            ("distillation", "weak_teacher", "lambda"): 0.205,
            ("distillation", "weak_teacher", "event_ema_enabled"): True,
            ("distillation", "weak_teacher", "event_mix"): 0.75,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.76,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.9,
        },
    ),
    (
        "pseudolabel_consistency",
        {
            ("distillation", "consistency", "enabled"): True,
            ("distillation", "consistency", "lambda"): 0.15,
            ("distillation", "consistency", "metric"): "avg",
            ("distillation", "consistency", "max_score"): 0.88,
            ("distillation", "weak_teacher", "lambda"): 0.2,
            ("distillation", "weak_teacher", "event_ema_enabled"): True,
            ("distillation", "weak_teacher", "event_mix"): 0.72,
            ("distillation", "uncertainty_weight", "enabled"): True,
            ("distillation", "uncertainty_weight", "scale"): 1.3,
            ("distillation", "uncertainty_weight", "power"): 1.1,
        },
    ),
    (
        "pseudolabel_stage",
        {
            ("training", "epochs"): 18,
            ("optim", "lr"): 1.45e-05,
            ("distillation", "start_fraction"): 0.2,
            ("distillation", "end_fraction"): 0.72,
            ("distillation", "weak_teacher", "lambda"): 0.215,
            ("distillation", "weak_teacher", "event_mix"): 0.78,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.8,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.95,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.0,
            ("loss", "event_reweight", "enabled"): True,
            ("loss", "event_reweight", "warmup_steps"): 4,
            ("loss", "event_reweight", "min_size"): 3,
            ("loss", "event_reweight", "scale"): 0.4,
            ("loss", "event_reweight", "power"): 1.2,
            ("loss", "event_reweight", "focus_positive"): True,
        },
    ),
]


def set_nested(cfg: dict, keys: Tuple[str, ...], value: Any) -> None:
    target = cfg
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def apply_strategy(base_cfg: dict, strategy: Strategy) -> dict:
    name, modifications = strategy
    cfg = copy.deepcopy(base_cfg)
    for keys, value in modifications.items():
        set_nested(cfg, keys, value)
    cfg.setdefault("logging", {})["output_dir"] = "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
    cfg.setdefault("notes", {})["strategy_name"] = name
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pseudo-label configs and job list.")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_weakteacher_eventema_stage.yaml",
    )
    parser.add_argument("--out-dir", default="dynamic_distill/configs/wefend_event_pseudolabel")
    parser.add_argument("--job-file", default="scripts/wefend_event_pseudolabel_jobs.txt")
    parser.add_argument("--log-dir", default="logs/wefend_event_pseudolabel")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    job_lines: List[str] = []
    for name, strategy in STRATEGIES:
        cfg = apply_strategy(base_cfg, (name, strategy))
        cfg["logging"]["run_name"] = name
        cfg_path = out_dir / f"{name}.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        for seed in args.seeds:
            log_path = log_dir / f"{name}_seed{seed:02d}.log"
            job_lines.append(f"{cfg_path} {seed} {log_path}")

    Path(args.job_file).write_text("\n".join(job_lines) + "\n")
    print(f"Generated {len(STRATEGIES)} configs in {out_dir}")
    print(f"Job list written to {args.job_file}")
    print(f"Logs will be stored under {log_dir}")


if __name__ == "__main__":
    main()
