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
        "pos_boost_strict",
        {
            ("distillation", "positive_distill_boost"): 2.0,
            ("distillation", "positive_stage_boost"): 2.3,
            ("distillation", "positive_stage_start_fraction"): 0.40,
            ("distillation", "positive_stage_conf_margin"): 0.96,
            ("distillation", "positive_student_conf_margin"): 0.95,
            ("distillation", "confidence_gate", "margin"): 0.16,
            ("distillation", "require_student_mistake"): True,
            ("distillation", "positive_event_gate", "only"): True,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.95,
            ("distillation", "positive_event_gate", "student_conf"): 0.84,
        },
    ),
    (
        "pos_boost_soft",
        {
            ("distillation", "positive_distill_boost"): 1.8,
            ("distillation", "positive_stage_boost"): 2.0,
            ("distillation", "positive_stage_start_fraction"): 0.36,
            ("distillation", "positive_stage_conf_margin"): 0.94,
            ("distillation", "positive_student_conf_margin"): 0.92,
            ("distillation", "require_student_mistake"): False,
            ("distillation", "positive_event_gate", "only"): False,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.94,
            ("distillation", "positive_event_gate", "student_conf"): 0.87,
            ("distillation", "confidence_gate", "margin"): 0.14,
        },
    ),
    (
        "cost_sensitive",
        {
            ("loss", "class_weights", "index1"): 2.5,
            ("distillation", "lambda_kl"): 1.12,
            ("distillation", "agreement_confidence_gap"): 0.025,
            ("distillation", "positive_distill_boost"): 1.75,
            ("distillation", "confidence_gate", "margin"): 0.15,
        },
    ),
    (
        "fusion_emphasis",
        {
            ("distillation", "lambda_fusion_to_text"): 0.16,
            ("distillation", "lambda_fusion_to_vision"): 0.16,
            ("distillation", "fusion_confidence"): 0.18,
            ("distillation", "positive_distill_boost"): 1.6,
            ("distillation", "require_student_mistake"): True,
        },
    ),
    (
        "teacher_match",
        {
            ("distillation", "require_student_mistake"): True,
            ("distillation", "fusion_teacher_must_match"): True,
            ("distillation", "confidence_gate", "margin"): 0.15,
            ("distillation", "positive_distill_boost"): 1.7,
        },
    ),
    (
        "uncertainty_focus",
        {
            ("distillation", "uncertainty_weight", "enabled"): True,
            ("distillation", "uncertainty_weight", "scale"): 1.4,
            ("distillation", "uncertainty_weight", "power"): 1.2,
            ("distillation", "delta"): 0.06,
            ("distillation", "start_fraction"): 0.18,
        },
    ),
    (
        "event_relaxed",
        {
            ("distillation", "event_filter", "teacher_min_acc"): 0.85,
            ("distillation", "event_filter", "teacher_min_conf"): 0.90,
            ("distillation", "event_filter", "warmup_steps"): 3,
            ("distillation", "start_fraction"): 0.18,
            ("distillation", "end_fraction"): 0.60,
        },
    ),
    (
        "event_strict",
        {
            ("distillation", "event_filter", "teacher_min_acc"): 0.90,
            ("distillation", "event_filter", "teacher_min_conf"): 0.93,
            ("distillation", "confidence_gate", "margin"): 0.18,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.96,
            ("distillation", "positive_event_gate", "student_conf"): 0.85,
        },
    ),
    (
        "early_stage",
        {
            ("distillation", "start_fraction"): 0.14,
            ("distillation", "positive_stage_start_fraction"): 0.32,
            ("distillation", "positive_distill_boost"): 1.9,
            ("distillation", "end_fraction"): 0.58,
        },
    ),
    (
        "late_stage",
        {
            ("distillation", "start_fraction"): 0.26,
            ("distillation", "end_fraction"): 0.72,
            ("distillation", "positive_stage_start_fraction"): 0.50,
            ("distillation", "positive_stage_boost"): 2.4,
            ("distillation", "delta_schedule", "start_fraction"): 0.26,
            ("distillation", "delta_schedule", "end_fraction"): 0.72,
            ("distillation", "delta_schedule", "end_value"): 0.038,
        },
    ),
    (
        "delta_schedule",
        {
            ("distillation", "delta_schedule", "start_fraction"): 0.18,
            ("distillation", "delta_schedule", "end_fraction"): 0.66,
            ("distillation", "delta_schedule", "end_value"): 0.032,
            ("distillation", "delta"): 0.058,
            ("distillation", "temperature"): 2.38,
        },
    ),
    (
        "fallback_light",
        {
            ("distillation", "fallback_distill_enabled"): True,
            ("distillation", "fallback_min_pairs_fraction"): 0.10,
            ("distillation", "fallback_teacher_conf"): 0.94,
            ("distillation", "fallback_student_conf"): 0.93,
            ("distillation", "fallback_temperature"): 2.6,
            ("distillation", "fallback_lambda_scale"): 0.25,
            ("distillation", "positive_distill_boost"): 1.7,
        },
    ),
]

GPU_ROUND_ROBIN = [2, 3, 4, 5, 6]


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
        if keys == ("loss", "class_weights", "index1"):
            class_weights = new_cfg.setdefault("loss", {}).get("class_weights", [1.0, 1.0])
            if len(class_weights) < 2:
                class_weights = [1.0, 1.0]
            class_weights[1] = value
            new_cfg.setdefault("loss", {})["class_weights"] = class_weights
        else:
            set_nested(new_cfg, keys, value)
    # ensure fallback disabled when not explicitly enabled
    if not new_cfg.get("distillation", {}).get("fallback_distill_enabled"):
        new_cfg.setdefault("distillation", {}).update(
            {
                "fallback_distill_enabled": False,
                "fallback_min_pairs_fraction": 0.0,
            }
        )
    return new_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Create designed Wefend configs.")
    parser.add_argument("--base", default="dynamic_distill/configs/wefend_dynamic_distill_teacher_wcls_event_reliable.yaml")
    parser.add_argument("--out-dir", default="dynamic_distill/configs/auto_wefend")
    parser.add_argument("--run-script", default="scripts/run_wefend_strategies.sh")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_strategies"]

    job_idx = 0
    generated_paths = []
    for name, mods in STRATEGIES:
        cfg = apply_strategy(base_cfg, (name, mods))
        cfg.setdefault("notes", {})["strategy_name"] = name
        cfg.setdefault("logging", {})["output_dir"] = "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
        cfg.setdefault("logging", {})["run_name"] = f"wefend_{name}"

        cfg_path = out_dir / f"wefend_{name}.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        generated_paths.append(cfg_path)

        for seed in args.seeds:
            gpu = GPU_ROUND_ROBIN[job_idx % len(GPU_ROUND_ROBIN)]
            log_path = f"logs/wefend_strategies/{cfg_path.stem}_seed{seed}.log"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {cfg_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1

    run_lines.append("wait")
    run_lines.append("echo 'All strategy jobs finished.'")

    run_script = Path(args.run_script)
    with run_script.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_script.chmod(0o755)

    print(f"Generated {len(generated_paths)} configs to {out_dir}")
    print(f"Run `bash {run_script}` to launch all strategy jobs (GPUs 2-6).")


if __name__ == "__main__":
    main()
