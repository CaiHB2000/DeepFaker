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
        "teacher_match_delta",
        {
            ("distillation", "start_fraction"): 0.20,
            ("distillation", "end_fraction"): 0.64,
            ("distillation", "require_student_mistake"): True,
            ("distillation", "delta"): 0.058,
            ("distillation", "delta_schedule", "start_fraction"): 0.20,
            ("distillation", "delta_schedule", "end_fraction"): 0.68,
            ("distillation", "delta_schedule", "end_value"): 0.034,
        },
    ),
    (
        "teacher_match_posfocus",
        {
            ("distillation", "positive_distill_boost"): 1.95,
            ("distillation", "positive_stage_boost"): 2.35,
            ("distillation", "positive_stage_start_fraction"): 0.40,
            ("distillation", "positive_stage_conf_margin"): 0.94,
            ("distillation", "positive_student_conf_margin"): 0.90,
            ("distillation", "require_student_mistake"): True,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.96,
            ("distillation", "positive_event_gate", "student_conf"): 0.84,
        },
    ),
    (
        "teacher_match_cost",
        {
            ("loss", "class_weights", "index1"): 2.65,
            ("distillation", "lambda_kl"): 1.12,
            ("distillation", "positive_distill_boost"): 1.80,
            ("distillation", "positive_stage_start_fraction"): 0.42,
            ("distillation", "positive_stage_boost"): 2.20,
            ("distillation", "agreement_confidence_gap"): 0.035,
        },
    ),
    (
        "teacher_match_uncertainty",
        {
            ("distillation", "delta"): 0.060,
            ("distillation", "require_student_mistake"): True,
            ("distillation", "uncertainty_weight", "enabled"): True,
            ("distillation", "uncertainty_weight", "scale"): 1.35,
            ("distillation", "uncertainty_weight", "power"): 1.10,
            ("distillation", "uncertainty_weight", "clip"): 2.0,
        },
    ),
    (
        "teacher_match_curriculum",
        {
            ("distillation", "start_fraction"): 0.18,
            ("distillation", "end_fraction"): 0.62,
            ("distillation", "require_student_mistake"): True,
            ("distillation", "positive_stage_start_fraction"): 0.36,
            ("distillation", "positive_stage_boost"): 2.05,
            ("distillation", "delta_schedule", "start_fraction"): 0.16,
            ("distillation", "delta_schedule", "end_fraction"): 0.60,
            ("distillation", "delta_schedule", "end_value"): 0.036,
        },
    ),
    (
        "teacher_match_studentfocus",
        {
            ("distillation", "student_focus", "enabled"): True,
            ("distillation", "student_focus", "warmup"): 6,
            ("distillation", "student_focus", "threshold"): 0.88,
            ("distillation", "student_focus", "mode"): "pos_recall",
        },
    ),
    (
        "teacher_match_eventreweight",
        {
            ("loss", "event_reweight", "enabled"): True,
            ("loss", "event_reweight", "warmup_steps"): 8,
            ("loss", "event_reweight", "min_size"): 2,
            ("loss", "event_reweight", "scale"): 1.4,
            ("loss", "event_reweight", "power"): 1.5,
            ("loss", "event_reweight", "focus_positive"): True,
            ("loss", "event_reweight", "clip"): 2.5,
        },
    ),
    (
        "teacher_match_longtrain",
        {
            ("training", "epochs"): 20,
            ("training", "early_stopping_patience"): 8,
            ("distillation", "require_student_mistake"): True,
            ("distillation", "end_fraction"): 0.70,
            ("distillation", "delta_schedule", "end_fraction"): 0.74,
            ("distillation", "positive_stage_start_fraction"): 0.48,
        },
    ),
    (
        "teacher_match_late",
        {
            ("distillation", "start_fraction"): 0.26,
            ("distillation", "end_fraction"): 0.72,
            ("distillation", "require_student_mistake"): True,
            ("distillation", "positive_stage_start_fraction"): 0.50,
            ("distillation", "positive_stage_boost"): 2.10,
            ("distillation", "delta"): 0.065,
        },
    ),
    (
        "teacher_match_tempplus",
        {
            ("distillation", "temperature"): 2.50,
            ("distillation", "adaptive_temperature", "base"): 2.50,
            ("distillation", "adaptive_temperature", "coeff"): 1.30,
            ("distillation", "lambda_kl"): 1.08,
            ("distillation", "delta"): 0.062,
        },
    ),
    (
        "teacher_match_tempminus",
        {
            ("distillation", "temperature"): 2.30,
            ("distillation", "adaptive_temperature", "base"): 2.30,
            ("distillation", "adaptive_temperature", "coeff"): 1.20,
            ("distillation", "require_student_mistake"): True,
            ("distillation", "lambda_kl"): 1.00,
            ("distillation", "delta"): 0.055,
            ("distillation", "confidence_gate", "margin"): 0.16,
        },
    ),
    (
        "teacher_match_dualstage",
        {
            ("distillation", "positive_distill_boost"): 1.65,
            ("distillation", "positive_stage_boost"): 2.15,
            ("distillation", "positive_stage_start_fraction"): 0.34,
            ("distillation", "positive_stage_conf_margin"): 0.93,
            ("distillation", "positive_student_conf_margin"): 0.89,
        },
    ),
    (
        "teacher_match_nomistake",
        {
            ("distillation", "require_student_mistake"): False,
            ("distillation", "confidence_gate", "margin"): 0.16,
            ("distillation", "positive_stage_boost"): 2.00,
            ("distillation", "positive_stage_start_fraction"): 0.38,
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

    new_cfg.setdefault("notes", {})["strategy_name"] = name
    return new_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Create teacher-match focused Wefend configs.")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_dynamic_distill_teacher_wcls_teacher_match.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="dynamic_distill/configs/wefend_teacher_match_batch",
    )
    parser.add_argument(
        "--run-script",
        default="scripts/run_wefend_teacher_match_batch.sh",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_teacher_match_batch"]

    job_idx = 0
    total_jobs = len(STRATEGIES) * len(args.seeds)
    for name, mods in STRATEGIES:
        cfg = apply_strategy(base_cfg, (name, mods))
        cfg.setdefault("logging", {})["output_dir"] = "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
        cfg.setdefault("logging", {})["run_name"] = f"wefend_{name}"

        cfg_path = out_dir / f"wefend_{name}.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        for seed in args.seeds:
            gpu = GPU_ROUND_ROBIN[job_idx % len(GPU_ROUND_ROBIN)]
            log_path = f"logs/wefend_teacher_match_batch/{cfg_path.stem}_seed{seed}.log"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {cfg_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1
            if job_idx % len(GPU_ROUND_ROBIN) == 0 and job_idx < total_jobs:
                run_lines.append("wait")
                run_lines.append("echo '--- batch complete ---'")

    run_lines.append("wait")
    run_lines.append("echo 'All teacher-match batch jobs finished.'")

    run_script = Path(args.run_script)
    with run_script.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_script.chmod(0o755)

    print(f"Generated {len(STRATEGIES)} configs to {out_dir}")
    print(f"Run `bash {run_script}` to launch all strategy jobs (GPUs 2-6).")


if __name__ == "__main__":
    main()
