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
        "longtrain_delta_compact",
        {
            ("distillation", "delta"): 0.062,
            ("distillation", "start_fraction"): 0.20,
            ("distillation", "end_fraction"): 0.66,
            ("distillation", "delta_schedule", "start_fraction"): 0.18,
            ("distillation", "delta_schedule", "end_fraction"): 0.70,
            ("distillation", "delta_schedule", "end_value"): 0.034,
        },
    ),
    (
        "longtrain_delta_soft",
        {
            ("distillation", "delta"): 0.055,
            ("distillation", "temperature"): 2.32,
            ("distillation", "adaptive_temperature", "base"): 2.32,
            ("distillation", "adaptive_temperature", "coeff"): 1.10,
            ("distillation", "lambda_kl"): 1.0,
            ("distillation", "delta_schedule", "start_fraction"): 0.22,
            ("distillation", "delta_schedule", "end_fraction"): 0.68,
            ("distillation", "delta_schedule", "end_value"): 0.030,
        },
    ),
    (
        "longtrain_pos_mild",
        {
            ("distillation", "positive_distill_boost"): 1.55,
            ("distillation", "positive_stage_boost"): 2.05,
            ("distillation", "positive_stage_start_fraction"): 0.44,
            ("distillation", "positive_stage_conf_margin"): 0.94,
        },
    ),
    (
        "longtrain_pos_balanced",
        {
            ("distillation", "positive_distill_boost"): 1.65,
            ("distillation", "positive_stage_boost"): 2.15,
            ("distillation", "positive_stage_start_fraction"): 0.46,
            ("distillation", "positive_student_conf_margin"): 0.89,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.955,
            ("distillation", "positive_event_gate", "student_conf"): 0.86,
        },
    ),
    (
        "longtrain_event_reweight",
        {
            ("distillation", "event_filter", "teacher_min_acc"): 0.88,
            ("distillation", "event_filter", "teacher_min_conf"): 0.915,
            ("loss", "event_reweight", "enabled"): True,
            ("loss", "event_reweight", "warmup_steps"): 6,
            ("loss", "event_reweight", "min_size"): 2,
            ("loss", "event_reweight", "scale"): 1.2,
            ("loss", "event_reweight", "power"): 1.3,
            ("loss", "event_reweight", "focus_positive"): True,
            ("loss", "event_reweight", "clip"): 2.2,
        },
    ),
    (
        "longtrain_temp_lowkl",
        {
            ("distillation", "temperature"): 2.30,
            ("distillation", "adaptive_temperature", "base"): 2.30,
            ("distillation", "adaptive_temperature", "coeff"): 1.05,
            ("distillation", "lambda_kl"): 0.98,
            ("distillation", "delta"): 0.065,
        },
    ),
    (
        "longtrain_feat_mix",
        {
            ("distillation", "lambda_fusion_to_text"): 0.14,
            ("distillation", "lambda_fusion_to_vision"): 0.14,
            ("distillation", "fusion_confidence"): 0.18,
            ("distillation", "positive_distill_boost"): 1.50,
            ("distillation", "positive_stage_boost"): 2.05,
        },
    ),
    (
        "longtrain_dual_mix",
        {
            ("distillation", "delta"): 0.058,
            ("distillation", "lambda_kl"): 1.08,
            ("distillation", "positive_distill_boost"): 1.60,
            ("distillation", "positive_stage_boost"): 2.10,
            ("distillation", "positive_stage_start_fraction"): 0.45,
            ("loss", "class_weights", "index1"): 2.45,
            ("distillation", "delta_schedule", "end_fraction"): 0.72,
            ("distillation", "delta_schedule", "end_value"): 0.032,
        },
    ),
    (
        "longtrain_student_focus",
        {
            ("distillation", "student_focus", "enabled"): True,
            ("distillation", "student_focus", "warmup"): 6,
            ("distillation", "student_focus", "threshold"): 0.92,
            ("distillation", "student_focus", "mode"): "pos_recall",
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
    parser = argparse.ArgumentParser(description="Create refined teacher-match configs.")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_longtrain.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="dynamic_distill/configs/wefend_teacher_match_refine",
    )
    parser.add_argument(
        "--run-script",
        default="scripts/run_wefend_teacher_match_refine.sh",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_teacher_match_refine"]

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
            log_path = f"logs/wefend_teacher_match_refine/{name}_seed{seed}.log"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {cfg_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1
            if job_idx % len(GPU_ROUND_ROBIN) == 0 and job_idx < total_jobs:
                run_lines.append("wait")
                run_lines.append("echo '--- refine batch slice complete ---'")

    run_lines.append("wait")
    run_lines.append("echo 'All refine jobs finished.'")

    run_script = Path(args.run_script)
    with run_script.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_script.chmod(0o755)

    print(f"Generated {len(generated_paths)} configs to {out_dir}")
    print(f"Run `bash {run_script}` to launch refine batch (GPUs 2-6).")


if __name__ == "__main__":
    main()
