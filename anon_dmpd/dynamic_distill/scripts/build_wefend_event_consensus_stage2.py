#!/usr/bin/env python
"""Stage-2 builder focusing on refined event-consensus strategies."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

Strategy = Tuple[str, Dict[Tuple[str, ...], Any]]

STRATEGIES: List[Strategy] = [
    (
        "event_consensus_balanced_ms",
        {
            ("loss", "class_weights", "index1"): 2.8,
            ("distillation", "weak_teacher", "lambda"): 0.19,
            ("distillation", "weak_teacher", "event_mix"): 0.62,
            ("distillation", "weak_teacher", "event_ema_decay"): 0.72,
            ("distillation", "event_layer", "weight_mid"): 0.82,
            ("distillation", "event_layer", "weight_low"): 0.52,
            ("distillation", "positive_stage_start_fraction"): 0.40,
        },
    ),
    (
        "event_consensus_dualgate_v2",
        {
            ("distillation", "start_fraction"): 0.18,
            ("distillation", "end_fraction"): 0.70,
            ("distillation", "delta_schedule", "start_fraction"): 0.18,
            ("distillation", "delta_schedule", "end_fraction"): 0.70,
            ("distillation", "delta_schedule", "end_value"): 0.045,
            ("distillation", "weak_teacher", "lambda"): 0.18,
            ("distillation", "weak_teacher", "threshold"): 0.835,
            ("distillation", "weak_teacher", "event_mix"): 0.58,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.962,
            ("distillation", "positive_event_gate", "student_conf"): 0.83,
        },
    ),
    (
        "event_consensus_calibrated_fix",
        {
            ("distillation", "event_calibration", "enabled"): True,
            ("distillation", "event_calibration", "min_seen"): 6,
            ("distillation", "event_calibration", "default_reliability"): 0.8,
            ("distillation", "event_calibration", "temp_scale"): 1.12,
            ("distillation", "event_calibration", "conf_scale"): 0.14,
            ("distillation", "event_calibration", "min_temp"): 0.78,
            ("distillation", "event_calibration", "max_temp"): 1.32,
            ("distillation", "weak_teacher", "lambda"): 0.18,
            ("distillation", "weak_teacher", "mix_ratio"): 0.65,
            ("distillation", "weak_teacher", "event_mix"): 0.65,
        },
    ),
    (
        "event_consensus_budget_positive",
        {
            ("distillation", "soft_quota", "enabled"): True,
            ("distillation", "soft_quota", "min_frac"): 0.05,
            ("distillation", "soft_quota", "max_frac"): 0.12,
            ("distillation", "soft_quota", "per_event_cap_frac"): 0.2,
            ("distillation", "soft_quota", "score_temp_coeff"): 0.30,
            ("distillation", "soft_quota", "score_kl_scale"): 0.4,
            ("distillation", "soft_quota", "min_reliability"): 0.6,
            ("distillation", "positive_event_gate", "only"): True,
            ("distillation", "positive_distill_boost"): 2.0,
            ("distillation", "weak_teacher", "lambda"): 0.17,
        },
    ),
    (
        "event_consensus_lambda_high",
        {
            ("distillation", "weak_teacher", "lambda"): 0.22,
            ("distillation", "weak_teacher", "threshold"): 0.86,
            ("distillation", "weak_teacher", "temperature"): 2.18,
            ("distillation", "weak_teacher", "event_mix"): 0.66,
            ("distillation", "weak_teacher", "mix_ratio"): 0.55,
            ("loss", "class_weights", "index1"): 2.9,
            ("distillation", "confidence_gate", "margin"): 0.19,
        },
    ),
    (
        "event_consensus_consistency_mix",
        {
            ("distillation", "consistency", "enabled"): True,
            ("distillation", "consistency", "lambda"): 0.16,
            ("distillation", "consistency", "metric"): "avg",
            ("distillation", "consistency", "max_score"): 0.88,
            ("distillation", "weak_teacher", "lambda"): 0.175,
            ("distillation", "weak_teacher", "event_mix"): 0.64,
            ("distillation", "weak_teacher", "mix_ratio"): 0.5,
        },
    ),
    (
        "event_consensus_ema_mixed",
        {
            ("distillation", "weak_teacher", "mix_ratio"): 0.45,
            ("distillation", "weak_teacher", "event_mix"): 0.78,
            ("distillation", "weak_teacher", "event_ema_decay"): 0.76,
            ("distillation", "weak_teacher", "lambda"): 0.18,
            ("distillation", "uncertainty_weight", "enabled"): True,
            ("distillation", "uncertainty_weight", "scale"): 1.3,
            ("distillation", "uncertainty_weight", "power"): 1.1,
        },
    ),
    (
        "event_consensus_posgate_strong",
        {
            ("distillation", "positive_event_gate", "only"): True,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.968,
            ("distillation", "positive_event_gate", "student_conf"): 0.82,
            ("distillation", "positive_stage_boost"): 2.6,
            ("distillation", "positive_stage_start_fraction"): 0.38,
            ("distillation", "positive_distill_boost"): 2.05,
            ("distillation", "weak_teacher", "lambda"): 0.18,
            ("distillation", "weak_teacher", "threshold"): 0.84,
        },
    ),
]

GPU_LIST = [2, 3, 4, 5, 6]


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
    distil = new_cfg.setdefault("distillation", {})
    distil.setdefault("weak_teacher", {}).setdefault("enabled", True)
    distil.setdefault("event_layer", {}).setdefault("enabled", True)
    distil.setdefault("positive_event_gate", {}).setdefault("enabled", True)
    distil.setdefault("soft_quota", {}).setdefault("enabled", False)
    distil.setdefault("event_calibration", {}).setdefault("enabled", False)
    distil.setdefault("consistency", {}).setdefault("enabled", False)
    distil.setdefault("uncertainty_weight", {}).setdefault("enabled", False)
    return new_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stage-2 event-consensus configs.")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_weakteacher_eventema_stage.yaml",
    )
    parser.add_argument("--out-dir", default="dynamic_distill/configs/wefend_event_consensus_stage2")
    parser.add_argument("--run-script", default="scripts/run_wefend_event_consensus_stage2.sh")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_event_consensus_stage2"]

    job_idx = 0
    total_jobs = len(STRATEGIES) * len(args.seeds)
    generated_paths: List[Path] = []
    for name, mods in STRATEGIES:
        cfg = apply_strategy(base_cfg, (name, mods))
        cfg.setdefault("notes", {})["strategy_name"] = name
        cfg.setdefault("logging", {})["output_dir"] = (
            "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
        )
        cfg.setdefault("logging", {})["run_name"] = name

        cfg_path = out_dir / f"{name}.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        generated_paths.append(cfg_path)

        for seed in args.seeds:
            gpu = GPU_LIST[job_idx % len(GPU_LIST)]
            log_path = (
                f"logs/wefend_event_consensus_stage2/{cfg_path.stem}_seed{seed:02d}.log"
            )
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {cfg_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1
            if job_idx % len(GPU_LIST) == 0 and job_idx < total_jobs:
                run_lines.append("wait")
                run_lines.append("echo '--- stage2 slice complete ---'")

    run_lines.append("wait")
    run_lines.append("echo 'All stage-2 event-consensus jobs finished.'")

    run_script = Path(args.run_script)
    with run_script.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_script.chmod(0o755)

    print(f"Generated {len(generated_paths)} configs to {out_dir}")
    print(f"Run `bash {run_script}` to launch stage-2 jobs on GPUs {GPU_LIST}.")


if __name__ == "__main__":
    main()
