#!/usr/bin/env python
"""Deterministic batch builder for WEFEND event-consensus distillation strategies."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

Strategy = Tuple[str, Dict[Tuple[str, ...], Any]]

# Hand-designed strategy variants focusing on event-level consensus / weak-teacher control.
STRATEGIES: List[Strategy] = [
    (
        "event_consensus_highmix",
        {
            ("distillation", "weak_teacher", "lambda"): 0.185,
            ("distillation", "weak_teacher", "mix_ratio"): 0.6,
            ("distillation", "weak_teacher", "event_mix"): 0.78,
            ("distillation", "weak_teacher", "event_ema_decay"): 0.74,
            ("distillation", "weak_teacher", "threshold"): 0.845,
            ("distillation", "positive_distill_boost"): 1.82,
            ("distillation", "confidence_gate", "margin"): 0.175,
        },
    ),
    (
        "event_consensus_lowtemp",
        {
            ("distillation", "temperature"): 2.26,
            ("distillation", "adaptive_temperature", "base"): 2.26,
            ("distillation", "adaptive_temperature", "coeff"): 1.02,
            ("distillation", "delta"): 0.062,
            ("distillation", "weak_teacher", "temperature"): 2.05,
            ("distillation", "weak_teacher", "lambda"): 0.165,
            ("distillation", "weak_teacher", "threshold"): 0.83,
            ("distillation", "weak_teacher", "event_mix"): 0.6,
        },
    ),
    (
        "event_consensus_posfocus",
        {
            ("distillation", "positive_distill_boost"): 1.95,
            ("distillation", "positive_stage_start_fraction"): 0.40,
            ("distillation", "positive_stage_boost"): 2.55,
            ("distillation", "positive_event_gate", "only"): True,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.965,
            ("distillation", "positive_event_gate", "student_conf"): 0.83,
            ("loss", "class_weights", "index1"): 2.65,
            ("distillation", "weak_teacher", "lambda"): 0.175,
            ("distillation", "weak_teacher", "threshold"): 0.84,
        },
    ),
    (
        "event_consensus_dualgate",
        {
            ("distillation", "start_fraction"): 0.20,
            ("distillation", "end_fraction"): 0.68,
            ("distillation", "delta_schedule", "start_fraction"): 0.20,
            ("distillation", "delta_schedule", "end_fraction"): 0.68,
            ("distillation", "delta_schedule", "end_value"): 0.048,
            ("distillation", "weak_teacher", "threshold"): 0.82,
            ("distillation", "weak_teacher", "min_seen"): 4,
            ("distillation", "weak_teacher", "lambda"): 0.16,
            ("distillation", "weak_teacher", "event_mix"): 0.55,
            ("distillation", "event_layer", "min_seen"): 5,
            ("distillation", "event_layer", "weight_low"): 0.50,
            ("distillation", "event_layer", "weight_mid"): 0.78,
        },
    ),
    (
        "event_consensus_calibrated",
        {
            ("distillation", "event_calibration", "enabled"): True,
            ("distillation", "event_calibration", "min_seen"): 6,
            ("distillation", "event_calibration", "default_reliability"): 0.78,
            ("distillation", "event_calibration", "temp_scale"): 1.15,
            ("distillation", "event_calibration", "conf_scale"): 0.12,
            ("distillation", "event_calibration", "min_temp"): 0.75,
            ("distillation", "event_calibration", "max_temp"): 1.35,
            ("distillation", "weak_teacher", "lambda"): 0.17,
            ("distillation", "weak_teacher", "event_ema_decay"): 0.68,
            ("distillation", "weak_teacher", "event_mix"): 0.6,
        },
    ),
    (
        "event_consensus_softquota",
        {
            ("distillation", "soft_quota", "enabled"): True,
            ("distillation", "soft_quota", "min_frac"): 0.04,
            ("distillation", "soft_quota", "max_frac"): 0.10,
            ("distillation", "soft_quota", "per_event_cap_frac"): 0.18,
            ("distillation", "soft_quota", "score_temp_coeff"): 0.35,
            ("distillation", "soft_quota", "score_kl_scale"): 0.45,
            ("distillation", "soft_quota", "min_reliability"): 0.55,
            ("distillation", "weak_teacher", "lambda"): 0.165,
            ("distillation", "weak_teacher", "event_mix"): 0.62,
        },
    ),
    (
        "event_consensus_negassist",
        {
            ("distillation", "weak_teacher", "apply_negative"): True,
            ("distillation", "weak_teacher", "threshold"): 0.865,
            ("distillation", "weak_teacher", "lambda"): 0.14,
            ("distillation", "weak_teacher", "temperature"): 2.28,
            ("distillation", "student_focus", "enabled"): True,
            ("distillation", "student_focus", "warmup"): 4,
            ("distillation", "student_focus", "threshold"): 0.9,
            ("distillation", "student_focus", "mode"): "pos_recall",
            ("distillation", "positive_distill_boost"): 1.70,
        },
    ),
    (
        "event_consensus_costschedule",
        {
            ("loss", "class_weights", "index1"): 2.70,
            ("distillation", "lambda_kl"): 1.12,
            ("distillation", "delta"): 0.068,
            ("distillation", "temperature"): 2.34,
            ("distillation", "adaptive_temperature", "base"): 2.34,
            ("distillation", "weak_teacher", "lambda"): 0.18,
            ("distillation", "weak_teacher", "event_mix"): 0.65,
            ("distillation", "event_layer", "weight_mid"): 0.80,
        },
    ),
    (
        "event_consensus_curriculum",
        {
            ("training", "epochs"): 18,
            ("training", "max_steps_per_epoch"): 250,
            ("training", "early_stopping_patience"): 7,
            ("distillation", "start_fraction"): 0.18,
            ("distillation", "end_fraction"): 0.74,
            ("distillation", "delta_schedule", "start_fraction"): 0.18,
            ("distillation", "delta_schedule", "end_fraction"): 0.74,
            ("distillation", "delta_schedule", "end_value"): 0.040,
            ("distillation", "weak_teacher", "lambda"): 0.165,
            ("distillation", "weak_teacher", "mix_ratio"): 0.55,
            ("distillation", "weak_teacher", "event_mix"): 0.62,
        },
    ),
    (
        "event_consensus_dualview",
        {
            ("distillation", "lambda_fusion_to_text"): 0.18,
            ("distillation", "lambda_fusion_to_vision"): 0.18,
            ("distillation", "fusion_confidence"): 0.20,
            ("distillation", "weak_teacher", "mix_ratio"): 0.50,
            ("distillation", "weak_teacher", "event_mix"): 0.70,
            ("distillation", "weak_teacher", "lambda"): 0.17,
            ("distillation", "weak_teacher", "threshold"): 0.84,
            ("distillation", "agreement_confidence_gap"): 0.02,
        },
    ),
    (
        "event_consensus_emahard",
        {
            ("distillation", "weak_teacher", "mix_ratio"): 0.4,
            ("distillation", "weak_teacher", "event_ema_decay"): 0.78,
            ("distillation", "weak_teacher", "event_mix"): 0.80,
            ("distillation", "weak_teacher", "lambda"): 0.17,
            ("distillation", "uncertainty_weight", "enabled"): True,
            ("distillation", "uncertainty_weight", "scale"): 1.35,
            ("distillation", "uncertainty_weight", "power"): 1.1,
            ("distillation", "uncertainty_weight", "clip"): 2.0,
        },
    ),
    (
        "event_consensus_balanced",
        {
            ("distillation", "event_layer", "metric"): "avg",
            ("distillation", "event_layer", "high_threshold"): 0.93,
            ("distillation", "event_layer", "mid_threshold"): 0.86,
            ("distillation", "event_layer", "weight_high"): 1.05,
            ("distillation", "event_layer", "weight_mid"): 0.78,
            ("distillation", "event_layer", "weight_low"): 0.50,
            ("distillation", "event_layer", "default_weight"): 0.9,
            ("loss", "event_reweight", "enabled"): True,
            ("loss", "event_reweight", "warmup_steps"): 4,
            ("loss", "event_reweight", "min_size"): 3,
            ("loss", "event_reweight", "scale"): 0.35,
            ("loss", "event_reweight", "power"): 1.15,
            ("loss", "event_reweight", "focus_positive"): True,
            ("loss", "event_reweight", "clip"): 2.0,
            ("distillation", "weak_teacher", "lambda"): 0.16,
            ("distillation", "weak_teacher", "event_mix"): 0.58,
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

    # ensure optional sub-structures exist with sane defaults
    distil = new_cfg.setdefault("distillation", {})
    distil.setdefault("soft_quota", {"enabled": False, "min_frac": 0.0})
    distil.setdefault("positive_event_gate", {}).setdefault("enabled", True)
    distil.setdefault("event_layer", {}).setdefault("enabled", True)
    distil.setdefault("weak_teacher", {}).setdefault("enabled", True)
    distil.setdefault("uncertainty_weight", {})
    return new_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Create event-consensus Wefend configs.")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_weakteacher_eventema_stage.yaml",
    )
    parser.add_argument("--out-dir", default="dynamic_distill/configs/wefend_event_consensus")
    parser.add_argument("--run-script", default="scripts/run_wefend_event_consensus.sh")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_event_consensus"]

    job_idx = 0
    generated_paths: List[Path] = []
    for name, mods in STRATEGIES:
        cfg = apply_strategy(base_cfg, (name, mods))
        cfg.setdefault("notes", {})["strategy_name"] = name
        cfg.setdefault("logging", {})["output_dir"] = (
            "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
        )
        cfg.setdefault("logging", {})["run_name"] = f"{name}"

        cfg_path = out_dir / f"{name}.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        generated_paths.append(cfg_path)

        for seed in args.seeds:
            gpu = GPU_ROUND_ROBIN[job_idx % len(GPU_ROUND_ROBIN)]
            log_path = f"logs/wefend_event_consensus/{cfg_path.stem}_seed{seed}.log"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {cfg_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1

    run_lines.append("wait")
    run_lines.append("echo 'All event-consensus jobs finished.'")

    run_script = Path(args.run_script)
    with run_script.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_script.chmod(0o755)

    print(f"Generated {len(generated_paths)} configs to {out_dir}")
    print(f"Run `bash {run_script}` to launch all strategy jobs (GPUs 2-6).")


if __name__ == "__main__":
    main()
