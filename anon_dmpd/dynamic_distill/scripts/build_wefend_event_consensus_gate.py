#!/usr/bin/env python
"""Build WeFEND configs with consensus gating variants."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

Strategy = Tuple[str, Dict[Tuple[str, ...], Any]]

STRATEGIES: List[Strategy] = [
    (
        "consensus_gate_strict",
        {
            ("distillation", "consensus_gate", "enabled"): True,
            ("distillation", "consensus_gate", "min_conf"): 0.94,
            ("distillation", "consensus_gate", "require_agreement"): True,
            ("distillation", "positive_event_gate", "only"): True,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.965,
            ("distillation", "positive_event_gate", "student_conf"): 0.83,
            ("distillation", "weak_teacher", "lambda"): 0.195,
        },
    ),
    (
        "consensus_gate_relaxed",
        {
            ("distillation", "consensus_gate", "enabled"): True,
            ("distillation", "consensus_gate", "min_conf"): 0.90,
            ("distillation", "consensus_gate", "require_agreement"): False,
            ("distillation", "event_layer", "weight_mid"): 0.8,
            ("distillation", "weak_teacher", "lambda"): 0.19,
            ("distillation", "soft_quota", "enabled"): True,
            ("distillation", "soft_quota", "min_frac"): 0.03,
            ("distillation", "soft_quota", "max_frac"): 0.08,
        },
    ),
    (
        "consensus_gate_dualgate",
        {
            ("distillation", "consensus_gate", "enabled"): True,
            ("distillation", "consensus_gate", "min_conf"): 0.92,
            ("distillation", "consensus_gate", "require_agreement"): True,
            ("distillation", "start_fraction"): 0.18,
            ("distillation", "end_fraction"): 0.70,
            ("distillation", "delta_schedule", "start_fraction"): 0.18,
            ("distillation", "delta_schedule", "end_fraction"): 0.70,
            ("distillation", "delta_schedule", "end_value"): 0.045,
            ("distillation", "weak_teacher", "lambda"): 0.2,
            ("distillation", "event_filter", "teacher_min_acc"): 0.91,
            ("distillation", "event_filter", "teacher_min_conf"): 0.93,
        },
    ),
    (
        "consensus_gate_budget",
        {
            ("distillation", "consensus_gate", "enabled"): True,
            ("distillation", "consensus_gate", "min_conf"): 0.91,
            ("distillation", "consensus_gate", "require_agreement"): True,
            ("distillation", "soft_quota", "enabled"): True,
            ("distillation", "soft_quota", "min_frac"): 0.04,
            ("distillation", "soft_quota", "max_frac"): 0.1,
            ("distillation", "soft_quota", "per_event_cap_frac"): 0.18,
            ("distillation", "weak_teacher", "lambda"): 0.2,
            ("distillation", "weak_teacher", "event_mix"): 0.75,
        },
    ),
]


def set_nested(cfg: Dict, keys: Tuple[str, ...], value: Any) -> None:
    target = cfg
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def apply_strategy(base: Dict, strategy: Strategy) -> Dict:
    name, mods = strategy
    cfg = copy.deepcopy(base)
    for keys, value in mods.items():
        set_nested(cfg, keys, value)
    cfg.setdefault("logging", {})["output_dir"] = "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
    cfg.setdefault("notes", {})["strategy_name"] = name
    cfg["logging"]["run_name"] = name
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consensus gate configs")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_dynamic_distill_teacher_wcls_event_reliable.yaml",
    )
    parser.add_argument("--out-dir", default="dynamic_distill/configs/wefend_event_consensus_gate")
    parser.add_argument("--job-file", default="scripts/wefend_event_consensus_gate_jobs.txt")
    parser.add_argument("--log-dir", default="logs/wefend_event_consensus_gate")
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
        cfg_path = out_dir / f"{name}.yaml"
        with cfg_path.open("w") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        for seed in args.seeds:
            log_path = log_dir / f"{name}_seed{seed:02d}.log"
            job_lines.append(f"{cfg_path} {seed} {log_path}")

    Path(args.job_file).write_text("\n".join(job_lines) + "\n")
    print(f"Generated {len(STRATEGIES)} configs in {out_dir}")
    print(f"Job list written to {args.job_file}")
    print(f"Logs will be stored in {log_dir}")


if __name__ == "__main__":
    main()
