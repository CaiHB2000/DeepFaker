#!/usr/bin/env python
"""Generate configs that use pseudo-labeled train CSV with consensus/dynamic gates."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Tuple, Any, List

import yaml

Strategy = Tuple[str, Dict[Tuple[str, ...], Any]]

STRATEGIES: List[Strategy] = [
    (
        "pseudolabel_consensus_gate",
        {
            ("distillation", "consensus_gate", "enabled"): True,
            ("distillation", "consensus_gate", "min_conf"): 0.93,
            ("distillation", "consensus_gate", "require_agreement"): True,
            ("distillation", "weak_teacher", "lambda"): 0.21,
            ("distillation", "weak_teacher", "event_mix"): 0.75,
            ("training", "epochs"): 4,
        },
    ),
    (
        "pseudolabel_dynamic_softquota",
        {
            ("distillation", "weak_teacher", "lambda"): 0.22,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.78,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.95,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.2,
            ("distillation", "soft_quota", "enabled"): True,
            ("distillation", "soft_quota", "min_frac"): 0.05,
            ("distillation", "soft_quota", "max_frac"): 0.1,
            ("distillation", "soft_quota", "per_event_cap_frac"): 0.18,
            ("training", "epochs"): 4,
        },
    ),
    (
        "pseudolabel_dualgate_reweight",
        {
            ("distillation", "start_fraction"): 0.2,
            ("distillation", "end_fraction"): 0.72,
            ("distillation", "delta_schedule", "start_fraction"): 0.2,
            ("distillation", "delta_schedule", "end_fraction"): 0.72,
            ("distillation", "delta_schedule", "end_value"): 0.045,
            ("distillation", "weak_teacher", "lambda"): 0.215,
            ("loss", "event_reweight", "enabled"): True,
            ("loss", "event_reweight", "warmup_steps"): 3,
            ("loss", "event_reweight", "scale"): 0.45,
            ("loss", "event_reweight", "power"): 1.1,
            ("loss", "event_reweight", "focus_positive"): True,
            ("training", "epochs"): 4,
        },
    ),
]


def set_nested(cfg: Dict, keys: Tuple[str, ...], value: Any) -> None:
    target = cfg
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def apply_strategy(base: Dict, strategy: Strategy, pseudo_csv: Path) -> Dict:
    name, mods = strategy
    cfg = copy.deepcopy(base)
    root_path = Path(cfg["data"]["root"]).expanduser().resolve()
    try:
        pseudo_rel = pseudo_csv.resolve().relative_to(root_path)
        csv_value = str(pseudo_rel).replace("\\", "/")
    except Exception:
        csv_value = str(pseudo_csv)
    set_nested(cfg, ("data", "train", "csv_file"), csv_value)
    for keys, value in mods.items():
        set_nested(cfg, keys, value)
    cfg.setdefault("logging", {})["output_dir"] = "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
    cfg.setdefault("notes", {})["strategy_name"] = name
    cfg["logging"]["run_name"] = name
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pseudolabel consensus configs")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_dynamic_distill_teacher_wcls_event_reliable.yaml",
    )
    parser.add_argument("--pseudo-csv", default="datasets/wechat/pseudolabels/wefend_train_with_pseudo.csv")
    parser.add_argument("--out-dir", default="dynamic_distill/configs/wefend_pseudolabel_consensus")
    parser.add_argument("--job-file", default="scripts/wefend_pseudolabel_consensus_jobs.txt")
    parser.add_argument("--log-dir", default="logs/wefend_pseudolabel_consensus")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pseudo_csv = Path(args.pseudo_csv)

    job_lines: List[str] = []
    for name, strategy in STRATEGIES:
        cfg = apply_strategy(base_cfg, (name, strategy), pseudo_csv)
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
