#!/usr/bin/env python
"""Build dual-stage WeFEND configs (stage1 pseudo-teacher + stage2 finetune)."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

Strategy = Tuple[str, Dict[Tuple[str, ...], Any], Dict[Tuple[str, ...], Any]]

STRATEGIES: List[Strategy] = [
    (
        "dualstage_balanced",
        {
            ("loss", "class_weights", "index1"): 2.85,
            ("distillation", "weak_teacher", "lambda"): 0.205,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.78,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.92,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.2,
            ("distillation", "event_layer", "weight_mid"): 0.82,
            ("distillation", "event_layer", "weight_low"): 0.50,
        },
        {
            ("optim", "lr"): 9.0e-06,
            ("training", "epochs"): 8,
            ("training", "early_stopping_patience"): 4,
            ("distillation", "start_fraction"): 0.1,
            ("distillation", "end_fraction"): 0.62,
            ("distillation", "weak_teacher", "lambda"): 0.12,
            ("distillation", "weak_teacher", "threshold"): 0.86,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.80,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.94,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.0,
            ("training", "max_steps_per_epoch"): 200,
        },
    ),
    (
        "dualstage_dualgate",
        {
            ("distillation", "start_fraction"): 0.18,
            ("distillation", "end_fraction"): 0.68,
            ("distillation", "delta_schedule", "start_fraction"): 0.18,
            ("distillation", "delta_schedule", "end_fraction"): 0.68,
            ("distillation", "delta_schedule", "end_value"): 0.046,
            ("distillation", "weak_teacher", "lambda"): 0.2,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.74,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.9,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.1,
        },
        {
            ("optim", "lr"): 8.5e-06,
            ("training", "epochs"): 8,
            ("distillation", "weak_teacher", "lambda"): 0.11,
            ("distillation", "delta"): 0.058,
            ("distillation", "start_fraction"): 0.12,
            ("distillation", "end_fraction"): 0.58,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.05,
        },
    ),
    (
        "dualstage_posfocus",
        {
            ("distillation", "positive_event_gate", "only"): True,
            ("distillation", "positive_event_gate", "teacher_conf"): 0.965,
            ("distillation", "positive_event_gate", "student_conf"): 0.82,
            ("distillation", "positive_stage_boost"): 2.5,
            ("distillation", "positive_distill_boost"): 2.05,
            ("distillation", "weak_teacher", "lambda"): 0.19,
            ("distillation", "weak_teacher", "dynamic_scale", "enabled"): True,
            ("distillation", "weak_teacher", "dynamic_scale", "min_score"): 0.76,
            ("distillation", "weak_teacher", "dynamic_scale", "max_score"): 0.91,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.3,
        },
        {
            ("optim", "lr"): 8.5e-06,
            ("training", "epochs"): 8,
            ("distillation", "positive_stage_start_fraction"): 0.32,
            ("distillation", "positive_stage_boost"): 2.2,
            ("distillation", "weak_teacher", "lambda"): 0.12,
            ("distillation", "weak_teacher", "dynamic_scale", "power"): 1.0,
        },
    ),
]

GPU_LIST = [2, 3, 4, 5, 6]


def set_nested(cfg: dict, keys: Tuple[str, ...], value: Any) -> None:
    target = cfg
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def apply_mods(cfg: dict, mods: Dict[Tuple[str, ...], Any]) -> dict:
    new_cfg = copy.deepcopy(cfg)
    for keys, value in mods.items():
        if keys == ("loss", "class_weights", "index1"):
            weights = new_cfg.setdefault("loss", {}).get("class_weights", [1.0, 1.0])
            if len(weights) < 2:
                weights = [1.0, 1.0]
            weights[1] = value
            new_cfg["loss"]["class_weights"] = weights
        else:
            set_nested(new_cfg, keys, value)
    return new_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-stage event strategies builder")
    parser.add_argument(
        "--base",
        default="dynamic_distill/configs/wefend_weakteacher_eventema_stage.yaml",
    )
    parser.add_argument("--out-dir", default="dynamic_distill/configs/wefend_event_dualstage")
    parser.add_argument("--run-script", default="scripts/run_wefend_event_dualstage.sh")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_lines = ["#!/usr/bin/env bash", "set -e", "mkdir -p logs/wefend_event_dualstage"]
    job_idx = 0
    generated_stage1: List[Path] = []
    stage2_paths: List[Tuple[Path, int]] = []

    total_stage1_jobs = len(STRATEGIES) * len(args.seeds)
    for name, mods_stage1, mods_stage2 in STRATEGIES:
        stage1_cfg = apply_mods(base_cfg, mods_stage1)
        stage1_cfg.setdefault("logging", {})["output_dir"] = (
            "paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2"
        )
        stage1_cfg["logging"]["run_name"] = f"{name}_stage1"
        stage1_cfg.setdefault("notes", {})["strategy_name"] = f"{name}_stage1"
        stage1_path = out_dir / f"{name}_stage1.yaml"
        with stage1_path.open("w") as f:
            yaml.safe_dump(stage1_cfg, f, sort_keys=False)
        generated_stage1.append(stage1_path)

        for seed in args.seeds:
            gpu = GPU_LIST[job_idx % len(GPU_LIST)]
            log_path = f"logs/wefend_event_dualstage/{name}_stage1_seed{seed:02d}.log"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
                f"--config {stage1_path} --seed {seed} --progress > {log_path} 2>&1 &"
            )
            run_lines.append(cmd)
            job_idx += 1
            if job_idx % len(GPU_LIST) == 0 and job_idx < total_stage1_jobs:
                run_lines.append("wait")
                run_lines.append("echo '--- dualstage stage1 slice complete ---'")

        # prepare stage2 configs per seed
        for seed in args.seeds:
            stage2_cfg = apply_mods(stage1_cfg, mods_stage2)
            stage2_cfg.setdefault("logging", {})["run_name"] = f"{name}_stage2"
            stage2_cfg.setdefault("notes", {})["strategy_name"] = f"{name}_stage2"
            stage2_cfg.setdefault("model", {})["init_checkpoint"] = (
                f"paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2/"
                f"{name}_stage1_seed{seed:02d}/model_best.pt"
            )
            stage2_path = out_dir / f"{name}_stage2_seed{seed:02d}.yaml"
            with stage2_path.open("w") as f:
                yaml.safe_dump(stage2_cfg, f, sort_keys=False)
            stage2_paths.append((stage2_path, seed))

    run_lines.append("wait")
    run_lines.append("echo '--- dualstage stage1 complete ---'")

    # stage2 commands reuse GPU round robin but start from 0 again
    job_idx = 0
    total_stage2_jobs = len(stage2_paths)
    for idx, (path, seed) in enumerate(stage2_paths, start=1):
        gpu = GPU_LIST[job_idx % len(GPU_LIST)]
        log_path = f"logs/wefend_event_dualstage/{path.stem}.log"
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} python dynamic_distill/scripts/train_mvp.py "
            f"--config {path} --seed {seed} --progress > {log_path} 2>&1 &"
        )
        run_lines.append(cmd)
        job_idx += 1
        if job_idx % len(GPU_LIST) == 0 and idx < total_stage2_jobs:
            run_lines.append("wait")
            run_lines.append("echo '--- dualstage stage2 slice complete ---'")
    run_lines.append("wait")
    run_lines.append("echo 'All dual-stage jobs finished.'")

    run_script = Path(args.run_script)
    with run_script.open("w") as f:
        f.write("\n".join(run_lines) + "\n")
    run_script.chmod(0o755)

    print(
        f"Generated {len(generated_stage1)} stage1 configs and {len(stage2_paths)} stage2 configs to {out_dir}"
    )
    print(f"Run `bash {run_script}` to execute stage1 then stage2 on GPUs {GPU_LIST}.")


if __name__ == "__main__":
    main()
