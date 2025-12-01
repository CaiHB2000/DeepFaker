#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def set_nested(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Fakeddit6 student strategy YAMLs and a job list")
    ap.add_argument("--base", default="dynamic_distill/configs/fakeddit6_student_event_reliable.yaml")
    ap.add_argument("--outdir", default="dynamic_distill/configs/fakeddit6_strategies")
    ap.add_argument("--jobs", default="scripts/jobs_fakeddit6_time_students_stage1.txt")
    args = ap.parse_args()

    base_path = Path(args.base)
    with base_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    variants = {
        "event_reliable": {},
        "no_consensus": {("distillation", "consensus_gate", "enabled"): False},
        "no_posgate": {("distillation", "positive_event_gate", "enabled"): False},
        "bootstrap0": {("distillation", "bootstrap_fraction"): 0.0},
        "temp_low": {("distillation", "temperature"): 1.6},
        "temp_high": {("distillation", "temperature"): 2.6},
    }

    jobs: List[str] = []
    for name, patch in variants.items():
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("logging", {})
        cfg["logging"]["run_name"] = f"fakeddit6_student_{name}"
        for path, value in patch.items():
            set_nested(cfg, path, value)
        out_yaml = outdir / f"{name}.yaml"
        with out_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        # 3 seeds
        for seed in (0, 1, 2):
            log = f"logs/fakeddit6_time/{name}_seed{seed:02d}.log"
            jobs.append(f"{out_yaml.as_posix()} {seed} {log}")

    jobs_path = Path(args.jobs)
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    with jobs_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(jobs) + "\n")
    print(f"Wrote {len(jobs)} jobs -> {jobs_path}")


if __name__ == "__main__":
    main()

