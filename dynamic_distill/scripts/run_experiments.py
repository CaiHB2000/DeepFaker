#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install pyyaml to run experiment orchestrator: `pip install pyyaml`.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for dynamic distillation ablations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dynamic_distill/configs/default_mvp.yaml"),
        help="Base configuration file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Root directory for training outputs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs for quick experiments.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps per epoch (e.g., 50 for smoke tests).",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Subset of experiment keys to run (defaults to all).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="List of seeds to run for each experiment.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to invoke training script.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display tqdm progress bars during training runs.",
    )
    parser.add_argument(
        "--trace-loader",
        action="store_true",
        help="Print detailed data-loading timing for the first few batches.",
    )
    return parser.parse_args()


def deep_update(base: Dict, overrides: Dict) -> Dict:
    for key, value in overrides.items():
        if isinstance(value, dict):
            node = base.setdefault(key, {})
            deep_update(node, value)
        else:
            base[key] = value
    return base


def load_summary(run_dir: Path) -> Dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found for run: {run_dir}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as base_handle:
        base_config = yaml.safe_load(base_handle)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiments: Dict[str, Dict] = {
        "dynamic_full": {},
        "no_distill": {
            "loss": {"gamma": 0.0},
            "teacher": {"use_ema": False},
        },
        "no_feature": {
            "distillation": {"lambda_feat": 0.0},
        },
        "no_evidence": {
            "loss": {"beta": 0.0},
        },
        "no_ema": {
            "teacher": {"use_ema": False},
        },
        "high_delta": {
            "distillation": {"delta": 0.1},
        },
    }

    if args.experiments:
        missing = [key for key in args.experiments if key not in experiments]
        if missing:
            raise ValueError(f"Unknown experiment keys: {missing}")
        experiments = {key: experiments[key] for key in args.experiments}

    seeds = args.seeds if args.seeds else [None]

    results: List[Dict] = []
    for name, overrides in experiments.items():
        config_copy = copy.deepcopy(base_config)
        if args.epochs is not None:
            config_copy.setdefault("training", {})["epochs"] = args.epochs
        if args.max_steps is not None:
            config_copy.setdefault("training", {})["max_steps_per_epoch"] = args.max_steps
        config_copy.setdefault("logging", {})["output_dir"] = str(args.output_root)
        config_copy["logging"]["overwrite"] = True
        deep_update(config_copy, overrides)

        for seed in seeds:
            config_seed = copy.deepcopy(config_copy)
            run_name_base = f"{timestamp}_{name}"
            config_seed["logging"]["run_name"] = run_name_base

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
                yaml.safe_dump(config_seed, tmp)
                tmp_path = Path(tmp.name)

            print(f"[orchestrator] Running experiment '{name}' seed={seed}...")
            cmd = [args.python, "dynamic_distill/scripts/train_mvp.py", "--config", str(tmp_path)]
            if seed is not None:
                cmd.extend(["--seed", str(seed)])
            if overrides.get("loss", {}).get("gamma", config_seed.get("loss", {}).get("gamma", 1.0)) == 0.0:
                cmd.append("--disable-distill")
            if args.progress:
                cmd.append("--progress")
            if args.trace_loader:
                cmd.append("--trace-loader")
            subprocess.run(cmd, check=True)

            run_dir = args.output_root / (
                run_name_base if seed is None else f"{run_name_base}_seed{seed:02d}"
            )
            summary = load_summary(run_dir)
            test_metrics = summary.get("test_metrics", {}) or {}
            record = {
                "experiment": name,
                "seed": seed,
                "run_dir": str(run_dir),
                **{f"test_{k}": test_metrics.get(k) for k in ("loss", "acc", "f1_macro", "f1_pos", "ece")},
            }
            results.append(record)
            tmp_path.unlink(missing_ok=True)

    if results:
        summary_path = args.output_root / f"experiment_summary_{timestamp}.yaml"
        with summary_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(results, handle)
        csv_path = args.output_root / f"experiment_summary_{timestamp}.csv"
        fieldnames = sorted({key for record in results for key in record.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"[orchestrator] Summary saved to {summary_path}")
        print(f"[orchestrator] CSV summary saved to {csv_path}")


if __name__ == "__main__":
    main()
