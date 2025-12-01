#!/usr/bin/env python
"""Generate WeFEND pseudo-labeled training CSV by filtering high-confidence predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def load_predictions(path: Path, prob_column: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"Prediction file {path} must contain 'id' column")
    if "prediction" not in df.columns:
        raise ValueError(f"Prediction file {path} must contain 'prediction' column")
    if prob_column and prob_column not in df.columns:
        raise ValueError(f"Column {prob_column} not found in {path}")
    if prob_column is None:
        if "prob_1" in df.columns and "prob_0" in df.columns:
            df["prob_max"] = df[["prob_0", "prob_1"]].max(axis=1)
        elif "confidence" in df.columns:
            df["prob_max"] = df["confidence"].astype(float)
        else:
            raise ValueError(f"Could not infer probability column in {path}")
    else:
        df["prob_max"] = df[prob_column].astype(float)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter WeFEND predictions into pseudo-labeled CSV")
    parser.add_argument("--train-csv", type=Path, required=True, help="Original train CSV path")
    parser.add_argument("--image-dir", type=Path, required=True, help="Image directory (for validation)")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--source", action="append", nargs=2, metavar=("CSV", "PREDS"),
        help="Pairs of source CSV (val/test) and prediction CSV to harvest from",
        required=True,
    )
    parser.add_argument("--threshold", type=float, default=0.97, help="Minimum max probability to accept")
    parser.add_argument("--require-correct", action="store_true", help="Only keep samples where prediction == label")
    parser.add_argument("--max-per-event", type=int, default=50, help="Limit pseudo samples per event")
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(args.train_csv)
    if "guid" not in base_df.columns:
        raise ValueError("Train CSV must contain 'guid' column")
    seen_ids = set(base_df["guid"].astype(str))

    pseudo_frames: List[pd.DataFrame] = []
    for csv_path_str, preds_path_str in args.source:
        csv_path = Path(csv_path_str)
        preds_path = Path(preds_path_str)
        src_df = pd.read_csv(csv_path)
        preds_df = load_predictions(preds_path)
        merged = preds_df.merge(src_df, left_on="id", right_on="guid", how="inner", suffixes=("_pred", ""))
        merged["prediction"] = merged["prediction"].astype(int)
        merged["label"] = merged["label"].astype(int)
        mask = merged["prob_max"] >= args.threshold
        if args.require_correct:
            mask &= merged["prediction"] == merged["label"]
        filtered = merged.loc[mask].copy()
        if filtered.empty:
            continue
        filtered["label"] = filtered["prediction"]
        filtered = filtered.drop(columns=[col for col in filtered.columns if col.endswith("_pred")])
        filtered["source_split"] = csv_path.stem
        pseudo_frames.append(filtered)

    if not pseudo_frames:
        raise RuntimeError("No pseudo-labeled samples matched the criteria")

    pseudo_df = pd.concat(pseudo_frames, ignore_index=True)
    if "event_id" in pseudo_df.columns:
        pseudo_df = pseudo_df.sort_values("prob_max", ascending=False)
        pseudo_df = (
            pseudo_df.groupby("event_id", group_keys=False)
            .head(max(args.max_per_event, 1))
        )

    pseudo_df = pseudo_df[~pseudo_df["guid"].astype(str).isin(seen_ids)]
    combined_df = pd.concat([base_df, pseudo_df[base_df.columns]], ignore_index=True)
    combined_df.to_csv(args.output_csv, index=False)
    print(f"Saved combined train CSV with {len(combined_df)} rows (pseudo added: {len(pseudo_df)}) -> {args.output_csv}")


if __name__ == "__main__":
    main()
