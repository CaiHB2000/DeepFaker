# -*- coding: utf-8 -*-
"""
Compute early propagation features for face-tracked Reddit posts.

Example:
    python -m p2p.analysis.compute_face_propagation \\
        --watch-dir tmp/reddit_seed_realnews \\
        --out-csv tmp/reddit_seed_realnews/face_propagation_features.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate propagation dynamics features for face-tracked posts.")
    ap.add_argument(
        "--watch-dir",
        required=True,
        help="Directory containing posts_status_faces.csv and propagation_timeseries.csv (watcher output).",
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Destination CSV path (default: <watch-dir>/face_propagation_features.csv).",
    )
    ap.add_argument(
        "--max-hours",
        type=float,
        default=6.0,
        help="Upper limit (hours since creation) for the 'early window' features (default 6h).",
    )
    return ap


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    return pd.read_csv(path)


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _window_snapshot(group: pd.DataFrame, max_hours: float) -> pd.Series | None:
    window = group[group["hours_since_create"] <= max_hours]
    if window.empty:
        return None
    return window.iloc[-1]


def compute_features(faces_df: pd.DataFrame, prop_df: pd.DataFrame, early_hours: float) -> pd.DataFrame:
    face_posts = set(faces_df["posting_id"])
    prop_faces = prop_df[prop_df["posting_id"].isin(face_posts)].copy()
    if prop_faces.empty:
        return pd.DataFrame(columns=["posting_id"])

    prop_faces = prop_faces.sort_values(["posting_id", "snapshot_utc"])
    records: List[Dict[str, float]] = []
    for posting_id, group in prop_faces.groupby("posting_id"):
        group = group.sort_values("snapshot_utc")
        if group.empty:
            continue

        created = group["created_utc"].iloc[0]
        group = group.assign(hours_since_create=(group["snapshot_utc"] - created) / 3600.0)

        first = group.iloc[0]
        last = group.iloc[-1]
        duration_hours = max((last["snapshot_utc"] - first["snapshot_utc"]) / 3600.0, 1e-6)

        windows = {
            "5m": 5.0 / 60.0,
            "15m": 15.0 / 60.0,
            "60m": 60.0 / 60.0,
            "6h": early_hours,
        }
        window_snaps: Dict[str, pd.Series | None] = {name: _window_snapshot(group, hours) for name, hours in windows.items()}
        has_window = {name: snap is not None for name, snap in window_snaps.items()}

        early_snap = window_snaps["6h"] if has_window["6h"] else None
        early_duration = max(early_snap["hours_since_create"], 1e-6) if early_snap is not None else float("nan")

        score_growth = (last["score"] - first["score"]) / duration_hours
        comments_growth = (last["num_comments"] - first["num_comments"]) / duration_hours
        early_score = early_snap["score"] if early_snap is not None else float("nan")
        early_comments = early_snap["num_comments"] if early_snap is not None else float("nan")
        early_score_growth = (
            (early_score - first["score"]) / early_duration if early_snap is not None else float("nan")
        )
        early_comment_growth = (
            (early_comments - first["num_comments"]) / early_duration if early_snap is not None else float("nan")
        )

        first_comment = group[group["num_comments"] > 0]
        first_comment_delay = (
            first_comment["hours_since_create"].iloc[0] if not first_comment.empty else float("nan")
        )

        records.append(
            {
                "posting_id": posting_id,
                "snapshots": len(group),
                "duration_hours": duration_hours,
                "first_score": first["score"],
                "last_score": last["score"],
                "first_comments": first["num_comments"],
                "last_comments": last["num_comments"],
                "score_growth_per_hour": score_growth,
                "comments_growth_per_hour": comments_growth,
                "early_score": early_score,
                "early_comments": early_comments,
                "early_score_growth_per_hour": early_score_growth,
                "early_comment_growth_per_hour": early_comment_growth,
                "early_window_hours": early_duration,
                "points_6h": len(group[group["hours_since_create"] <= early_hours]),
                "has_window_5m": int(has_window["5m"]),
                "has_window_15m": int(has_window["15m"]),
                "has_window_60m": int(has_window["60m"]),
                "has_window_6h": int(has_window["6h"]),
                "score_5m": window_snaps["5m"]["score"] if has_window["5m"] else float("nan"),
                "score_15m": window_snaps["15m"]["score"] if has_window["15m"] else float("nan"),
                "score_60m": window_snaps["60m"]["score"] if has_window["60m"] else float("nan"),
                "comments_5m": window_snaps["5m"]["num_comments"] if has_window["5m"] else float("nan"),
                "comments_15m": window_snaps["15m"]["num_comments"] if has_window["15m"] else float("nan"),
                "comments_60m": window_snaps["60m"]["num_comments"] if has_window["60m"] else float("nan"),
                "hours_5m": window_snaps["5m"]["hours_since_create"] if has_window["5m"] else float("nan"),
                "hours_15m": window_snaps["15m"]["hours_since_create"] if has_window["15m"] else float("nan"),
                "hours_60m": window_snaps["60m"]["hours_since_create"] if has_window["60m"] else float("nan"),
                "score_growth_0_5m": (
                    (window_snaps["5m"]["score"] - first["score"]) / max(window_snaps["5m"]["hours_since_create"], 1e-6)
                    if has_window["5m"]
                    else float("nan")
                ),
                "score_growth_5_15m": (
                    (window_snaps["15m"]["score"] - window_snaps["5m"]["score"])
                    / max(
                        window_snaps["15m"]["hours_since_create"] - window_snaps["5m"]["hours_since_create"],
                        1e-6,
                    )
                    if has_window["5m"] and has_window["15m"]
                    else float("nan")
                ),
                "score_growth_15_60m": (
                    (window_snaps["60m"]["score"] - window_snaps["15m"]["score"])
                    / max(
                        window_snaps["60m"]["hours_since_create"] - window_snaps["15m"]["hours_since_create"],
                        1e-6,
                    )
                    if has_window["15m"] and has_window["60m"]
                    else float("nan")
                ),
                "burst_ratio_5_to_60": (
                    (window_snaps["60m"]["score"] - window_snaps["5m"]["score"])
                    / max(window_snaps["60m"]["score"], 1.0)
                    if has_window["5m"] and has_window["60m"] and window_snaps["60m"]["score"] not in (0, float("nan"))
                    else float("nan")
                ),
                "first_comment_delay_hours": first_comment_delay,
            }
        )

    features_df = pd.DataFrame.from_records(records)
    merged = faces_df.merge(features_df, on="posting_id", how="left")
    return merged


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    watch_dir = Path(args.watch_dir).resolve()
    posts_faces = _load_csv(watch_dir / "posts_status_faces.csv")
    propagation = _load_csv(watch_dir / "propagation_timeseries.csv")

    _to_numeric(
        posts_faces,
        [
            "score",
            "num_comments",
            "upvote_ratio",
            "total_awards_received",
            "view_count",
            "created_utc",
            "first_seen_utc",
            "last_snapshot_utc",
            "face_crops",
        ],
    )
    _to_numeric(
        propagation,
        [
            "snapshot_utc",
            "score",
            "num_comments",
            "upvote_ratio",
            "total_awards_received",
            "view_count",
            "upvotes",
            "downvotes",
            "created_utc",
        ],
    )

    features = compute_features(posts_faces, propagation, args.max_hours)
    out_path = Path(args.out_csv).resolve() if args.out_csv else watch_dir / "face_propagation_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    print(f"[face-propagation] wrote {len(features)} rows -> {out_path}")


if __name__ == "__main__":
    main()
