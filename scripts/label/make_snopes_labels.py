#!/usr/bin/env python3
"""Derive structured labels from Snopes metadata."""
import argparse
import json
import pathlib
from datetime import datetime, timezone

RATING_MAP = {
    "true": "true",
    "false": "false",
    "mixture": "partial",
    "miscaptioned": "partial",
    "outdated": "partial",
    "satire": "partial",
    "unproven": "unknown",
    "context": "partial",
}


def normalize_rating(value: str) -> str:
    if not value:
        return "unknown"
    key = value.strip().lower()
    return RATING_MAP.get(key, key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="datasets/dpmd_modalities/raw_posts/snopes")
    parser.add_argument("--output", default="datasets/dpmd_modalities/labels/snopes_auto.jsonl")
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as fh:
        for raw_file in sorted(raw_dir.glob("*.json")):
            data = json.loads(raw_file.read_text())
            rating = data.get("rating") or "unknown"
            entry = {
                "event_id": data["event_id"],
                "fact_check_url": data.get("fact_check_url"),
                "veracity": normalize_rating(rating),
                "raw_rating": rating,
                "rating_explanation": data.get("rating_explanation"),
                "auto_source": "snopes_rating",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote labels to {output}")


if __name__ == "__main__":
    main()
