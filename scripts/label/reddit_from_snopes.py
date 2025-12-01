#!/usr/bin/env python3
"""Project Snopes event labels onto Reddit posts per event."""
import argparse
import json
import pathlib
from datetime import datetime, timezone

def load_snopes_labels(path: pathlib.Path):
    labels = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            labels[obj["event_id"]] = obj
    return labels


def iter_reddit_posts(raw_dir: pathlib.Path):
    for raw_file in sorted(raw_dir.glob("*.jsonl")):
        with raw_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snopes-labels", default="datasets/dpmd_modalities/labels/snopes_auto.jsonl")
    parser.add_argument("--reddit-raw", default="datasets/dpmd_modalities/raw_posts/reddit")
    parser.add_argument("--output", default="datasets/dpmd_modalities/labels/reddit_from_snopes.jsonl")
    args = parser.parse_args()

    snopes_path = pathlib.Path(args.snopes_labels)
    reddit_dir = pathlib.Path(args.reddit_raw)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = load_snopes_labels(snopes_path)
    count = 0
    with output.open("w", encoding="utf-8") as fh:
        for post in iter_reddit_posts(reddit_dir):
            event_id = post.get("event_id")
            label_obj = labels.get(event_id)
            if not label_obj:
                continue
            entry = {
                "event_id": event_id,
                "post_id": post.get("id"),
                "fact_check_url": label_obj.get("fact_check_url"),
                "veracity": label_obj.get("veracity", "unknown"),
                "source_label": label_obj,
                "label_source": "snopes_event_projection",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} reddit labels -> {output}")


if __name__ == "__main__":
    main()
