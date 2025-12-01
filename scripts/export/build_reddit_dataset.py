#!/usr/bin/env python3
"""Package reddit views + labels into dataset splits and stats."""
import argparse
import json
import pathlib
from collections import Counter, defaultdict


def load_labels(path: pathlib.Path):
    mapping = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mapping[(obj.get("event_id"), obj.get("post_id"))] = obj
    return mapping


def iter_view_dirs(views_root: pathlib.Path):
    for event_dir in sorted(views_root.iterdir()):
        if not event_dir.is_dir():
            continue
        for post_dir in event_dir.iterdir():
            if not post_dir.is_dir():
                continue
            full_file = post_dir / "full_bundle.json"
            text_file = post_dir / "observed_textonly.json"
            if not full_file.exists():
                continue
            yield event_dir.name, post_dir.name, full_file, text_file if text_file.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views-dir", default="datasets/dpmd_modalities/views")
    parser.add_argument("--labels", default="datasets/dpmd_modalities/labels/reddit_from_snopes.jsonl")
    parser.add_argument("--output-dir", default="datasets/dpmd_modalities/releases/reddit_v0")
    args = parser.parse_args()

    views_dir = pathlib.Path(args.views_dir)
    labels = load_labels(pathlib.Path(args.labels))
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "full_views.jsonl"
    text_path = out_dir / "textonly_views.jsonl"

    label_stats = Counter()
    event_counts = defaultdict(int)

    with full_path.open("w", encoding="utf-8") as full_fh, text_path.open("w", encoding="utf-8") as text_fh:
        for event_id, post_id, full_file, text_file in iter_view_dirs(views_dir):
            label_obj = labels.get((event_id, post_id)) or labels.get((event_id, None))
            veracity = (label_obj or {}).get("veracity", "unknown")
            entry = json.loads(full_file.read_text())
            entry["veracity"] = veracity
            entry["fact_check_url"] = (label_obj or {}).get("fact_check_url")
            entry["label_source"] = (label_obj or {}).get("label_source")
            full_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            label_stats[veracity] += 1
            event_counts[event_id] += 1
            if text_file:
                text_entry = json.loads(text_file.read_text())
                text_entry["veracity"] = veracity
                text_entry["fact_check_url"] = entry.get("fact_check_url")
                text_entry["label_source"] = entry.get("label_source")
                text_fh.write(json.dumps(text_entry, ensure_ascii=False) + "\n")
    stats = {
        "total_posts": sum(label_stats.values()),
        "label_distribution": dict(label_stats),
        "events": dict(event_counts),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dataset to {out_dir} (posts={stats['total_posts']})")


if __name__ == "__main__":
    main()
