#!/usr/bin/env python3
"""Generate keyword lists for each event using heuristics + optional GPT assist."""
import argparse
import csv
import json
import os
import pathlib
import re
import sys
from typing import List

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

try:
    import requests  # type: ignore
except ImportError:
    requests = None


def load_config(path: pathlib.Path) -> dict:
    if yaml is None:
        raise SystemExit("PyYAML is required. Install via `pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def simple_keywords(title: str) -> List[str]:
    title = re.sub(r"[\\p{Punct}]+", " ", title)
    base = [tok for tok in re.split(r"\s+", title) if tok]
    uniq = []
    for tok in base:
        tok = tok.strip().lower()
        if tok and tok not in uniq:
            uniq.append(tok)
    return uniq[:10]


def gpt_keywords(title: str, api_cfg: dict) -> List[str]:
    if not api_cfg.get("key"):
        return []
    if requests is None:
        return []
    payload = {
        "model": api_cfg.get("model", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": "提取不超过8个关键词或短语，使用中文或常见音译。"},
            {"role": "user", "content": title},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_cfg['key']}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(api_cfg.get("endpoint", "https://api.openai.com/v1/chat/completions"), json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parts = re.split(r"[，,\n;]+", content)
        return [p.strip() for p in parts if p.strip()]
    except Exception as exc:
        print(f"[warn] GPT keyword generation failed: {exc}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--events", help="Override events csv path")
    parser.add_argument("--output", help="keywords.json output")
    args = parser.parse_args()

    config = load_config(pathlib.Path(args.config))
    csv_path = pathlib.Path(args.events or config["storage"]["events_csv"])
    if not csv_path.exists():
        raise SystemExit(f"Events file not found: {csv_path}")

    output_path = pathlib.Path(args.output or config["storage"]["keywords_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gpt_cfg = {
        "key": os.getenv("GPT_API_KEY"),
        "endpoint": config.get("api", {}).get("gpt_endpoint"),
        "model": config.get("api", {}).get("gpt_model", "gpt-4o-mini"),
    }

    data = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            title = row["title"]
            keywords = simple_keywords(title)
            ai_kw = gpt_keywords(title, gpt_cfg)
            merged = []
            for word in keywords + ai_kw:
                word = word.strip()
                if word and word not in merged:
                    merged.append(word)
            data[row["event_id"]] = {
                "title": title,
                "keywords": merged,
                "fact_check_url": row["fact_check_url"],
            }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"Wrote keywords for {len(data)} events -> {output_path}")


if __name__ == "__main__":
    main()
