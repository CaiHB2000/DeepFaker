#!/usr/bin/env python3
"""Generate full/observed views from Snopes raw posts."""
import argparse
import json
import pathlib
import re
from datetime import datetime, timezone

import requests


def sanitize_filename(url: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "-", url.split("?")[0].split("/")[-1])
    return name.strip("-") or "image"


def download_image(url: str, dest: pathlib.Path) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        fh.write(resp.content)
    return str(dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="datasets/dpmd_modalities/raw_posts/snopes")
    parser.add_argument("--views-dir", default="datasets/dpmd_modalities/views")
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    views_dir = pathlib.Path(args.views_dir)
    views_dir.mkdir(parents=True, exist_ok=True)

    for raw_file in raw_dir.glob("*.json"):
        data = json.loads(raw_file.read_text())
        event_id = data["event_id"]
        base_dir = views_dir / event_id
        base_dir.mkdir(parents=True, exist_ok=True)
        text_bundle = {
            "title": data.get("title"),
            "summary": data.get("summary"),
            "claim": data.get("claim"),
            "article_excerpt": data.get("article_excerpt", []),
            "rating": data.get("rating"),
        }
        image_path = None
        if data.get("og_image"):
            filename = sanitize_filename(data["og_image"])
            dest = base_dir / "media" / f"{filename}.jpg"
            try:
                image_path = download_image(data["og_image"], dest)
            except Exception as exc:
                print(f"[warn] failed to download image for {event_id}: {exc}")
                image_path = None
        timestamp = datetime.now(timezone.utc).isoformat()
        full_view = {
            "event_id": event_id,
            "view_id": f"{event_id}-full",
            "source": data.get("source"),
            "fact_check_url": data.get("fact_check_url"),
            "modality": {
                "text": True,
                "image": bool(image_path),
            },
            "text_bundle": text_bundle,
            "image_path": image_path,
            "generated_at": timestamp,
            "variant": "full_bundle",
        }
        observed_view = {
            "event_id": event_id,
            "view_id": f"{event_id}-textonly",
            "source": data.get("source"),
            "fact_check_url": data.get("fact_check_url"),
            "modality": {
                "text": True,
                "image": False,
            },
            "text_bundle": text_bundle,
            "missing_modalities": ["image"],
            "missing_reason": "simulated_drop_image",
            "generated_at": timestamp,
            "variant": "observed_view",
        }
        (base_dir / "full_bundle.json").write_text(json.dumps(full_view, ensure_ascii=False, indent=2), encoding="utf-8")
        (base_dir / "observed_textonly.json").write_text(json.dumps(observed_view, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Built views for {event_id}")


if __name__ == "__main__":
    main()
