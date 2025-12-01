#!/usr/bin/env python3
"""Fetch fact-check RSS feeds and build/update events.csv."""
import argparse
import csv
import datetime as dt
import hashlib
import os
import pathlib
import sys
import textwrap
import urllib.request
import xml.etree.ElementTree as ET

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DEFAULT_COLUMNS = [
    "event_id",
    "title",
    "summary",
    "fact_check_url",
    "category",
    "language",
    "published_at",
    "source_feed",
]


def load_config(path: pathlib.Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    if yaml is None:
        raise SystemExit("PyYAML is required. Install via `pip install pyyaml`.\n")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fetch_feed(url: str) -> ET.Element:
    with urllib.request.urlopen(url) as resp:  # nosec B310
        data = resp.read()
    return ET.fromstring(data)


def iter_items(root: ET.Element):
    channel = root.find("channel")
    items = channel.findall("item") if channel is not None else root.findall("item")
    for item in items:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()
        pub_date = (item.findtext("pubDate") or item.findtext("published") or "").strip()
        category = (item.findtext("category") or "").strip()
        yield {
            "title": title,
            "link": link,
            "summary": description,
            "published_at": pub_date,
            "category": category,
        }


def normalize_time(value: str) -> str:
    if not value:
        return dt.datetime.utcnow().strftime(ISO_FORMAT)
    try:
        parsed = dt.datetime.fromtimestamp(dt.datetime.strptime(value[:25], "%a, %d %b %Y %H:%M:%S").timestamp())
    except Exception:
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
        except Exception:
            return dt.datetime.utcnow().strftime(ISO_FORMAT)
    return parsed.strftime(ISO_FORMAT)


def make_event_id(title: str, url: str) -> str:
    base = (title or url or "event").encode("utf-8")
    digest = hashlib.sha1(base).hexdigest()[:10]
    slug = "".join(ch if ch.isalnum() else "-" for ch in title.lower())[:40].strip("-") or "event"
    return f"evt-{slug}-{digest}"


def load_existing(csv_path: pathlib.Path) -> dict:
    if not csv_path.exists():
        return {}
    existing = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            existing[row["fact_check_url"]] = row
    return existing


def append_events(csv_path: pathlib.Path, rows: list):
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DEFAULT_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--output", help="Override events.csv path")
    args = parser.parse_args()

    config = load_config(pathlib.Path(args.config))
    feeds = config.get("api", {}).get("fact_check_rss", [])
    if not feeds:
        raise SystemExit("No fact_check_rss configured")

    output = pathlib.Path(args.output or config.get("storage", {}).get("events_csv", "events.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing(output)
    new_rows = []
    for feed_url in feeds:
        try:
            root = fetch_feed(feed_url)
        except Exception as exc:
            print(f"[warn] failed to fetch {feed_url}: {exc}", file=sys.stderr)
            continue
        for item in iter_items(root):
            link = item["link"]
            if not link or link in existing:
                continue
            title = item["title"]
            event_id = make_event_id(title, link)
            row = {
                "event_id": event_id,
                "title": title,
                "summary": item["summary"],
                "fact_check_url": link,
                "category": item["category"] or "",
                "language": config.get("language", "zh"),
                "published_at": normalize_time(item["published_at"]),
                "source_feed": feed_url,
            }
            new_rows.append(row)
    if not new_rows:
        print("No new events.")
        return
    append_events(output, new_rows)
    print(f"Added {len(new_rows)} events to {output}")


if __name__ == "__main__":
    main()
