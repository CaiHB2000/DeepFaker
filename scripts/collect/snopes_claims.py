#!/usr/bin/env python3
"""Fetch Snopes fact-check pages for each event and save structured metadata."""
import argparse
import csv
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Dict, Any

import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) DPMD-Dataset/0.1",
}


def read_events(csv_path: pathlib.Path):
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def parse_claimreview(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    data: Dict[str, Any] = {}
    for script in soup.find_all("script", type="application/ld+json"):
        text = script.string
        if not text:
            continue
        try:
            candidate = json.loads(text)
        except Exception:
            continue
        if isinstance(candidate, dict) and candidate.get("@type") == "ClaimReview":
            data["claim"] = candidate.get("claimReviewed")
            rating = candidate.get("reviewRating") or {}
            data["rating"] = rating.get("alternateName") or rating.get("ratingValue")
            data["rating_explanation"] = rating.get("description")
            break
    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        data["og_image"] = og_img["content"]
    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        data["og_description"] = og_desc["content"].strip()
    article_body = soup.find("div", class_="article-content")
    if article_body:
        paras = [p.get_text(strip=True) for p in article_body.find_all("p") if p.get_text(strip=True)]
        if paras:
            data["article_excerpt"] = paras[:5]
    return data


def fetch_url(url: str) -> str:
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="datasets/dpmd_modalities/events/events.csv")
    parser.add_argument("--output-dir", default="datasets/dpmd_modalities/raw_posts/snopes")
    args = parser.parse_args()

    csv_path = pathlib.Path(args.events)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in read_events(csv_path):
        url = row.get("fact_check_url")
        if not url or "snopes.com" not in url:
            continue
        event_id = row.get("event_id")
        out_path = out_dir / f"{event_id}.json"
        if out_path.exists():
            continue
        try:
            html = fetch_url(url)
        except Exception as exc:
            print(f"[warn] fetch failed {url}: {exc}", file=sys.stderr)
            continue
        data = parse_claimreview(html)
        payload = {
            "event_id": event_id,
            "source": "snopes",
            "fact_check_url": url,
            "title": row.get("title"),
            "summary": row.get("summary"),
            "category": row.get("category"),
            "claim": data.get("claim"),
            "rating": data.get("rating"),
            "rating_explanation": data.get("rating_explanation"),
            "og_image": data.get("og_image"),
            "og_description": data.get("og_description"),
            "article_excerpt": data.get("article_excerpt", []),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
