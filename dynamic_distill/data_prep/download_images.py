import argparse
import concurrent.futures
import hashlib
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DataPrepBot/1.0)",
}


def download_one(row: pd.Series, out_dir: Path, timeout: float = 10.0) -> Optional[str]:
    url = row["image_url"]
    if not url or not isinstance(url, str):
        return None
    guid = row["guid"]
    ext = ".jpg"
    hashed = hashlib.sha256(guid.encode("utf-8")).hexdigest()
    out_path = out_dir / f"{hashed}{ext}"
    if out_path.exists():
        return str(out_path)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200:
            out_path.write_bytes(resp.content)
            return str(out_path)
    except Exception:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download images for multimodal dataset CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV with columns guid,image_url")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for _, row in df.iterrows():
            futures.append(executor.submit(download_one, row, args.output_dir))
        for fut in concurrent.futures.as_completed(futures):
            fut.result()


if __name__ == "__main__":
    main()
