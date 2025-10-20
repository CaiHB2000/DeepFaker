# p2p/runners/run_webmeta_tables.py
# -*- coding: utf-8 -*-
"""
用法：
  python -m p2p.runners.run_webmeta_tables \
    --webmeta_jsonl results/sample/webmeta.jsonl \
    --out_dir       results/sample
"""

from __future__ import annotations
import argparse
from p2p.datasource.webmeta_tables import jsonl_to_tables

def build_argparser():
    ap = argparse.ArgumentParser(description="Convert webmeta JSONL to CSV and build skeleton tables.")
    ap.add_argument("--webmeta_jsonl", required=True, help="Path to webmeta.jsonl (output of run_webmeta).")
    ap.add_argument("--out_dir", required=True, help="Output directory for CSV files.")
    ap.add_argument("--webmeta_csv_name", default="webmeta.csv")
    ap.add_argument("--post_table_name", default="post_table.csv")
    ap.add_argument("--media_manifest_name", default="media_manifest.csv")
    ap.add_argument("--post_map_name", default="post_map.csv")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    paths = jsonl_to_tables(
        webmeta_jsonl=args.webmeta_jsonl,
        out_dir=args.out_dir,
        webmeta_csv_name=args.webmeta_csv_name,
        post_table_name=args.post_table_name,
        media_manifest_name=args.media_manifest_name,
        post_map_name=args.post_map_name,
    )

    print("[OK] Generated:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
