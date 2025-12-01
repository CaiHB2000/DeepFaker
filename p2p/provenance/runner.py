import argparse
import json
import sys
from typing import List
from p2p.datasource.utils import read_lines, append_jsonl
from p2p.provenance.probes.c2pa_probe import analyze_one as c2pa_analyze
from p2p.provenance.fingerprint import fingerprints_for_path

def run_c2pa(args):
    paths: List[str] = []
    if args.path:
        paths.append(args.path)
    if args.paths_file:
        paths.extend(list(read_lines(args.paths_file)))
    if not paths:
        print("Use --path or --paths_file", file=sys.stderr)
        sys.exit(2)
    for p in paths:
        rec = c2pa_analyze(p)
        out = rec.__dict__
        if args.out:
            append_jsonl(args.out, out)
        else:
            print(json.dumps(out, ensure_ascii=False))

def run_fingerprint(args):
    paths: List[str] = []
    if args.path:
        paths.append(args.path)
    if args.paths_file:
        paths.extend(list(read_lines(args.paths_file)))
    if not paths:
        print("Use --path or --paths_file", file=sys.stderr)
        sys.exit(2)
    for p in paths:
        rec = fingerprints_for_path(p)
        out = {
            "path": rec.path,
            "pdq": rec.pdq.__dict__ if rec.pdq else None,
            "vpdq": {
                "path": rec.vpdq.path,
                "available": rec.vpdq.available,
                "frames": [f.__dict__ for f in rec.vpdq.frames],
                "error": rec.vpdq.error,
                "raw_json": rec.vpdq.raw_json
            } if rec.vpdq else None,
            "tmk_path": rec.tmk_path,
            "errors": rec.errors
        }
        if args.out:
            append_jsonl(args.out, out)
        else:
            print(json.dumps(out, ensure_ascii=False))

def main():
    ap = argparse.ArgumentParser(description="Provenance local probes")
    sub = ap.add_subparsers(dest="cmd")

    p1 = sub.add_parser("c2pa", help="Analyze C2PA/Content Credentials of local files")
    p1.add_argument("--path", type=str)
    p1.add_argument("--paths_file", type=str)
    p1.add_argument("--out", type=str)

    p2 = sub.add_parser("fingerprint", help="Compute PDQ/vPDQ/TMK fingerprints for local files")
    p2.add_argument("--path", type=str)
    p2.add_argument("--paths_file", type=str)
    p2.add_argument("--out", type=str)

    args = ap.parse_args()
    if args.cmd == "c2pa":
        run_c2pa(args)
    elif args.cmd == "fingerprint":
        run_fingerprint(args)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
