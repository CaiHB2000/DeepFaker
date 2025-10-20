import argparse
from p2p.core.io import write_csv
from p2p.content.detectors.dfbench_csv import DFBenchCSV

def main():
    ap = argparse.ArgumentParser(description="Convert DeepFakeBench CSV to unified content schema")
    ap.add_argument("--in_csv", required=True, help="DeepFakeBench per-sample csv (image,label,prob)")
    ap.add_argument("--out_csv", required=True, help="Output csv path (image,content_score,label)")
    args = ap.parse_args()

    det = DFBenchCSV()
    rows = det.run(args.in_csv)
    write_csv(args.out_csv, rows, ["image", "content_score", "label"])
    print(f"[content] saved → {args.out_csv}  ({len(rows)} rows)")

if __name__ == "__main__":
    main()
