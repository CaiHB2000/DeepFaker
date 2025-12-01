import argparse
from p2p.core.io import write_csv, read_csv_as_dict
from p2p.policy.simple_threshold import SimpleThreshold

def main():
    ap = argparse.ArgumentParser(description="Apply simple threshold policy on P2P results")
    ap.add_argument("--p2p_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--hi",   type=float, default=0.8)
    ap.add_argument("--mid",  type=float, default=0.6)
    ap.add_argument("--hi_f", type=float, default=0.3)
    ap.add_argument("--mid_f",type=float, default=0.6)
    args = ap.parse_args()

    rows = read_csv_as_dict(args.p2p_csv)
    pol = SimpleThreshold(hi=args.hi, mid=args.mid, hi_f=args.hi_f, mid_f=args.mid_f)

    out = []
    for k, r in rows.items():
        res = pol.apply(r)
        out.append({
            "image": k,
            "p2p_risk": float(r["p2p_risk"]),
            **res
        })
    write_csv(args.out_csv, out, ["image","p2p_risk","policy","factor"])
    print(f"[policy] saved → {args.out_csv}  ({len(out)} rows)")

if __name__ == "__main__":
    main()
