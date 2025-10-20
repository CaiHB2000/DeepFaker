import argparse
from p2p.core.io import write_csv, read_csv_as_dict
from p2p.propagation.simulators.ba_sim import BASimulator

def main():
    ap = argparse.ArgumentParser(description="Simulate early cascade risk from content+cred")
    ap.add_argument("--content_csv", required=True, help="CSV: image,content_score,label?")
    ap.add_argument("--prov_csv",    required=True, help="CSV: image,cred_score")
    ap.add_argument("--out_csv",     required=True, help="Output: image,early_risk")
    ap.add_argument("--n", type=int, default=300, help="Graph nodes")
    ap.add_argument("--steps", type=int, default=6, help="Early window steps")
    ap.add_argument("--base", type=float, default=0.03)
    ap.add_argument("--coef", type=float, default=0.30)
    args = ap.parse_args()

    content = read_csv_as_dict(args.content_csv, key="image")
    prov    = read_csv_as_dict(args.prov_csv,    key="image")

    items = []
    for k, r in content.items():
        items.append({
            "image": k,
            "content_score": float(r["content_score"]),
            "cred_score": float(prov.get(k, {}).get("cred_score", 0.5)),
        })
    sim = BASimulator(n=args.n, steps=args.steps, base=args.base, coef=args.coef)
    rows = sim.run(items)
    write_csv(args.out_csv, rows, ["image", "early_risk"])
    print(f"[prop] saved → {args.out_csv}  ({len(rows)} rows)")

if __name__ == "__main__":
    main()
