import argparse
from p2p.core.io import write_csv, read_csv_as_dict
from p2p.aggregator.p2p_sigmoid import P2PSigmoid


def main():
    ap = argparse.ArgumentParser(description="Aggregate content/cred/early into P2P risk")
    ap.add_argument("--content_csv", required=True)
    ap.add_argument("--prov_csv", required=True)
    ap.add_argument("--prop_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--alpha", type=float, default=0.33)
    ap.add_argument("--beta", type=float, default=0.33)
    ap.add_argument("--gamma", type=float, default=0.34)
    ap.add_argument("--k", type=float, default=5.0, help="sigmoid sharpness")
    ap.add_argument("--rescale", choices=["none", "linear", "quantile"], default="none")
    ap.add_argument("--expand_lo", type=float, default=None)
    ap.add_argument("--expand_hi", type=float, default=None)

    # 新增参数：支持 quantile 分段拉伸
    ap.add_argument("--q_low", type=float, default=0.20)
    ap.add_argument("--q_high", type=float, default=0.80)
    ap.add_argument("--tgt_low_lo", type=float, default=0.05)
    ap.add_argument("--tgt_low_hi", type=float, default=0.35)
    ap.add_argument("--tgt_mid_lo", type=float, default=0.35)
    ap.add_argument("--tgt_mid_hi", type=float, default=0.75)
    ap.add_argument("--tgt_high_lo", type=float, default=0.75)
    ap.add_argument("--tgt_high_hi", type=float, default=0.98)

    args = ap.parse_args()

    content = read_csv_as_dict(args.content_csv)
    prov = read_csv_as_dict(args.prov_csv)
    prop = read_csv_as_dict(args.prop_csv)

    expand_to = (args.expand_lo, args.expand_hi) if (
                args.expand_lo is not None and args.expand_hi is not None) else None
    tgt_low = (args.tgt_low_lo, args.tgt_low_hi)
    tgt_mid = (args.tgt_mid_lo, args.tgt_mid_hi)
    tgt_high = (args.tgt_high_lo, args.tgt_high_hi)

    # 根据传入的参数配置初始化 P2PSigmoid
    aggr = P2PSigmoid(alpha=args.alpha, beta=args.beta, gamma=args.gamma, k=args.k,
                      rescale=args.rescale, expand_to=expand_to,
                      q_low=args.q_low, q_high=args.q_high,
                      tgt_low=tgt_low, tgt_mid=tgt_mid, tgt_high=tgt_high)

    rows_in = []
    for key, r in content.items():
        rows_in.append({
            "image": key,
            "label": int(r.get("label", -1)),
            "content_score": float(r["content_score"]),
            "cred_score": float(prov.get(key, {}).get("cred_score", 0.5)),
            "early_risk": float(prop.get(key, {}).get("early_risk", 0.5)),
        })

    rows_out = aggr.run(rows_in)
    # ✅ 多落一列 raw
    write_csv(args.out_csv, rows_out,
              ["image", "label", "content_score", "cred_score", "early_risk", "p2p_risk", "p2p_risk_raw"])
    print(f"[aggr] saved → {args.out_csv}  ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()