# p2p/runners/run_provenance.py
import argparse, inspect
from p2p.core.io import write_csv, read_csv_as_dict
from p2p.core.registries import PROV_REG
import p2p.provenance.probes  # <<< 关键：触发自动导入与注册


def build_probe(probe_name: str, **kwargs):
    if probe_name not in PROV_REG:
        raise ValueError(f"Unknown probe '{probe_name}'. Available: {list(PROV_REG.keys())}")
    cls = PROV_REG[probe_name]
    sig = inspect.signature(cls.__init__)
    # 过滤出 __init__ 支持的参数
    valid_kwargs = {k:v for k,v in kwargs.items() if k in sig.parameters}
    return cls(**valid_kwargs)

def main():
    ap = argparse.ArgumentParser(description="Compute provenance cred_score from various probes")
    ap.add_argument("--probe", default="phash_stability", help="Probe name registered in PROV_REG")
    ap.add_argument("--items_csv", required=True, help="CSV with 'image' column")
    ap.add_argument("--lmdb_root", required=True)
    ap.add_argument("--out_csv", required=True)

    # 通用可能用到的参数（探针按需接收）
    ap.add_argument("--jpeg_q", type=float, default=70)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--crop_ratio", type=float, default=0.04)
    ap.add_argument("--blur_sigma", type=float, default=0.8)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--build_index", type=int, default=1)
    ap.add_argument("--rgb_fallback", type=int, default=1)
    ap.add_argument("--thresh", type=float, default=30.0)  # 给 ELA 用
    args = ap.parse_args()

    items_map = read_csv_as_dict(args.items_csv, key="image")
    probe = build_probe(
        args.probe,
        lmdb_root=args.lmdb_root,
        jpeg_q=args.jpeg_q,
        trials=args.trials,
        crop_ratio=args.crop_ratio,
        blur_sigma=args.blur_sigma,
        jitter=args.jitter,
        build_index=bool(args.build_index),
        rgb_fallback=bool(args.rgb_fallback),
        thresh=args.thresh,
    )
    rows = probe.run(items_map.values())
    write_csv(args.out_csv, rows, ["image","cred_score"])
    print(f"[prov:{args.probe}] saved → {args.out_csv}  ({len(rows)} rows)")

if __name__ == "__main__":
    main()
