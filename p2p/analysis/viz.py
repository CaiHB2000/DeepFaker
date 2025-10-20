#!/usr/bin/env python3
import argparse, os, csv
import math
import matplotlib
matplotlib.use("Agg")  # 无交互后端
import matplotlib.pyplot as plt

def read_csv(path):
    rows=[]
    with open(path,"r") as f:
        it = csv.DictReader(f)
        for r in it:
            rows.append(r)
    return rows

def to_float(x, default=None):
    try:
        return float(x)
    except:
        return default

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def hist_plot(vals, title, out_path, bins=50):
    vals = [v for v in vals if v is not None]
    plt.figure()
    plt.hist(vals, bins=bins, alpha=0.85)
    plt.xlabel("value"); plt.ylabel("count"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def scatter_plot(x, y, c=None, title="", out_path="scatter.png"):
    xs=[a for a,b in zip(x,y) if a is not None and b is not None]
    ys=[b for a,b in zip(x,y) if a is not None and b is not None]
    cs=None
    if c is not None:
        cs=[ci for a,b,ci in zip(x,y,c) if a is not None and b is not None and ci is not None]
    plt.figure()
    if cs is None:
        plt.scatter(xs, ys, s=6, alpha=0.5)
    else:
        sc=plt.scatter(xs, ys, c=cs, s=6, alpha=0.5)
        cb=plt.colorbar(sc); cb.set_label("label")
    plt.xlabel("x"); plt.ylabel("y"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def roc_curve_simple(probs, labels):
    pairs=[(p,l) for p,l in zip(probs,labels) if l in (0,1) and p is not None]
    if not pairs: return None, None, None
    pairs.sort(key=lambda x: x[0], reverse=True)
    P=sum(l for _,l in pairs); N=len(pairs)-P
    if P==0 or N==0: return None, None, None
    tps=fps=0
    roc=[(0,0)]
    prev=None
    for p,l in pairs:
        if prev is None or p!=prev:
            roc.append((fps/N, tps/P))
            prev=p
        if l==1: tps+=1
        else: fps+=1
    roc.append((fps/N, tps/P))
    xs=[x for x,_ in roc]; ys=[y for _,y in roc]
    # AUC trapezoid
    auc=0.0
    for i in range(1,len(xs)):
        auc += (xs[i]-xs[i-1])*(ys[i]+ys[i-1])/2.0
    return xs, ys, auc

def main():
    ap = argparse.ArgumentParser(description="Visualize P2P CSV results.")
    ap.add_argument("--content_csv", required=True)
    ap.add_argument("--prov_csv",    required=True)
    ap.add_argument("--prop_csv",    required=True)
    ap.add_argument("--p2p_csv",     required=True)
    ap.add_argument("--policy_csv",  required=False, default=None)
    ap.add_argument("--out_dir",     required=False, default=None, help="Save figures to this dir (default: same dir as p2p_csv)")
    args = ap.parse_args()

    # Load data
    content = read_csv(args.content_csv)
    prov    = read_csv(args.prov_csv)
    prop    = read_csv(args.prop_csv)
    p2p     = read_csv(args.p2p_csv)
    policy  = read_csv(args.policy_csv) if (args.policy_csv and os.path.exists(args.policy_csv)) else []

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.p2p_csv)) or "."
    ensure_dir(out_dir)

    # Prepare arrays
    cs  = [to_float(r.get("content_score", None)) for r in content]
    ls  = [to_float(r.get("label", None)) for r in content]
    crs = [to_float(r.get("cred_score", None)) for r in prov]
    ers = [to_float(r.get("early_risk", None)) for r in prop]
    p2p_scores = [to_float(r.get("p2p_risk", None)) for r in p2p]

    # 1) 分布图
    hist_plot(cs,  "Content Score Distribution", os.path.join(out_dir, "fig_content_hist.png"))
    hist_plot(crs, "Credential Score Distribution", os.path.join(out_dir, "fig_cred_hist.png"))
    hist_plot(ers, "Early Risk Distribution", os.path.join(out_dir, "fig_early_hist.png"))
    hist_plot(p2p_scores, "P2P Risk Distribution", os.path.join(out_dir, "fig_p2p_hist.png"))

    # 2) content vs cred（按 label 着色，如果有）
    # 对齐键：根据 p2p 的 image 顺序来绘制
    p2p_map = {r["image"]: r for r in p2p}
    c_map = {r["image"]: r for r in content}
    pr_map= {r["image"]: r for r in prov}

    x=[]; y=[]; c=[]
    for k in p2p_map.keys():
        if k in c_map and k in pr_map:
            x.append(to_float(c_map[k].get("content_score", None)))
            y.append(to_float(pr_map[k].get("cred_score", None)))
            c.append(to_float(c_map[k].get("label", None)))
    scatter_plot(x, y, c=c, title="Content vs Credential (colored by label if present)",
                 out_path=os.path.join(out_dir, "fig_scatter_content_vs_cred.png"))

    # 3) ROC 曲线（若 content 有 label）
    if any(v in (0,1) for v in ls):
        xs, ys, auc = roc_curve_simple(cs, [int(v) if v in (0,1) else None for v in ls])
        if xs is not None:
            plt.figure()
            plt.plot(xs, ys, label=f"Content ROC (AUC={auc:.3f})")
            plt.plot([0,1],[0,1],'--')
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - Content")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_content_roc.png")); plt.close()

    # 4) 策略分布（若有 policy）
    if policy:
        from collections import Counter
        pols = [r.get("policy","none") for r in policy]
        c = Counter(pols)
        labels=list(c.keys()); vals=[c[k] for k in labels]
        plt.figure()
        plt.bar(labels, vals)
        plt.title("Policy Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_policy_bar.png")); plt.close()

    print(f"[viz] saved figures to: {out_dir}")

if __name__ == "__main__":
    main()
