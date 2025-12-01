#!/usr/bin/env python3
import argparse, os, csv, json
from collections import Counter
import math

def read_csv(path, key="image"):
    rows = []
    with open(path, "r") as f:
        it = csv.DictReader(f)
        fields = it.fieldnames or []
        for r in it:
            rows.append(r)
    return rows

def must_have_cols(name, rows, cols):
    if not rows:
        raise ValueError(f"[{name}] CSV is empty.")
    miss = [c for c in cols if c not in rows[0].keys()]
    if miss:
        raise ValueError(f"[{name}] missing columns: {miss}")

def check_range01(name, rows, col, allow_nan=False):
    bad = []
    nan_cnt = 0
    for r in rows:
        v = r.get(col, None)
        if v in (None, ''):
            nan_cnt += 1
            if not allow_nan:
                bad.append(("missing", v))
            continue
        try:
            f = float(v)
        except Exception:
            bad.append(("not_float", v))
            continue
        if math.isnan(f) or math.isinf(f):
            nan_cnt += 1
            if not allow_nan:
                bad.append(("nan_or_inf", v))
        elif f < 0.0 or f > 1.0:
            bad.append(("out_of_range", f))
    return {"col": col, "total": len(rows), "nan": nan_cnt, "bad": bad[:10], "bad_count": len(bad)}

def check_unique_key(name, rows, key="image"):
    keys = [r.get(key,"") for r in rows]
    c = Counter(keys)
    dups = [k for k, v in c.items() if v > 1]
    return {"unique": len(dups)==0, "dup_count": len(dups), "dup_examples": dups[:5]}

def join_coverage(content, other, other_name, key="image"):
    set_c = {r[key] for r in content}
    set_o = {r[key] for r in other}
    only_c = list(set_c - set_o)
    only_o = list(set_o - set_c)
    cov = 1.0 - len(only_c)/max(1,len(set_c))
    return {
        "base_count": len(set_c),
        "other_count": len(set_o),
        "coverage_on_base": cov,
        "missing_in_other": only_c[:10],
        "extra_in_other": only_o[:10]
    }

def safe_float(v, default=None):
    try:
        return float(v)
    except:
        return default

def auc_binary(probs, labels):
    # Minimal AUROC (no sklearn dependency)
    paired = [(p, y) for p,y in zip(probs, labels) if y in (0,1)]
    if not paired:
        return None
    paired.sort(key=lambda x: x[0], reverse=True)
    tp = fp = 0
    P = sum(y for _,y in paired)
    N = len(paired) - P
    if P==0 or N==0:
        return None
    prevp = None
    area = 0.0
    last_fp = last_tp = 0
    for p,y in paired:
        if prevp is None or p != prevp:
            area += (fp - last_fp) * (tp + last_tp) / 2.0
            last_fp, last_tp = fp, tp
            prevp = p
        if y == 1: tp += 1
        else: fp += 1
    area += (fp - last_fp) * (tp + last_tp) / 2.0
    return area / (P * N)

def main():
    ap = argparse.ArgumentParser(description="Validate P2P CSV artifacts.")
    ap.add_argument("--content_csv", required=True)
    ap.add_argument("--prov_csv",    required=True)
    ap.add_argument("--prop_csv",    required=True)
    ap.add_argument("--p2p_csv",     required=True)
    ap.add_argument("--report_json", required=False, default=None)
    args = ap.parse_args()

    content = read_csv(args.content_csv)
    prov    = read_csv(args.prov_csv)
    prop    = read_csv(args.prop_csv)
    p2p     = read_csv(args.p2p_csv)

    # Schema checks
    must_have_cols("content", content, ["image","content_score"])
    must_have_cols("prov",    prov,    ["image","cred_score"])
    must_have_cols("prop",    prop,    ["image","early_risk"])
    must_have_cols("p2p",     p2p,     ["image","p2p_risk","content_score","cred_score","early_risk"])

    # Range checks
    r_content = check_range01("content", content, "content_score")
    r_cred    = check_range01("prov",    prov,    "cred_score")
    r_early   = check_range01("prop",    prop,    "early_risk")
    r_p2p     = check_range01("p2p",     p2p,     "p2p_risk")

    # Unique key checks
    u_content = check_unique_key("content", content)
    u_prov    = check_unique_key("prov",    prov)
    u_prop    = check_unique_key("prop",    prop)
    u_p2p     = check_unique_key("p2p",     p2p)

    # Join coverage
    j_content_prov = join_coverage(content, prov, "prov")
    j_content_prop = join_coverage(content, prop, "prop")
    j_content_p2p  = join_coverage(content, p2p,  "p2p")

    # Optional AUROC if label in content/p2p
    labels_c = [safe_float(r.get("label", None), None) for r in content]
    probs_c  = [safe_float(r.get("content_score", None), None) for r in content]
    if all(x in (0,1,None) for x in labels_c) and any(x in (0,1) for x in labels_c):
        auc_c = auc_binary([p for p,l in zip(probs_c,labels_c) if l in (0,1)],
                           [int(l) for l in labels_c if l in (0,1)])
    else:
        auc_c = None

    labels_p = [safe_float(r.get("label", None), None) for r in p2p]
    probs_p  = [safe_float(r.get("p2p_risk", None), None) for r in p2p]
    if all(x in (0,1,None) for x in labels_p) and any(x in (0,1) for x in labels_p):
        auc_p = auc_binary([p for p,l in zip(probs_p,labels_p) if l in (0,1)],
                           [int(l) for l in labels_p if l in (0,1)])
    else:
        auc_p = None

    report = {
        "counts": {
            "content": len(content),
            "prov":    len(prov),
            "prop":    len(prop),
            "p2p":     len(p2p),
        },
        "range_checks": {
            "content_score": r_content,
            "cred_score":    r_cred,
            "early_risk":    r_early,
            "p2p_risk":      r_p2p,
        },
        "unique_key": {
            "content": u_content,
            "prov":    u_prov,
            "prop":    u_prop,
            "p2p":     u_p2p,
        },
        "join_coverage": {
            "content_vs_prov": j_content_prov,
            "content_vs_prop": j_content_prop,
            "content_vs_p2p":  j_content_p2p,
        },
        "metrics": {
            "content_auc": auc_c,
            "p2p_auc":     auc_p,
        }
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
