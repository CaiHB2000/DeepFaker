#!/usr/bin/env python3
"""
Scan paper_results/*/wefend_teacher_match* summary.json files and emit:
  - tables/main_results.tex (method vs metrics)
  - tables/ablations.tex (teacher-match variants)
  - metrics_macros.tex (best scores as macros)

Usage:
  python papers/arr_rolling_review/scripts/gen_tables_from_results.py \
    --root paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2 \
    --out papers/arr_rolling_review
"""
import argparse
import json
from pathlib import Path


def load_summaries(root: Path):
    rows = []
    for p in root.rglob("summary.json"):
        name = p.parent.name
        try:
            obj = json.loads(p.read_text())
        except Exception:
            obj = None
        if not obj:
            # alternative schema
            try:
                obj = json.loads(p.read_text())
            except Exception:
                continue
        # tolerate different keys (test or test_metrics)
        metrics = obj.get("test_metrics") or obj.get("test") or obj.get("metrics") or {}
        acc = metrics.get("acc") or metrics.get("accuracy")
        mf = metrics.get("f1_macro") or metrics.get("macro_f1")
        pf = metrics.get("f1_pos") or metrics.get("pos_f1")
        ece = metrics.get("ece")
        if mf is None and "test" in obj:  # older schema
            t = obj["test"]
            acc = t.get("acc", acc)
            mf = t.get("f1", mf)
            pf = t.get("pos_f1", pf)
            ece = t.get("ece", ece)
        rows.append({
            "name": name,
            "acc": acc,
            "mf": mf,
            "pf": pf,
            "ece": ece,
        })
    return rows


def _tex_escape(s: str) -> str:
    if s is None:
        return ""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def apply_filters(rows, include=None, exclude=None):
    def match_any(name, patterns):
        return any(p in name for p in patterns)

    out = rows
    if include:
        out = [r for r in out if match_any(r["name"], include)]
    if exclude:
        out = [r for r in out if not match_any(r["name"], exclude)]
    return out


def _shorten(name: str) -> str:
    # Heuristic shortening for long run names
    n = name
    n = n.replace('wefend_teacher_match_', 'TMatch-')
    n = n.replace('wefend_dynamic_distill_', 'DMPD-')
    n = n.replace('wefend_longtrain_', 'DMPD-LT-')
    n = n.replace('_seed00', '')
    # Token-level compaction
    repl = {
        'longtrain': 'LT', 'eventreweight': 'ER', 'studentfocus': 'SF',
        'curriculum': 'Cur', 'dualstage': 'DS', 'tempminus': 'T-',
        'tempplus': 'T+', 'nomistake': 'NoMist', 'delta': 'Delta',
        'posfocus': 'PF', 'uncertainty': 'Unc', 'cost': 'Cost', 'late': 'Late',
        'pos_balanced': 'PosBal', 'dynamic': 'Dyn', 'temp': 'Temp'
    }
    for k, v in repl.items():
        n = n.replace(k, v)
    # Final cleanup: underscores -> hyphens
    n = n.replace('_', '-')
    return n


def tex_table(rows, caption, label, columns=("mf", "ece")):
    # Compose header with selected columns
    col_map = {"acc": "Acc", "mf": "Macro-F1", "pf": "Pos-F1", "ece": "ECE"}
    cols = [col_map[c] for c in columns]
    head = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}%\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        f"\\begin{{tabular}}{{l{''.join(['c' for _ in columns])}}}\\toprule\n"
        f"Method & {' & '.join(cols)} \\\\ \\midrule\n"
    )
    body = []
    for r in sorted(rows, key=lambda x: (-(x["mf"] or 0), x["ece"] or 9)):
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "--"
        name = _tex_escape(_shorten(r['name']))
        vals = []
        for c in columns:
            vals.append(fmt(r.get(c)))
        body.append(f"{name} & {' & '.join(vals)} \\\\ ")
    tail = (
        "\\bottomrule\n\\end{tabular}%\n}%% end resizebox\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    return head + "\n".join(body) + "\n" + tail


def write_macros(rows, out_dir: Path):
    best = max([r for r in rows if r["mf"] is not None], key=lambda r: r["mf"], default=None)
    out = out_dir / "metrics_macros.tex"
    if best:
        out.write_text(
            "\n".join([
                f"% auto-generated from results under {out_dir}",
                f"\\renewcommand{{\\bestMF}}{{{best['mf']:.4f}}}",
                f"\\renewcommand{{\\bestPF}}{{{(best['pf'] or 0):.4f}}}",
                f"\\renewcommand{{\\bestECE}}{{{(best['ece'] or 0):.4f}}}",
            ])
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--topk-main", type=int, default=0, help="limit rows in main table")
    ap.add_argument("--topk-abl", type=int, default=0, help="limit rows in ablation table")
    ap.add_argument("--include", nargs="*", default=[], help="only include rows containing any of these substrings")
    ap.add_argument("--exclude", nargs="*", default=[], help="exclude rows containing any of these substrings")
    args = ap.parse_args()

    rows = load_summaries(args.root)
    tables_dir = args.out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Main results: all teacher_match* entries if present, otherwise all.
    main_rows = [r for r in rows if r["name"].startswith("wefend_teacher_match_")] or rows
    main_rows = apply_filters(main_rows, include=args.include or None, exclude=args.exclude or None)
    main_rows = sorted(main_rows, key=lambda r: (-(r["mf"] or 0), r["ece"] or 9))
    if args.topk_main and args.topk_main > 0:
        main_rows = main_rows[: args.topk_main]
    (tables_dir / "main_results.tex").write_text(
        tex_table(main_rows, "Main results on WeFEND teacher-match variants.", "tab:main", columns=("mf","ece"))
    )

    # Ablations: group by simple keyword filters
    abl_rows = [r for r in rows if any(k in r["name"] for k in ["delta", "longtrain", "eventreweight", "temp", "dualstage"]) ]
    abl_rows = apply_filters(abl_rows, include=args.include or None, exclude=args.exclude or None)
    abl_rows = sorted(abl_rows, key=lambda r: (-(r["mf"] or 0), r["ece"] or 9))
    if args.topk_abl and args.topk_abl > 0:
        abl_rows = abl_rows[: args.topk_abl]
    (tables_dir / "ablations.tex").write_text(
        tex_table(abl_rows, "Ablations on gating, scheduling, and calibration.", "tab:abl", columns=("mf","ece"))
    )

    write_macros(rows, args.out)
    print(f"Wrote tables to {tables_dir} and macros to {args.out}/metrics_macros.tex")


if __name__ == "__main__":
    main()
