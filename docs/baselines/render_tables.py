#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import defaultdict

def load_rows(path):
    rows=[]
    with open(path, newline='', encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def fmt(v, default='—'):
    return v if (v and v.strip()) else default

def methods_x_datasets(rows):
    # For each method, collect headline per dataset (Acc/F1)
    by_method=defaultdict(list)
    for r in rows:
        by_method[r['method']].append(r)
    lines=["| Method | Venue (Year) | Dataset | Acc | Macro-F1 | Notes |",
           "|---|---:|---|---:|---:|---|",
    ]
    for m in sorted(by_method):
        for r in sorted(by_method[m], key=lambda x: (x['dataset'], x['granularity'])):
            lines.append("| {method} | {venue} {year} | {dataset} {gran} | {acc} | {mf} | {notes} |".format(
                method=r['short_name'] or r['method'],
                venue=fmt(r['venue']),year=fmt(r['year']),
                dataset=r['dataset'],gran=f"({r['granularity']})" if r['granularity'] else '',
                acc=fmt(r['metric_acc']), mf=fmt(r['metric_macro_f1']),
                notes=fmt(r['notes'])
            ))
    return "\n".join(lines)+"\n"

def datasets_x_methods(rows):
    by_dataset=defaultdict(list)
    for r in rows:
        by_dataset[(r['dataset'],r['granularity'])].append(r)
    lines=["| Dataset | Granularity | Method | Venue (Year) | Acc | Macro-F1 | Notes |",
           "|---|---|---|---:|---:|---:|---|",
    ]
    for (d,g) in sorted(by_dataset):
        for r in sorted(by_dataset[(d,g)], key=lambda x: (x['method'], x['year'])):
            lines.append("| {dataset} | {gran} | {method} | {venue} {year} | {acc} | {mf} | {notes} |".format(
                dataset=d, gran=g or '—', method=r['short_name'] or r['method'],
                venue=fmt(r['venue']), year=fmt(r['year']),
                acc=fmt(r['metric_acc']), mf=fmt(r['metric_macro_f1']), notes=fmt(r['notes'])
            ))
    return "\n".join(lines)+"\n"

def main():
    root=Path(__file__).resolve().parent
    rows=load_rows(root/"method_results.csv")
    (root/"methods_x_datasets.md").write_text(methods_x_datasets(rows),encoding='utf-8')
    (root/"datasets_x_methods.md").write_text(datasets_x_methods(rows),encoding='utf-8')
    print("Wrote:", root/"methods_x_datasets.md")
    print("Wrote:", root/"datasets_x_methods.md")

if __name__=='__main__':
    main()

