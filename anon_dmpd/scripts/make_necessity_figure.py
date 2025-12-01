#!/usr/bin/env python3
"""
Build the motivation figure (Figure 1) from teachers'/student predictions.

Outputs: figures/necessity_overview.pdf with 3 panels (A/B/C):
 - (A) Histogram of event-level teacher reliability r_e (accuracy EMA or per-event accuracy)
 - (B) Teacher confidence vs. error rate (bin plot; highlights overconfident mistakes)
 - (C) Modality disagreement (e.g., |p_t(pos)-p_v(pos)|) vs. error rate (bin plot)

Assumptions:
 - You can produce CSVs with columns: id,event,y_true,
   p_t_pos,p_v_pos,p_f_pos,y_pred_t,y_pred_v,y_pred_f (teacher posteriors/labels),
   y_pred_s (optional student label/prob).
 - If you don't have teacher CSVs, run the teacher eval scripts to dump per-sample predictions first.

Usage:
  python papers/arr_rolling_review/scripts/make_necessity_figure.py \
    --pred teachers_test.csv --out papers/arr_rolling_review/figures
"""
import argparse
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def bin_curve(x, err, nbins=12):
    bins = np.linspace(0, 1, nbins + 1)
    idx = np.digitize(x, bins) - 1
    xs, ys = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < 5:
            continue
        xs.append((bins[b] + bins[b + 1]) / 2)
        ys.append(err[m].mean())
    return np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=Path, required=True, help='CSV with per-sample predictions')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--nbins', type=int, default=12)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.pred)

    # (A) event reliability: per-event teacher-fusion accuracy
    df['correct_f'] = (df['y_pred_f'] == df['y_true']).astype(int)
    evt = df.groupby('event', as_index=False)['correct_f'].mean().rename(columns={'correct_f':'acc_evt'})

    # (B) confidence vs error (teacher fusion)
    df['conf_f'] = df['p_f_pos'].clip(0, 1)
    df['err_f'] = 1 - df['correct_f']
    xb, yb = bin_curve(df['conf_f'].values, df['err_f'].values, nbins=args.nbins)

    # (C) modality disagreement vs error
    df['disagree_tv'] = (df['p_t_pos'] - df['p_v_pos']).abs().clip(0, 1)
    xc, yc = bin_curve(df['disagree_tv'].values, df['err_f'].values, nbins=args.nbins)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.6))

    axes[0].hist(evt['acc_evt'].values, bins=20, color='#4e79a7')
    axes[0].set_title('(A) Event reliability')
    axes[0].set_xlabel('Per-event teacher accuracy')
    axes[0].set_ylabel('Count')

    axes[1].plot(xb, yb, '-o', color='#f28e2b', markersize=3)
    axes[1].set_title('(B) Confidence vs. error')
    axes[1].set_xlabel('Teacher confidence (fusion)')
    axes[1].set_ylabel('Error rate')
    axes[1].set_ylim(0, 1)

    axes[2].plot(xc, yc, '-o', color='#59a14f', markersize=3)
    axes[2].set_title('(C) Modality disagreement vs. error')
    axes[2].set_xlabel('|p_t(pos)-p_v(pos)|')
    axes[2].set_ylabel('Error rate')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    outp = args.out / 'necessity_overview.pdf'
    fig.savefig(outp)
    print('Wrote', outp)


if __name__ == '__main__':
    main()

