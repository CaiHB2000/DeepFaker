# Propagation Tracker Guide

This repository now includes a lightweight toolkit for monitoring Reddit post engagement over time and folding those signals back into the provenance/content correlation workflow.

## 1. Prerequisites

- Reddit API credentials exported in the shell:
  ```bash
  export REDDIT_CLIENT_ID=xxxx
  export REDDIT_CLIENT_SECRET=yyyy
  export REDDIT_USER_AGENT="p2p-propagation-tracker/0.1"
  ```
- A seed list created by `p2p.runners.run_reddit_seed` (e.g. `tmp/reddit_seed_run_mass/reddit_posts.jsonl`).

## 2. One-off snapshots

For occasional polling (single run):
```bash
python -m p2p.tools.track_propagation \
  --posts-jsonl tmp/reddit_seed_run_mass/reddit_posts.jsonl \
  --out-csv tmp/reddit_seed_run_mass/propagation_timeseries.csv
```
Every invocation appends a new row per post with the latest score, upvote ratio, comment count, awards, etc.

## 3. Continuous watcher

To keep tracking in the background, start the watcher in a dedicated terminal:
```bash
python -m p2p.tools.watch_propagation \
  --posts-jsonl tmp/reddit_seed_run_mass/reddit_posts.jsonl \
  --out-csv tmp/reddit_seed_run_mass/propagation_timeseries.csv \
  --interval 600 \
  --jitter 30 \
  --refresh-targets
```

Key options:
- `--interval`: base sleep (seconds) between polls. Default 600 (10 min).
- `--jitter`: random ±jitter added to each sleep to desynchronise calls.
- `--refresh-targets`: re-read the input files before each poll (useful when new posts are appended to the list).
- `--max-iterations`: stop after N cycles; omit for indefinite run (Ctrl+C to exit).

The watcher prints a short log each iteration and writes into the same CSV as the one-off script, so both modes can coexist.

## 4. Feature refresh & correlation

After collecting fresh time-series snapshots, update the derived propagation features and run the combined analysis:
```bash
python -m p2p.tools.propagation_features \
  --posts-jsonl tmp/reddit_seed_run_mass/reddit_posts.jsonl \
  --out-csv tmp/reddit_seed_run_mass/propagation_features.csv

python -m p2p.tools.analyze_correlations \
  --posts-summary tmp/reddit_seed_run_mass/posts_summary_with_dfbench_multi.csv \
  --provenance tmp/reddit_seed_run_mass/provenance_media.csv \
  --inventory tmp/reddit_seed_run_mass/media_inventory.csv \
  --propagation tmp/reddit_seed_run_mass/propagation_features.csv \
  --out-dir tmp/reddit_seed_run_mass/analysis
```

Outputs land under `tmp/reddit_seed_run_mass/analysis/`, including updated correlation tables and plots that now reflect the propagation signals (`logistic_*`, `gompertz_*`, `hawkes_*`, etc.).

## 5. Automation tips

- Pair `watch_propagation` with `systemd --user` or `tmux` for unattended runs.
- Run `propagation_features` + `analyze_correlations` on a daily cron to refresh reports.
- If you curate new posts, add them to `reddit_posts.jsonl` (or supply `--ids-file`) and restart the watcher with `--refresh-targets` to include them automatically.
