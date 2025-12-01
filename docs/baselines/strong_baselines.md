# Strong Baselines (Literature) — Multimodal Misinformation Detection

This note consolidates methods and datasets we cite, with dataset years, reported metrics, and any special settings that affect comparability. It serves as a standing “strong baseline” reference for our experiments.

Status (2025-11-12): initial pass with verified citations and a subset of metrics. Rows marked “TBD/verify” need number confirmation from the primary paper or code release.

## Coverage Snapshot
- Methods in refs: EANN (event-invariant, KDD 2018), SAFE (similarity-aware fusion, PAKDD 2020), GCAN (graph co-attention, ACL 2020), FANG (graph + social, CIKM 2020), baselines on Fakeddit (LREC 2020). KD/uncertainty/cali papers are methodological (not direct FND baselines).
- Datasets in scope: Weibo/Twitter (KDD 2018), Fakeddit (LREC 2020), WeChat/WeFEND (this repo), and PHEME (rumor veracity on Twitter conversations).
- Gaps to consider adding: recent LVLM/VLM-based detectors (2024–2025), out-of-context and cross-evidence checks, MMFakeBench-style evaluations.

## Datasets (year, split protocol)
- Weibo (KDD 2018, Wang et al.): event-consistent rumor/fake posts with images; common metrics: Accuracy, Macro-F1.
- Twitter (KDD 2018, Wang et al.): paired text-image rumor set; metrics: Accuracy, Macro-F1.
- Fakeddit (LREC 2020, Nakamura et al.): 2/3/5-way labels; multimodal; metrics often Accuracy / Macro-F1; ensure same label granularity.
- WeChat / WeFEND (internal in this repo): Chinese WeChat public-posts multimodal set; binary; we report Acc/Macro‑F1/Pos‑F1/ECE; event‑consistent training recommended.
- PHEME (approx. 2016 initial release; rumor veracity on Twitter conversation trees): labels vary by release (true/false/unverified), with conversation‑level structure; many methods use text+structure and evaluate Accuracy/Macro‑F1. Verify exact split/year for the chosen release.

## Table A — Methods × Datasets (headline metrics)

| Method | Venue (Year) | Weibo Acc | Weibo Macro-F1 | Twitter Acc | Twitter Macro-F1 | Fakeddit Acc/M-F1 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| EANN | KDD 2018 | 0.827 | 0.829 | 0.715 | 0.719 | — | Weibo/Twitter (content-level). Values from Table 2 of the paper. |
| SAFE | PAKDD 2020 | TBD | TBD | TBD | TBD | — | Similarity-aware fusion; numbers to extract from paper tables (not Weibo/Twitter in original). |
| GCAN | ACL 2020 | — | — | 0.877 | 0.825 | — | Twitter15/16 content+user features (retweet/user profiles), Acc: 0.8767 (T15), 0.9084 (T16); F1: 0.8250 (T15), 0.7593 (T16). |
| MAGIC (graph-based) | arXiv 2024 | — | — | — | — | 0.976 (Acc, Fakeddit 2‑way subset) | Uses custom Fakeddit (n≈3.1k) and MFND (Weibo 3‑way); not directly comparable to full Fakeddit or KDD’18 splits. |
| FANG | CIKM 2020 | — | — | — | — | — | New dataset; metrics not directly comparable; list separately below. |
| Our DMPD (student) | — | TBD | TBD | TBD | TBD | TBD | Keep for later comparison once experiments finalize. |

Sources: EANN metrics on Weibo/Twitter as reported in KDD’18 and survey summaries; SAFE/GCAN numbers to be pulled from their tables; Fakeddit baselines per LREC’20.

## Table B — Datasets × Methods (reverse view)

| Dataset | Year | Label Granularity | Methods Reported | Best Macro-F1 | Best Acc | Notes |
|---|---:|---|---|---:|---:|---|
| Weibo (KDD’18 split) | 2018 | binary | EANN, SAFE (TBD), GCAN (if evaluated) | 0.829 (EANN F1) | 0.827 (EANN Acc) | Ensure event-consistent split; verify whether methods use extra social graph. |
| Twitter (KDD’18 split) | 2018 | binary | EANN, SAFE (TBD), GCAN (separate Twitter15/16) | 0.719 (EANN F1) | 0.715 (EANN Acc) | GCAN is on Twitter15/16 with Acc≈0.877/0.908; task setup differs (conversation-based). |
| Fakeddit (2-way) | 2020 | 2-way | Paper baselines (multimodal only) incl. BERT+ResNet50 (max) | — | 0.8909 (Acc, test) | Multimodal-only subset; combination “maximum” reported best. |
| Fakeddit (3-way) | 2020 | 3-way | Paper baselines (multimodal only) incl. BERT+ResNet50 (max) | — | 0.8890 (Acc, test) | Same setting as above. |
| Fakeddit (6-way) | 2020 | 6-way | Paper baselines (multimodal only) incl. BERT+ResNet50 (max) | — | 0.8588 (Acc, test) | Same setting as above. |

## Table C — WeChat / WeFEND (binary)

| Method | Venue (Year) | Macro‑F1 | Pos‑F1 | Acc | ECE | Notes |
|---|---:|---:|---:|---:|---:|---|
| EANN (adapted) | KDD 2018 | TBD | TBD | TBD | — | If used, ensure image pipeline and event-consistent split.
| SAFE (adapted) | PAKDD 2020 | TBD | TBD | TBD | — | Similarity-aware fusion; verify preprocessing.
| GCAN (content-only variant) | ACL 2020 | TBD | TBD | TBD | — | Report separately if graph is unavailable.
| DMPD (ours, student) | — | TBD | TBD | TBD | TBD | Our main target; report mean ± s.e.m.

## Table D — PHEME (rumor veracity on Twitter conversations)

| Method | Venue (Year) | Accuracy | Macro‑F1 | Split | Notes |
|---|---:|---:|---:|---|---|
| GCAN (conversation graph) | ACL 2020 | TBD | TBD | official | Uses graph; not directly comparable to content-only baselines.
| EANN/SAFE (text+image only) | KDD’18/PAKDD’20 | TBD | TBD | adapted | If adapted, document whether tree context ignored.
| Recent LVLM/VLM (if any) | 2024–2025 | TBD | TBD | varied | Track only if task matches veracity (not stance).

## Special Settings (affecting comparability)
- Social Graph vs. Content Only: GCAN/FANG leverage user/repost graphs; EANN/SAFE/DMPD are content-level; report separately or note when graph is used.
- Label Granularity: Fakeddit has 2/3/5-way; do not mix across tables; convert to a common binary only with explicit mapping.
- Pretrained Backbones: Base vs. Large encoders (e.g., BERT-base vs. -large) change scores; note sizes.
- Multi-seed Averages: Where papers report single-run results, indicate; our tables prefer mean ± s.e.m.

## References
- EANN (KDD 2018): Wang et al., Event Adversarial Neural Networks for Multi-Modal Fake News Detection. KDD’18.
- SAFE (PAKDD 2020): Zhou et al., Similarity-Aware Multi-Modal Fake News Detection. PAKDD’20.
- GCAN (ACL 2020): Lu & Li, Graph-aware Co-Attention Networks for Explainable Fake News Detection. ACL’20.
- FANG (CIKM 2020): Nguyen et al., FANG: Leveraging Social Context for Fake News Detection Using Graph Representation. CIKM’20.
- Fakeddit (LREC 2020): Nakamura et al., Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection. LREC’20.

## TODO (verification & expansion)
- Extract SAFE/GCAN numerical results for Weibo/Twitter (Accuracy/Macro-F1) from the primary PDFs and fill Table A.
- Add Fakeddit (2-way, 3-way, 5-way) baselines (BOW, BERT, ResNet/ViT, fusion) with exact splits from LREC’20.
- WeChat / WeFEND: backfill content‑only baselines (EANN/SAFE) and our DMPD student; record event‑consistent split details.
- PHEME: fill conversation‑graph baselines (GCAN and successors) and any content‑only variants; ensure the same veracity task and split.
- Add 2024–2025 LVLM/VLM detectors and recent out‑of‑context benchmarks as a separate table, to avoid mixing task definitions.
