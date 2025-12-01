# Catalog — Methods for Multimodal Misinformation/Rumor Detection

> Scope: content-level and graph/conversation-level methods that use text and image and/or social context; 2018–2025 with emphasis on reproducible baselines used in Weibo/Twitter/Fakeddit/WeFEND/PHEME.

- EANN (Event-Adversarial Neural Networks), KDD 2018 — content-level, text+image; Weibo/Twitter splits; headline Acc/F1 reported in paper (filled in strong_baselines.md).
- TI-CNN (Text and Image CNN), 2018 — content-level early baseline; Weibo/Twitter; TODO/verify numbers from original.
- SAFE (Similarity-Aware Fusion), PAKDD 2020 — content-level, text+image; datasets vary; TODO/verify numbers on each dataset used in paper.
- SpotFake / SpotFake+ (2019–2020) — content-level BERT + CNN/VGG; reported on Twitter/Weibo variants; TODO/verify exact splits/metrics.
- MPFN (Multimodal Progressive Fusion Network), ~2019 — content-level; reports on Twitter; TODO/verify venue/metrics.
- AMFB (Attention-based Multimodal Factorized Bilinear), ~2019 — content-level; reported ~90.4% on Weibo per secondary mention; TODO confirm primary.
- MAGIC (Multimodal Adaptive Graph-based Intelligent Classification), arXiv 2024-11 — graph-enhanced multimodal; Fakeddit 2-way subset Acc≈97.6; MFND (Weibo 3-way) Acc≈85.96; note: not directly comparable to KDD’18 content-only.
- GCAN (Graph-aware Co-Attention), ACL 2020 — conversation graph on Twitter15/16; Acc≈0.8767/0.9084; F1≈0.8250/0.7593.
- DUCK (Detection with User and Comment Networks), 2021 — conversation graph; Twitter; TODO/verify metrics/splits.
- GACN (Graph Attention Capsule Network), 2021 — conversation propagation graphs; Twitter15/16 Acc≈0.889/0.900 per cited summaries; TODO/verify primary.
- Heterogeneous Graph w/ multimodal (Weibo 2016/2021), ~2021 — >92% Acc claimed; TODO/trace primary and fill exact numbers.
- CLIP/VLM-based detectors (2024–2025) — OOC/image-text mismatch and misinformation detection; include only when task equals veracity on the same datasets; TODO curate.

> Action: For each line, add per-dataset rows to method_results.csv with verified source (paper table/figure, page), split name, any special setting (graph vs content-only, multimodal-only subset, seed counts), and year.

