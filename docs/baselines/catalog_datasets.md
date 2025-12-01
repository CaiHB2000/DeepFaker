# Catalog — Datasets for Multimodal Misinformation/Rumor Detection

- Weibo 2016/2018/2021 (binary rumor; text+image); event-consistent splits; Acc/Macro-F1 common.
- Twitter (KDD’18 binary rumor; text+image); Acc/Macro-F1 common.
- Fakeddit (LREC’20; 2/3/5/6‑way; multimodal; ensure same label granularity; Acc/Macro-F1).
- WeChat / WeFEND (binary; text+image; event-consistent training recommended; Acc/Macro‑F1/Pos‑F1/ECE).
- MFND (Weibo 3‑way; text+image+comments; used by MAGIC; Acc/Macro-F1).
- PHEME (Twitter conversation veracity; true/false/(unverified); conversation trees; Acc/Macro‑F1).
- Twitter15/16 (rumor; conversation graphs; Acc/Macro‑F1; used by GCAN/GACN).
- FakeNewsNet (PolitiFact/GossipCop; primarily text; some image links; if using multimodal, document preprocessing).
- Out-of-context (OOC) image-text benchmarks (e.g., COSMOS, NewsCLIPpings) — mismatch detection rather than veracity; keep separate to avoid task mixing.

> Action: For each dataset, record year/split/granularity, any social-graph requirement, standard metrics, and links to primary paper/code.
