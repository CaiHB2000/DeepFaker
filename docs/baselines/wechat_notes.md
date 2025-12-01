WeChat / WeFEND Dataset Notes

- Dataset: WeChat (a.k.a. WeFEND) — Chinese public‑post news articles (headline + body; sometimes image/metadata) used in WeFEND (AAAI 2020).
- Year: collected around 2018–2019; first published with WeFEND (AAAI 2020; arXiv:1912.12520).
- Task: binary fake news detection (content‑level; no conversation trees). Primary metric commonly reported is Accuracy.
- Split: the WeFEND paper uses a chronological split (train/val/test by time). Keep this when comparing to WeFEND numbers.
- Baselines we catalog (accuracy on test):
  - WeFEND (with RL selector): 0.824 — arXiv:1912.12520 (AAAI 2020; Table 2 wording in paper; cross‑checked by later survey tables).
  - WeFEND‑ (ablated; without selector): 0.807 — AI Open 2022 survey Table 6 (WeChat column).
- Caveats:
  - Some follow‑up surveys list additional methods in a unified table across datasets; ensure that any number we cite on WeChat originates from an experiment actually run on the WeChat split, not from other Chinese rumor datasets (e.g., Weibo, MFND).
  - If re‑running baselines on WeChat, fix the chronological split and report Accuracy and Macro‑F1 (if class balance changes). Document tokenizers (Chinese BERT vs. multilingual BERT) and image backbones if multimodal.

References
- Wang et al., "Weak Supervision for Fake News Detection via Reinforcement Learning" (AAAI 2020). DOI: 10.1609/aaai.v34i01.5389. arXiv:1912.12520.
- Comprehensive survey tables referencing WeFEND’s WeChat results (e.g., AI Open 2022, Table 6).

