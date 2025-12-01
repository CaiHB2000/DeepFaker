# Project Brief — Writing / Planning Session (ARR submission)

## Scope and Positioning
- Setting: **supervised multimodal rumor/fake-news detection** with **explicit event IDs** and a **fixed multi-branch teacher** (text / image / fusion).
- Goal: selective, reliability-aware distillation to reduce propagation of teacher errors; **not** a general KD framework for unlabeled data or single-head VLMs.
- Hard cases: very weak teachers or no event structure (e.g., strict time-split Fakeddit) — treated as analysis only.

## Paper Status (main.pdf up to date)
- Figures: only the method diagram (Fig. \ref{fig:framework}) + an appendix bar chart (WeFEND Macro-F1/ECE). No reliability/gate/event plots in the main text.
- Tables: Table 1 (main results) and Table 2 (ablations) are accurate and self-contained.
- Text tightened: scope/assumptions moved upfront; contributions de‑amped; calibration framed as “does not worsen, sometimes improves”; Fakeddit marked as hard case.

## Key Results (current Table 1)
- **Weibo (zh)** — mean ± sd (seeds in parentheses)
  - Text-only 0.939/0.939/0.044 (3) ; SAFE 0.938/0.938/0.044 (3) ; EANN 0.937/0.936/0.050 (3)
  - Noisy-Student KD 0.941/0.941/0.056 (3)
  - **Noisy-Student KD** 0.941/0.941/0.056 (3)
  - Fusion teacher 0.943/0.943/0.036 (1)
  - **DMPD** 0.948/0.948/0.040 (3)
- **WeFEND (zh)** — mean ± sd
  - Text-only 0.927/0.893/0.063 (3) ; SAFE 0.937/0.905/0.040 (3) ; EANN 0.932/0.899/0.052 (3)
  - Noisy-Student KD 0.939/0.908/0.057 (3)
  - **Noisy-Student KD** 0.939/0.908/0.057 (3)
  - Fusion teacher 0.935/0.902/0.043 (1)
  - **DMPD** 0.939/0.913/0.030 (5)
- **MiRAGeNews (en, OOD test2–5 avg)**
  - Teacher 0.883/0.882/0.057 (3)
  - **Noisy-Student KD** 0.901/0.901/0.062 (3) (mean ± sd: Acc 0.9013 ± 0.0081, MF1 0.9009 ± 0.0081, ECE 0.0622 ± 0.0080)
  - **DMPD** 0.902/0.902/0.053 (3)

## Ablations (Table 2, one seed each)
- Weibo: full 0.9495/0.0323 ECE; –gate 0.9447/0.0367; –event 0.9474/0.0408; –evi 0.9440/0.0474.
- WeFEND: full 0.9436/0.9186/0.0378; –gate 0.9293/0.8938/0.0504; –event 0.9381/0.9088/0.0532; –evi 0.9304/0.8991/0.0485.

## Baselines (already in text/appendix)
- Simple-Conf-KD (per-instance confidence) — weaker than DMPD:
  - Weibo MF1 ≈ 0.942, ECE ≈ 0.043 (3 seeds)
  - WeFEND MF1 ≈ 0.905, ECE ≈ 0.053 (3 seeds)
- Selector ablations: confidence-based selector > always-text/always-vision on both datasets (numbers in appendix).

## Negative / Hard Cases
- Strict time-split Fakeddit: all content baselines low; DMPD does not beat teacher; declared as hard case in Limitations + Appendix.

## Analyses (now in appendix)
- Error cases (Weibo): teacher high-conf wrong vs student correct; teacher correct vs student overconfident wrong.
- Per-class calibration: Weibo student ECE (real,fake) ≈ (0.036, 0.036) vs teacher (0.040,0.040); WeFEND student (0.042,0.042) vs teacher (0.046,0.046); NLL Weibo 0.227/0.189 (std/tchr), WeFEND 0.230/0.222.
- WeFEND bar chart (Macro‑F1 / ECE) derived directly from Table 1; no other plots are used in the main text.

## Experiments in Flight (as of now)
- **MiRAGeNews Noisy-Student KD seeds 0/1/2** running on GPU6 (output dirs: `paper_results/miragenews_noisy_student_seed0{0,1,2}`). Replace the single-seed row once done.
- Weibo/WeFEND Noisy-Student KD (3 seeds each) **finished**; already in Table 1.

## Future Direction / Innovation Idea (for next revision)
- Elevate current heuristic gates into a **learnable reliability-aware selective KD**: model each teacher branch (text/img/fusion) as a noisy annotator with event-conditioned reliability. Use a learnable reliability module (per-branch, per-event) to weight teacher outputs, forming a posterior over latent labels; drive distillation by this posterior (entropy-based soft gate) instead of hard thresholds. This unifies instance gate + event EMA into a probabilistic selective KD view and can generalize beyond fixed thresholds or specific modalities.
- Optional upgrades:
  - Derive coverage–performance curves from this soft-gate posterior.
  - Show that randomizing event IDs (destroying event structure) harms performance/calibration, evidencing structured-noise modeling.
  - Connect to multi-annotator / noisy-label literature (e.g., Dawid–Skene) to position contribution as “latent-truth distillation under event-structured noise.”

## If time allows (optional next steps)
- Compute cross-modal disagreement vs. error (text vs image posteriors) and add a one-sentence finding in appendix.
- Run a “looser gate” coverage sanity run (τ_f/τ_m e.g., 0.90/0.85) to show gate coverage > 0; put curve in appendix (not needed for ARR if time is short).
- If MiRAGeNews Noisy-Student mean exceeds DMPD, adjust wording in Experiments/Limitations to acknowledge it (stress-test setting; DMPD still > teacher).

## Paths / Pointers
- Paper: `papers/arr_rolling_review/main.tex` (+ sections/*, tables/main_results.tex, tables/ablations.tex, figures/wefend_bar.pdf).
- Data sources: `papers/arr_rolling_review/DATA_SOURCES.md` (maps every table row to summary files).
- New KD configs: `dynamic_distill/configs/weibo_noisy_student.yaml`, `wefend_noisy_student.yaml`, `miragenews_noisy_student.yaml`.
- Running logs: `logs/weibo_noisy_student_seed*.log`, `logs/wefend_noisy_student_seed*.log`, `logs/miragenews_noisy_student_seed*.log`.
