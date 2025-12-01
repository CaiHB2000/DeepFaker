This file documents where each numeric result in the ARR paper comes from in the project tree.
Paths are relative to the repository root (e.g., `/home/siyu/HDD/chb/DeepFaker`).

The main convention is:
- For each run: `paper_results/<dataset>/<run_name>/summary.json`
  - Use `.test_metrics.acc`, `.test_metrics.f1_macro`, `.test_metrics.ece`, etc.
- For MiRAGeNews OOD subsets: `paper_results/miragenews/<run_name>/test{2..5}_*.json`

All numbers in `tables/main_results.tex` and `tables/ablations.tex` are either single
`summary.json` values or simple averages across seeds as described below.

---

## 1. Weibo (zh, text+image)

### 1.1 Main baselines and DMPD (Table 1, first block)

- **Text-only BERT (3 seeds)**
  - `paper_results/weibo_text_only/weibo_text_only_seed00/summary.json`
  - `paper_results/weibo_text_only/weibo_text_only_seed01/summary.json`
  - `paper_results/weibo_text_only/weibo_text_only_seed02/summary.json`
  - Fields: `.test_metrics.acc`, `.test_metrics.f1_macro`, `.test_metrics.ece`
  - Table uses the mean over the three seeds; std is the sample std.

- **Image-only ViT (1 seed)**
  - `paper_results/weibo_image_only/weibo_image_only_seed00/summary.json`
  - Fields: `.test_metrics.acc`, `.test_metrics.f1_macro`, `.test_metrics.ece`

- **SAFE (PAKDD'20, 3 seeds)**
  - `paper_results/weibo_safe/weibo_safe_seed00/summary.json`
  - `paper_results/weibo_safe/weibo_safe_seed01/summary.json`
  - `paper_results/weibo_safe/weibo_safe_seed02/summary.json`
  - Fields: `.test_metrics.acc`, `.test_metrics.f1_macro`, `.test_metrics.ece`
  - Table values are mean ± std over these three seeds.

- **EANN (KDD'18, 3 seeds)**
  - `paper_results/weibo_eann/weibo_eann_seed00/summary.json`
  - `paper_results/weibo_eann/weibo_eann_seed01/summary.json`
  - `paper_results/weibo_eann/weibo_eann_seed02/summary.json`
  - Fields: `.test_metrics.acc`, `.test_metrics.f1_macro`, `.test_metrics.ece`

- **Teacher fusion baseline (1 seed)**
  - `paper_results/weibo_teacher_baseline/weibo_teacher_baseline_seed00/summary.json`
  - Fields: `.test_metrics.acc`, `.test_metrics.f1_macro`, `.test_metrics.ece`

- **DMPD (ours, 3 seeds)**
  - Current table uses the best student run as representative:
    - `paper_results/weibo_dynamic_distill/weibo_dynamic_distill_seed01/summary.json`
      - `.test_metrics.acc ≈ 0.9495`
      - `.test_metrics.f1_macro ≈ 0.9495`
      - `.test_metrics.ece ≈ 0.0323`
  - If needed, additional seeds live at:
    - `paper_results/weibo_dynamic_distill/weibo_dynamic_distill_seed00/summary.json`
    - `paper_results/weibo_dynamic_distill/weibo_dynamic_distill_seed02/summary.json`

### 1.2 Ablations on Weibo (Table 2)

All values are from single runs (one seed):

- **DMPD (full)**
  - `paper_results/weibo_dynamic_distill/weibo_dynamic_distill_seed01/summary.json`
  - `.test_metrics.acc = 0.949488... → 0.9495`
  - `.test_metrics.f1_macro = 0.949481... → 0.9495`
  - `.test_metrics.ece = 0.0323406... → 0.0323`

- **– gate**
  - `paper_results/weibo_ablation/weibo_nogate_seed00/summary.json`
  - `.test_metrics.acc = 0.9447099... → 0.9447`
  - `.test_metrics.f1_macro = 0.9446801... → 0.9447`
  - `.test_metrics.ece = 0.0367072... → 0.0367`

- **– event rel.**
  - `paper_results/weibo_ablation/weibo_noevent_seed00/summary.json`
  - `.test_metrics.acc = 0.9474403... → 0.9474`
  - `.test_metrics.f1_macro = 0.9474262... → 0.9474`
  - `.test_metrics.ece = 0.0407872... → 0.0408`

- **– evidential**
  - `paper_results/weibo_ablation/weibo_noevi_seed00/summary.json`
  - `.test_metrics.acc = 0.9440273... → 0.9440`
  - `.test_metrics.f1_macro = 0.9439876... → 0.9440`
  - `.test_metrics.ece = 0.0473742... → 0.0474`

---

## 2. WeFEND (zh, WeChat public accounts)

### 2.1 Main baselines and DMPD (Table 1, second block)

- **Text-only BERT (3 seeds)**
  - `paper_results/wefend_text_only/wefend_text_only_seed00/summary.json`
  - `paper_results/wefend_text_only/wefend_text_only_seed01/summary.json`
  - `paper_results/wefend_text_only/wefend_text_only_seed02/summary.json`
  - Fields: `.test_metrics.acc`, `.test_metrics.f1_macro`, `.test_metrics.ece`
  - Table: mean ± std over three seeds.

- **SAFE (text+image, 3 seeds)**
  - `paper_results/wefend_safe/wefend_safe_seed00/summary.json`
  - `paper_results/wefend_safe/wefend_safe_seed01/summary.json`
  - `paper_results/wefend_safe/wefend_safe_seed02/summary.json`

- **EANN (text+image, 3 seeds)**
  - `paper_results/wefend_eann/wefend_eann_seed00/summary.json`
  - `paper_results/wefend_eann/wefend_eann_seed01/summary.json`
  - `paper_results/wefend_eann/wefend_eann_seed02/summary.json`

- **Teacher fusion baseline (1 seed)**
  - `paper_results/wefend_teacher_baseline/wefend_teacher_baseline_seed00/summary.json`

- **DMPD (ours, 5 seeds aggregate; table row summarizes DMPD line)**
  - Individual seeds for the main soft-quota configuration:
    - `paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2/pseudolabel_dynamic_softquota_seed00/summary.json`
      - `.test_metrics.acc = 0.9436464`
      - `.test_metrics.f1_macro = 0.9186009`
      - `.test_metrics.ece = 0.0378296`
    - `.../pseudolabel_dynamic_softquota_seed01/summary.json`
    - `.../pseudolabel_dynamic_softquota_seed02/summary.json`
  - Additional seeds (if referenced in significance files) are under the same directory with seed indices 03/04.
  - Table uses mean ± std across all DMPD seeds used in the WeFEND summary.

### 2.2 Ablations on WeFEND (Table 2)

All from single-seed runs:

- **DMPD (full)**
  - `paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2/pseudolabel_dynamic_softquota_seed00/summary.json`
  - `.test_metrics.acc = 0.9436464... → 0.9436`
  - `.test_metrics.f1_macro = 0.9186009... → 0.9186`
  - `.test_metrics.ece = 0.0378296... → 0.0378`

- **– gate**
  - `paper_results/wefend_ablation/wefend_nogate_seed00/summary.json`
  - `.test_metrics.acc = 0.9292818... → 0.9293`
  - `.test_metrics.f1_macro = 0.8938214... → 0.8938`
  - `.test_metrics.ece = 0.0504099... → 0.0504`

- **– event rel.**
  - `paper_results/wefend_ablation/wefend_noevent_seed00/summary.json`
  - `.test_metrics.acc = 0.9381215... → 0.9381`
  - `.test_metrics.f1_macro = 0.9088253... → 0.9088`
  - `.test_metrics.ece = 0.0531936... → 0.0532`

- **– evidential**
  - `paper_results/wefend_ablation/wefend_noevi_seed00/summary.json`
  - `.test_metrics.acc = 0.9303867... → 0.9304`
  - `.test_metrics.f1_macro = 0.8990901... → 0.8991`
  - `.test_metrics.ece = 0.0485372... → 0.0485`

### 2.3 Selector ablations (Appendix text)

Values quoted in Appendix come from:

- `paper_results/wefend_selector_ablation/wefend_selector_confidence_seed00/summary.json`
- `paper_results/wefend_selector_ablation/wefend_selector_always_text_seed00/summary.json`
- `paper_results/wefend_selector_ablation/wefend_selector_always_vision_seed00/summary.json`

Fields: `.test_metrics.acc`, `.test_metrics.f1_macro`, `.test_metrics.f1_pos`, `.test_metrics.ece`.

---

## 3. MiRAGeNews (en, synthetic AI-generated news)

MiRAGeNews numbers in Table 1 are averages across OOD test subsets (test2–5) and seeds.

- **Teacher fusion (3 seeds; OOD test2–5)**
  - Seed 00:
    - `paper_results/miragenews/miragenews_teacher_seed00/test2_bbc_dalle_eval.json`
    - `paper_results/miragenews/miragenews_teacher_seed00/test3_cnn_dalle_eval.json`
    - `paper_results/miragenews/miragenews_teacher_seed00/test4_bbc_sdxl_eval.json`
    - `paper_results/miragenews/miragenews_teacher_seed00/test5_cnn_sdxl_eval.json`
  - Seed 01/02: analogous paths with `miragenews_teacher_seed01` and `miragenews_teacher_seed02`.
  - Each eval JSON has `acc`, `f1_macro`, `ece`.
  - For each seed: average metrics over test2–5; then average across the 3 seeds.

- **DMPD student (3 seeds; OOD test2–5)**
  - Seed 00:
    - `paper_results/miragenews/miragenews_student_seed00/test2_eval.json`
    - `paper_results/miragenews/miragenews_student_seed00/test3_eval.json`
    - `paper_results/miragenews/miragenews_student_seed00/test4_eval.json`
    - `paper_results/miragenews/miragenews_student_seed00/test5_eval.json`
  - Seed 01/02: `miragenews_student_seed01`, `miragenews_student_seed02` with the same suffixes.
  - Same averaging procedure as for the teacher.

The in-distribution test1 split is near-saturated for both teacher and student (values are in each run’s `summary.json` but not used in the main table).

---

## 4. Fakeddit strict time-split (Appendix, hard case)

All Fakeddit numbers mentioned in the appendix are summarized in:

- `paper_results/fakeddit_time_summary.md`

The underlying runs are:

- **Text-only / Image-only / EANN / teacher / student / SpotFake+**
  - Text-only:
    - `paper_results/fakeddit_time_baselines/fakeddit_time_text_only_seed*/summary.json`
  - Image-only:
    - `paper_results/fakeddit_time_baselines/fakeddit_time_image_only_seed*/summary.json`
  - EANN:
    - `paper_results/fakeddit_time_baselines/fakeddit_time_eann_seed*/summary.json`
  - Teacher (roberta-large + ViT-large):
    - `paper_results/fakeddit_time/fakeddit_time_teacher_seed00/summary.json`
    - `paper_results/fakeddit_time/fakeddit_time_teacher_seed01/summary.json`
  - Student (DMPD):
    - `paper_results/fakeddit_time_student/fakeddit_time_student_seed00/summary.json`
    - `paper_results/fakeddit_time_student/fakeddit_time_student_seed01/summary.json`
    - `paper_results/fakeddit_time_student/fakeddit_time_student_seed02/summary.json`
  - SpotFake+ (approximate reproduction):
    - `paper_results/fakeddit_time_spotfake/fakeddit_time_spotfake_seed00/summary.json`

The appendix text quotes the aggregate metrics from `fakeddit_time_summary.md`.

---

## 5. Calibration and significance

### 5.1 ECE in tables and text

All ECE values in the main and ablation tables are taken from:

- `.test_metrics.ece` in the corresponding `summary.json` files listed above.
  - Weibo: `weibo_*` runs.
  - WeFEND: `wefend_*` runs.
  - MiRAGeNews: per-split eval JSONs under `paper_results/miragenews`.

### 5.2 Bootstrap and McNemar significance

Significance results referenced in the paper (Weibo vs EANN, WeFEND vs teacher) are stored in:

- `experiments/shared/significance_weibo_wefend.json`
- `experiments/shared/significance_wefend_softquota.json`

These JSON files contain:
- Per-comparison bootstrap confidence intervals for ΔAcc and ΔMacro-F1.
- McNemar statistics (`b01`, `b10`, `p_value`).

---

## 6. Selector ablations (Weibo)

The Weibo selector ablation numbers mentioned in the appendix come from:

- `paper_results/weibo_selector_ablation/weibo_selector_confidence_seed00/summary.json`
- `paper_results/weibo_selector_ablation/weibo_selector_always_text_seed00/summary.json`
- `paper_results/weibo_selector_ablation/weibo_selector_always_vision_seed00/summary.json`

Fields: `.test_metrics.acc`, `.test_metrics.f1_macro`, `.test_metrics.f1_pos`, `.test_metrics.ece`.

---

## 7. Notes on gate coverage and event reliability

The current paper version does **not** include a gate-coverage or event-reliability figure in the main text. If future versions add such figures, the recommended data sources are:

- **Gate coverage over epochs**
  - From `train_history` in `summary.json`:
    - Weibo:
      - `paper_results/weibo_dynamic_distill/weibo_dynamic_distill_seed01/summary.json`
      - Use per-epoch fields `epoch` and `num_distill_pairs` (and optionally `num_fusion_pairs`).
    - WeFEND (trace run with CSV logs):
      - `paper_results/wefend_dynamic_distill_trace/wefend_dynamic_distill_trace_seed00_seed00/train_metrics.csv`
      - Columns: `epoch, num_distill_pairs, num_fusion_pairs, ...`

- **Event-level reliability distribution**
  - Not stored directly per event in the current logs.
  - To reconstruct, one can:
    - Use teacher predictions on the training set (e.g., from `*_teacher_*` prediction files),
    - Group by event ID and compute either:
      - The EMA-based reliability `r_e` per the paper definition, or
      - The per-event teacher accuracy as an approximation.
  - Any event-reliability histogram in future work should be derived from such a reconstruction script rather than from random placeholders.

