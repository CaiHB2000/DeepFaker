# Project Brief — Engineering / Experiments Session

## Environment
- Codebase root: `/home/siyu/HDD/chb/DeepFaker`
- Training entry: `dynamic_distill/scripts/train_mvp.py` (`--config`, `--seed`, `--output-dir`)
- Eval entry: `dynamic_distill/scripts/evaluate_model.py` (supports `--test-csv` override for MiRAGeNews splits)
- GPUs: 7×A100 80GB, currently all free when no jobs are running.
- Data locations:
  - Weibo: `datasets/weibo` (uses pickle id files + tweets/*.txt)
  - WeFEND: `datasets/wechat/processed_split/wefend_{train,val,test}.csv`, images in `datasets/wechat/images/`
  - MiRAGeNews: `datasets/miragenews/processed/{miragenews_train.csv, miragenews_validation.csv, miragenews_test*.csv}`
  - Fakeddit (strict time split): `datasets/fakeddit/processed_strict/...` (not active now)

## Current key results (for reference)
- DMPD (main paper numbers): Weibo MF1 ≈ 0.948 / ECE 0.040; WeFEND MF1 ≈ 0.913 / ECE 0.030; MiRAGeNews OOD avg MF1 ≈ 0.902 / ECE 0.053.
- Simple-Conf-KD: Weibo MF1 ≈ 0.942 / ECE ≈ 0.043; WeFEND MF1 ≈ 0.905 / ECE ≈ 0.053.
- Noisy-Student KD (full KD, no gate):
  - Weibo mean (3 seeds): Acc 0.9408 / MF1 0.9408 / ECE 0.0555.
  - WeFEND mean (3 seeds): Acc 0.9389 / MF1 0.9079 / ECE 0.0570.
  - MiRAGeNews (3 seeds finished): OOD avg Acc 0.9013 ± 0.0081, MF1 0.9009 ± 0.0081, ECE 0.0622 ± 0.0080 (per-seed OOD avg: seed00 0.9060/0.9055/0.0607; seed01 0.9080/0.9076/0.0532; seed02 0.8900/0.8894/0.0725).

## Strong-teacher runs (single seed, latest)
- Weibo strong teacher (hfl/chinese-roberta-wwm-ext-large + ViT-large): Acc 0.9488 / MF1 0.9488 / ECE 0.0324 (`paper_results/weibo_teacher_vitlarge_seed00/model_best.pt`).
- WeFEND strong teacher (same backbone): Acc 0.9403 / MF1 0.9121 / Pos-F1 0.8622 / ECE 0.0396 (`paper_results/wefend_teacher_vitlarge_seed00/model_best.pt`).
- MiRAGeNews strong teacher (roberta-large + ViT-large): test1 Acc/MF1 0.9980 / ECE 0.0040; OOD test2–5 mean Acc 0.8760 / MF1 0.8741 / ECE 0.0524 (`paper_results/miragenews_teacher_vitlarge_seed00/test{2..5}_eval.json`).
- Fakeddit 2-way time RaDMPD student (old teacher): Acc 0.5055 / MF1 0.5041 / ECE 0.3726 (`paper_results/fakeddit_radmpd_time_seed00/test_eval.json`).
- Fakeddit 6-way time RaDMPD student (old teacher): Acc 0.4824 / MF1 0.5621 / Pos-F1 0.7204 / ECE 0.3422 (`paper_results/fakeddit6_radmpd_time_seed00/test_eval.json`).

## Jobs in flight
None on GPU6 (MiRAGeNews Noisy-Student seeds 0/1/2 finished). GPUs 0–5 also idle if earlier KD runs have completed.

## Completed baselines / configs
- DMPD main:
  - Weibo: `dynamic_distill/configs/weibo_dynamic_distill.yaml`, outputs `paper_results/weibo_dynamic_distill/weibo_dynamic_distill_seed0{0,1,2}`
  - WeFEND: `dynamic_distill/configs/wefend_dynamic_distill_teacher_wcls_selective_v2.yaml`, outputs `paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2/pseudolabel_dynamic_softquota_seed0{0,1,2}`
- Noisy-Student KD:
  - Weibo: `dynamic_distill/configs/weibo_noisy_student.yaml` → `paper_results/weibo_noisy_student_seed0{0,1,2}`
  - WeFEND: `dynamic_distill/configs/wefend_noisy_student.yaml` → `paper_results/wefend_noisy_student_seed0{0,1,2}`
  - MiRAGeNews: `dynamic_distill/configs/miragenews_noisy_student.yaml` → seeds running
- Simple-Conf-KD:
  - Weibo: `dynamic_distill/configs/weibo_simple_conf_kd.yaml` → `paper_results/weibo_simple_conf_kd/*`
  - WeFEND: `dynamic_distill/configs/wefend_simple_conf_kd.yaml` → `paper_results/wefend_simple_conf_kd/*`
- Other baselines:
  - SAFE: `paper_results/weibo_safe/weibo_safe_seed0{0,1,2}`, `paper_results/wefend_safe/wefend_safe_seed0{0,1,2}`
  - EANN: `paper_results/weibo_eann/weibo_eann_seed0{0,1,2}`, `paper_results/wefend_eann/wefend_eann_seed0{0,1,2}`
  - Text-only / Image-only: `paper_results/weibo_text_only/*`, `weibo_image_only/*`, `wefend_text_only/*`
  - Teacher fusion: `paper_results/weibo_teacher_baseline/weibo_teacher_baseline_seed00`, `wefend_teacher_baseline/*`

## Data->plot/table mapping
- Main tables’ data paths listed in `papers/arr_rolling_review/DATA_SOURCES.md`.
- WeFEND bar plot: `papers/arr_rolling_review/figures/wefend_bar.pdf` (from Table 1).
- Error cases & per-class ECE/NLL in appendix sourced from:
  - Weibo student: `paper_results/weibo_dynamic_distill/weibo_dynamic_distill_seed01/test_predictions.csv`
  - Weibo teacher: `paper_results/weibo_teacher_baseline/weibo_teacher_baseline_seed00/test_predictions.csv`
  - WeFEND student/teacher predictions similarly named under their run dirs.
- Disagreement vs error: `papers/arr_rolling_review/figures/disagreement_error.pdf`; stats in `paper_results/weibo_disagreement_error.csv` and `paper_results/wefend_disagreement_error.csv`.
- Noise robustness (Weibo): `paper_results/weibo_noise_comparison.csv` (noise 0.1/0.2/0.3, DMPD vs Noisy-Student).

## Known quirks / pitfalls
- Earlier reliability/gate plots were placeholders; **do not reuse**. Only method diagram + bar chart are in paper.
- Weibo gate coverage in best run was near-zero (conservative thresholds) — loose-gate runs are in progress (`weibo_dynamic_distill_loosegate_seed00`, `wefend_dynamic_distill_loosegate_seed00`).
- WeFEND event reliability is extreme (many single-post events); per-event accuracy histogram is sharply peaked at 1.0.
- MiRAGeNews: `evaluate_model.py --split testX` expects either the default test split in config or `--test-csv` to override; don’t pass full path unless you set it correctly relative to dataset root.
- HF downloads can occasionally SSL fail; retry usually works (as seen in logs).

## If new experiments are needed (suggested order)
1) (Done) MiRAGeNews Noisy-Student 3 seeds finished; table updated.  
2) (Running) “Looser gate” DMPD on Weibo/WeFEND (τ_f=0.90, τ_m=0.85, δ=0.10): `paper_results/weibo_dynamic_distill_loosegate_seed00`, `.../wefend_dynamic_distill_loosegate_seed00`. Extract coverage (train_history.num_distill_pairs) and test metrics when done.  
3) (Done) Cross-modal disagreement vs. error plot: `papers/arr_rolling_review/figures/disagreement_error.pdf`, stats in `paper_results/weibo_disagreement_error.csv`, `paper_results/wefend_disagreement_error.csv`.  
4) (Done) Noise robustness small table: `paper_results/weibo_noise_comparison.csv` (noise 0.1/0.2/0.3, DMPD vs Noisy-Student, 1 seed).  
5) (Future) Stronger teacher/VLM teacher or synthetic-noise coverage curves (high effort; not for this ARR round).

## Quick commands (templates)
- Train:  
  `CUDA_VISIBLE_DEVICES=GPU python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/<cfg>.yaml --seed SEED --output-dir paper_results/<run>_seedXX > logs/<run>_seedXX.log 2>&1 &`
- Evaluate MiRAGeNews OOD split:  
  `python dynamic_distill/scripts/evaluate_model.py --config dynamic_distill/configs/miragenews_noisy_student.yaml --checkpoint paper_results/miragenews_noisy_student_seed00/miragenews_noisy_student_seed00/model_best.pt --test-csv miragenews_test2_bbc_dalle.csv --split test2 --output paper_results/miragenews_noisy_student_seed00/test2_eval.json`
