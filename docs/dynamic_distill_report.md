# Dynamic Modal Priority Distillation – Technical Report (2025-10-27)

## 1. Overview

We extend the `dynamic_distill` module to study **uncertainty-guided teacher selection** for multimodal fake-news detection. The system operates on paired text–image inputs (currently Weibo/Twitter) and supports future integration with propagation-graph models (MUGCL). Key ideas:

1. **Dirichlet Evidential Heads** estimate per-modality uncertainty (via evidential DL) alongside logits.
2. **Dynamic teacher selection** picks the lower-uncertainty modality per sample and distills knowledge to the other modality only when the teacher is confident.
3. **Stability mechanisms** (EMA teacher, adaptive temperatures, confidence gating, uncertainty smoothing, delayed activation) reduce noisy supervision.

The goal is to improve accuracy without sacrificing calibration—a critical requirement for real-world governance settings.

## 2. Model Architecture

### Encoders
- **Text**: BERT-based encoder (Weibo: `bert-base-chinese`, Twitter: `bert-base-uncased`) → pooled representation `x_t`.
- **Vision**: ViT-B/16 (`google/vit-base-patch16-224-in21k`) → CLS token embedding `x_v`.

### Heads (per modality)
- **Classification head**: MLP → logits `logits_t`, `logits_v`.
- **Evidential head**: MLP + softplus → evidence `evidence_t`, `evidence_v`, yielding Dirichlet concentration `alpha = evidence + 1` and uncertainty `u = C / sum(alpha)`.

### Fusion Branch
- Lightweight cross-modal fusion (currently uncertainty-weighted concatenation) produces `logits_fuse` for final predictions.

## 3. Distillation Strategy

### Baseline (v1)
- Choose teacher as modality with lower uncertainty (difference > δ).
- Distill logits via temperature-scaled KL divergence plus optional feature MSE.
- Freeze teacher gradients via `detach()`; optional EMA teacher network.

### Improvements (current)
1. **Delayed activation**: enable distillation only after `distillation.start_fraction` of epochs or `warmup_steps` of iterations.
2. **Adaptive temperature**: `T = base + coeff × max(0, u_student − u_teacher)` softens guidance when teacher is much more certain.
3. **Confidence gating**: require teacher top-1 minus top-2 probability ≥ margin before distillation.
4. **Uncertainty EMA**: smooth `u_t`, `u_v` over iterations to reduce jitter in teacher selection.
5. **Selective MSE**: apply feature matching only when KL is active and on pre-projected features (λ_feat ≤ 0.05 in weibo_full config).

### Loss Summary

```
L = L_fuse
  + α (L_cls_t + L_cls_v)
  + β (L_evi_t + L_evi_v)
  + γ L_distill

L_distill = λ_KL · KL(logits_teacher || logits_student)
          + λ_feat · MSE(z_teacher, z_student)
```

Additional controls: temperature schedule, confidence gate, EMA teacher (Polyak averaging), gradient clipping (5.0), AdamW optimizer with cosine LR schedule.

## 4. Key Hyperparameters (Weibo `weibo_full.yaml`)

| Component | Value |
|-----------|-------|
| Encoder hidden size | 768 |
| Optimizer | AdamW, lr 5e-5 (encoder/head), weight decay 0.01 |
| Epochs | 30 |
| Batch size | 16 |
| Distill warm-up steps | 500 |
| Distill start fraction | 0.4 |
| Temperature | base 2.0 + coeff 3.0 × Δu |
| Confidence gate | enabled, margin 0.25 |
| Uncertainty EMA | enabled, momentum 0.9 |
| Evidence KL anneal | 2000 warm-up iterations (linear) |
| Feature MSE weight | λ_feat = 0.05 |

Twitter (`twitter_mvp.yaml`) uses the same architecture with lighter schedule (20 epochs, batch 8, coeff 2.0, margin 0.2) suited to its smaller dataset.

## 5. Experimental Results (Weibo, multi-seed)

Trained with seeds {0,1,2,3,4} on GPU 5 under the enhanced configuration. Raw summaries: `paper_results/runs/weibo_full_seed0X_summary.json`.

| Seed | Accuracy | Macro-F1 | Positive-F1 | ECE |
|------|----------|----------|-------------|-----|
| 0 | 0.9147 | 0.9146 | 0.9167 | 0.0633 |
| 1 | **0.9474** | **0.9474** | **0.9484** | **0.0381** |
| 2 | 0.9195 | 0.9193 | 0.9224 | 0.0603 |
| 3 | 0.9174 | 0.9172 | 0.9212 | 0.0625 |
| 4 | 0.9085 | 0.9084 | 0.9118 | 0.0642 |

Aggregated statistics:
- Accuracy: 0.9215 ± 0.0135
- Macro-F1: 0.9214 ± 0.0135
- ECE: 0.0577 ± 0.0099

Observations:
1. **Accuracy/F1 gains**: Compared to the initial single-run (F1 ≈ 0.8935), the new pipeline adds ≈ +2.8 Macro-F1 on average, peaking at 0.947.
2. **Calibration improvement**: ECE dropped from 0.0691 to as low as 0.038, indicating more trustworthy confidence estimates.
3. **Stability**: Distillation becomes active only when teacher confidence is high, leading to consistent validation plateaus and reduced oscillation across seeds.

Artifacts prepared for manuscript:
- `paper_results/tables/weibo_multi_seed_results.csv`
- `paper_results/tables/weibo_multi_seed_summary.json`
- Raw predictions in `runs/weibo_full_seed0X/test_predictions.csv` (per run).

## 6. Next Steps

1. **Twitter multi-seed**: replicate analysis on the MediaEval Twitter corpus.
2. **Slice/OOD evaluation**: generate per-slice metrics (modality missing, sarcasm, early propagation) and visualization (reliability diagrams, t-SNE).
3. **Statistical significance**: bootstrap CI & McNemar tests vs. baseline (γ=0, no adaptive features).
4. **Fakeddit & WeChat (MUGCL)**: acquire datasets, build propagation graphs, and extend distillation to graph-aware setting.
5. **Baselines**: integrate official or authoritative implementations (e.g., EANN repo) and document reimplementation strategy for closed-source methods (CAFE, MUGCL, MRHFR).

This report will be updated as additional datasets/baselines and analyses are completed.

## 7. Operational Notes / Pitfalls

- **GPU 通信卡死**：在多 GPU 环境下，如果机器禁用 NCCL P2P/InfiniBand，会导致 `nn.DataParallel` 在首个 batch 的前向阶段永久阻塞。我们据此把训练器默认改为**单卡模式**（`TrainerConfig.allow_data_parallel=False`）。需要多卡时，请显式在配置中设置 `training.allow_data_parallel: true`，并提前导出下列环境变量：`MASTER_ADDR/MASTER_PORT`、`NCCL_IB_DISABLE=1`、`NCCL_P2P_DISABLE=1`、`DISABLE_TORCHCODEC=1` 等，确保 NCCL 退化到安全路径，否则日志会停在 `[trace] step 1 ... forward start` 不再推进。
