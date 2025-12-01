# DPMD 项目论文写作资料汇总

## 1. 研究目标与方法概述
- **目标**：在多模态假新闻/谣言数据集（Weibo、WeFEND、Fakeddit）上，提出“基于不确定性的双向教师-学生蒸馏”框架（Dynamic Probabilistic Multi-teacher Distillation, DPMD），要求：
  1. 数据划分严格、无信息泄露（按事件/子板块/账号或时间）；
  2. 与公开强基线（SpotFake+/CAFE/MAGIC 等）在同分布测试集对比；
  3. 蒸馏版本需优于未蒸馏教师 ≥1.5pt，并领先其他方法。
- **主要创新点**：
  - 文本/视觉双分支 + 不确定性证据头（Dirichlet EDL），融合层使用不确定性权重；
  - 双向蒸馏（fusion↔text/vision）+ 事件/正类门控、软配额、弱教师 EMA、事件共识等策略；
  - 面向事件/子板块的动态筛选与重加权，保障跨事件泛化。

## 2. 数据资源与划分
### 2.1 WeFEND（微信/WeChat）
- 已按账号事件划分（train≈7.2k / val≈0.9k / test≈0.9k，正负≈3.7:1），图像按哈希命名。
- 蒸馏最佳：`event_consensus_balanced_seed00` → test Acc 0.9436 / Macro-F1 0.9152 / Pos-F1 0.8661。
- 失败/改进记录：`dynamic_distill/experiments/wefend_failed_attempts.md`。

### 2.2 Fakeddit
- 原始多模态样本 56 万+；通过 `download_images.py` 批量拉取图像，目前本地 `datasets/fakeddit/images/` 约 43.9 万张。
- 已构建严格划分脚本 `dynamic_distill/data_prep/fakeddit_strict_split.py`：
  - `processed_strict/subreddit/`：按 subreddit 事件互斥（train 30.4 万、val 4.8 万、test 8.8 万，严重正负失衡：train neg≈64%，val neg≈90%，test pos≈75%）。
  - `processed_strict/time/`：按时间顺序划分（用于跨时泛化）。
- 教师实验：
  - BERT+ViT baseline（旧）: test Acc ≈0.83。
  - Subreddit 严格划分：`fakeddit_uncertainty_fusion_subreddit`（BERT-base）test Acc≈0.36；`fakeddit_teacher_roberta_subreddit`（RoBERTa+ViT-L）正在多 seed 训练（GPU2–4）。
- 蒸馏配置（event consensus 等）已准备但需等待教师稳定后再跑。
- 日志、失败总结：`dynamic_distill/experiments/fakeddit_failed_attempts.md`。

### 2.3 Weibo
- 历史划分文件 `paper_results/tables/weibo_multi_seed_results.csv`，平均 Acc≈0.9215 / ECE≈0.0577；蒸馏配置沿用动态蒸馏框架（细节在 docs/report_2025-10-29.md）。
- 尚未套用严格事件划分脚本，后续需补充。

## 3. 实验配置与脚本
- 统一训练入口：`dynamic_distill/scripts/train_mvp.py`。
- 批量调度：`scripts/run_with_free_gpus.sh`（仅使用 GPU2–6，日志简洁无 tqdm）。
- 数据集定义：`dynamic_distill/src/data/{wefend,fakeddit,weibo}.py`。
- 主要模型/配置：
  - Baseline 教师：`configs/fakeddit_uncertainty_fusion*.yaml`、`configs/wefend_uncertainty_fusion_v2.yaml`。
  - 蒸馏策略：`configs/*event_consensus*`、`configs/wefend_dynamic_distill_teacher_wcls_*` 等。
  - 新教师：`configs/fakeddit_teacher_roberta_subreddit.yaml`（RoBERTa + ViT-L，大 batch、class weights、长训练）。
- 失败/经验记录：
  - `dynamic_distill/experiments/wefend_failed_attempts.md`
  - `dynamic_distill/experiments/fakeddit_failed_attempts.md`

## 4. 当前结果概览
| 数据集 | 配置 | Test Acc | Macro-F1 | 备注 |
| --- | --- | --- | --- | --- |
| WeFEND | `event_consensus_balanced_seed00` | 0.9436 | 0.9152 | 目标差值 +1.0pt（需 ≥1.5pt） |
| Fakeddit（子板块划分） | `fakeddit_uncertainty_fusion_subreddit` (BERT) | 0.3611 | 0.3604 | 教师过弱，正类召回低 |
| Fakeddit（子板块划分） | `fakeddit_teacher_roberta_subreddit` | 训练至 epoch16，val Acc≈0.69 | – | 正在多 seed 运行，需平衡 val/test |
| Weibo | 旧多种子结果 | Acc≈0.92 | – | 后续需应用严格划分与蒸馏改进 |

## 5. 待完成事项（亦可作为论文写作中的“未来工作/实验计划”）
1. **Fakeddit**：
   - 平衡验证/测试（或切换到 time 划分）+ 使用 class-balanced sampler / focal loss；重新训练 RoBERTa+ViT-L 教师（多 seed）。
   - 复现 SpotFake+/CAFE/MAGIC 等强基线，全部使用严格划分的 CSV；整理对比表。
   - 蒸馏版本：在教师稳定后，运行 `event_consensus_*` 系列 + 不同种子，确保蒸馏提升 ≥1.5pt。
2. **WeFEND**：继续寻找 ≥1.5pt 的蒸馏差值（目前 +1.0pt）。可尝试 pseudolabel、事件过滤、弱教师增强等策略。
3. **Weibo**：应用严谨划分脚本（例如按事件/时间），跑所有基线 + 蒸馏，形成三数据集统一对比。
4. **论文素材**：
   - 折线/柱状图：`paper_results/figures/`（部分已有，如 Twitter seed 图）。
   - 失败案例、日志引用：见 `dynamic_distill/experiments/*.md`。
   - 数据 pipelines/脚本说明：`docs/report_2025-10-29.md`、`dynamic_distill/data_prep/*.py`。

## 6. 推荐写作结构提示
1. **数据集与划分**：强调我们重新划分、严格控制泄露；提供统计表（样本数、正负比例、事件数量）。
2. **方法**：详细描述双向蒸馏、不确定性建模、事件门控、弱教师 EMA、positive focus 等模块（可引用 YAML 字段解释）。
3. **实验设置**：说明训练细节、硬件（GPU2–6, A100 80GB）、批量脚本、超参（见 YAML）。
4. **结果**：
   - 逐数据集比较（Teacher vs Student vs 强基线）。
   - 用表格呈现 Acc、Macro-F1、Pos-F1、ECE 等。
   - 可视化：折线图（性能随策略迭代）、柱状图（多方法对比）。
5. **消融/失败分析**：引用 `..._failed_attempts.md` 中的策略与结论。
6. **讨论**：强调严格划分带来的挑战、模型在不同数据集的泛化差异，以及后续计划（如更平衡的验证策略）。

---
> 最新状态：RoBERTa+ViT-L 教师（seed0/1/2）在 GPU2–4 运行中，待收敛后可获得更新的指标。若需要更多原始日志/配置，可参阅对应目录。
