# DPMD 论文写作资料（IEEE 期刊投递版）

本文件汇集本项目可直接用于论文撰写与复现实验的完整素材：方法描述要点、数据与划分细节、训练配置与脚本、可复现命令、当前最优结果、失败与威胁分析、推荐图表与表格清单等。面向 IEEE TIFS/TPAMI/Access 等稿件结构组织。

---

## 1. 方法总览（可直接写入“Method”）
- 核心框架：Dynamic Probabilistic Multi‑teacher Distillation (DPMD)
  - 双分支编码：文本（BERT/RoBERTa/DeBERTa）与视觉（ViT/CLIP），编码维度 768/1024。
  - 不确定性证据头：Dirichlet EDL（`DirichletEvidenceHead`），输出 α=evidence+1，基于强度计算不确定性；融合层 `UncertaintyWeightedFusion` 以模态不确定性加权融合。
  - 双向蒸馏：fusion ↔ text / vision；温度 `T`、KL 系数 `λ_kl`、特征蒸馏 `λ_feat` 可配（详见 YAML）。
  - 事件感知门控：
    - `event_filter`：基于教师在事件上的历史准确率/平均置信度与最小样本数控制蒸馏资格；
    - `positive_event_gate`：对正类事件单独放宽门槛与增益；
    - `weak_teacher`：事件级多视角弱教师（EMA/共识）补充监督；
    - `consistency`：事件/样本一致性正则化；
    - `fusion_teacher_must_match`、`require_student_mistake` 等布尔开关控制触发场景。
- 训练流程（伪代码）：
  1) 前向得到 text/vision logits 与 EDL evidence，计算不确定性；
  2) 融合层以不确定性加权生成融合 logits；
  3) 依据 batch 的 `event_id` 更新事件统计；
  4) 构造蒸馏掩码：满足事件可靠度与置信度门槛（以及正类门控）才参与 KL/feature 蒸馏；
  5) 总损失：`L = α·CE_fusion + β·(CE_text+CE_vision) + γ·KL + EDL_loss + consistency + weak_teacher`；
  6) 采用余弦退火 + warmup 调度，早停以 Macro‑F1 或自定义指标。
- 代码位置：
  - 训练主循环：`dynamic_distill/scripts/train_mvp.py`
  - Trainer 与门控逻辑：`dynamic_distill/src/training/trainer.py`
  - 模型与头：`dynamic_distill/src/models/{multimodal.py, encoders.py, heads.py, fusion.py}`

---

## 2. 数据与严格划分（可写入“Dataset & Protocol”）
### 2.1 WeFEND（WeChat）
- 路径：`datasets/wechat/processed_split/wefend_{train,val,test}.csv`
- 规模与分布：train 7,231（负 5,698 / 正 1,533）、val 905（713/192）、test 905（713/192）。
- 划分原则：按公众号（账号）事件互斥；图像文件以 guid 的 SHA256 命名；数据脚本见 `data_prep/wefend_make_split.py`。

### 2.2 Fakeddit
- 原始文件：`datasets/fakeddit/raw/multimodal_only_samples/multimodal_{train,validate,test_public}.tsv`
- 图像下载：`data_prep/download_images.py`（User-Agent + 多线程），本地已缓存约 439,101 张哈希图。
- 严格划分脚本：`data_prep/fakeddit_strict_split.py`（筛选“图文齐全”的样本并生成严格划分 CSV）。
  - 子板块互斥（subreddit）：
    - `datasets/fakeddit/processed_strict/subreddit/fakeddit_{train,val,test}.csv`
    - 规模：train 303,552（负 192,087 / 正 111,465；unique subreddits=15），val 47,857（负 43,150 / 正 4,707；events=4），test 87,692（正 65,959 / 负 21,733；events=3）。
    - 特点：跨 split 标签与事件分布极端不一致（val≈全负、test≈多正），是目前最严酷设定。
  - 时间切分（time）：
    - `datasets/fakeddit/processed_strict/time/fakeddit_{train,val,test}.csv`
    - 规模：train 307,370（负 207,748 / 正 99,622），val 65,865（正 37,946 / 负 27,919），test 65,866（正 44,563 / 负 21,303）。
    - 特点：更符合“跨时间泛化”，分布相对平衡，建议用作主协议之一。

### 2.3 Weibo
- 历史结果表：`paper_results/tables/weibo_multi_seed_results.csv`（均值 Acc≈0.92 / ECE≈0.058）。
- 后续需补充严格“事件/时间”划分脚本以统一协议。

---

## 3. 训练配置与关键超参（可写入“Implementation Details”）
- 统一入口：`python dynamic_distill/scripts/train_mvp.py --config <yaml> --seed <s>`。
- 批量调度：`scripts/run_with_free_gpus.sh <job_list.txt>`（仅占用 GPU2–6，日志无 tqdm）。Job 行格式：`<config> <seed> <log_path> [extra_args]`。
- 典型 YAML 字段（以 WeFEND 最佳 `event_consensus_balanced.yaml` 为例）：
  - `distillation.temperature=2.32`、`delta=0.065`、`lambda_kl=1.05`、`lambda_feat=0.1`；
  - `confidence_gate.margin=0.18`；`uncertainty_ema.momentum=0.9`；
  - `require_student_mistake=true`；`fusion_teacher_must_match=true`；
  - `event_filter.{min_size=2, teacher_min_acc=0.9, teacher_min_conf=0.92, warmup_steps=5}`；
  - `positive_event_gate.{enabled=true, teacher_conf=0.955, student_conf=0.84}`；
  - 视觉增强：flip 0.5、color_jitter 0.2、random_rescale 0.1；文本 word_dropout 0.1。
- 大模型教师（Fakeddit 子板块划分）：`fakeddit_teacher_roberta_subreddit.yaml`（RoBERTa‑large + ViT‑large，epoch 18，warmup 1200，class_weights 0.65/1.45）。

---

## 4. 可复现实验命令（可写入“Reproducibility”）
- 环境：A100 80GB；仅使用 GPU2–6；`conda activate ddistill`；首次下载 HF 权重可能超时需重试。
- 下载图像（示例）：
```bash
python dynamic_distill/data_prep/download_images.py \
  --csv datasets/fakeddit/processed/fakeddit_train.csv \
  --output-dir datasets/fakeddit/images --workers 32
```
- 生成严格划分：
```bash
# 子板块互斥
python dynamic_distill/data_prep/fakeddit_strict_split.py \
  --raw-dir datasets/fakeddit/raw/multimodal_only_samples \
  --processed-dir datasets/fakeddit/processed \
  --images-dir datasets/fakeddit/images \
  --output-dir datasets/fakeddit/processed_strict/subreddit \
  --strategy subreddit
# 时间切分
python dynamic_distill/data_prep/fakeddit_strict_split.py \
  --raw-dir datasets/fakeddit/raw/multimodal_only_samples \
  --processed-dir datasets/fakeddit/processed \
  --images-dir datasets/fakeddit/images \
  --output-dir datasets/fakeddit/processed_strict/time \
  --strategy time
```
- 启动教师/学生（批处理，不输出 tqdm）：
```bash
# 多 seed 教师（RoBERTa + ViT‑L）
cat > runs/joblists/fkd_teacher.txt <<EOF
dynamic_distill/configs/fakeddit_teacher_roberta_subreddit.yaml 0 logs/fkd/teacher_seed00.log
dynamic_distill/configs/fakeddit_teacher_roberta_subreddit.yaml 1 logs/fkd/teacher_seed01.log
dynamic_distill/configs/fakeddit_teacher_roberta_subreddit.yaml 2 logs/fkd/teacher_seed02.log
EOF
scripts/run_with_free_gpus.sh runs/joblists/fkd_teacher.txt

# WeFEND 蒸馏最优策略（示例）
python dynamic_distill/scripts/train_mvp.py \
  --config dynamic_distill/configs/wefend_event_consensus/event_consensus_balanced.yaml \
  --seed 0
```
- 结果目录结构：`paper_results/<dataset>/<run_name>_seedXX/`，包含 `summary.json`、`train_metrics.csv`、`val_metrics.csv`、`test_predictions.csv`。

---

## 5. 当前指标与对比（可写入“Experiments”）
- WeFEND（事件互斥划分）：
  - 学生最优（event consensus balanced, seed0）：test Acc 0.9436 / Macro‑F1 0.9152 / Pos‑F1 0.8661 / ECE 0.0506。
  - 路径：`paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2/event_consensus_balanced_seed00/summary.json`。
- Fakeddit（严格子板块划分）：
  - BERT+ViT baseline（旧 i.i.d.）：test Acc ≈0.83。
  - 新协议（subreddit 互斥）：BERT‑base 教师 test Acc ≈0.36（正类召回低）；RoBERTa+ViT‑L 教师多 seed 训练中，val Acc≈0.69、Macro‑F1≈0.53 左右（需平衡验证/测试集）。
- Weibo：多 seed 旧结果 Acc≈0.92（后续需按严格协议重跑）。

> 备注：Fakeddit 子板块划分的分布极端不一致（val≈全负、test≈多正），是造成教师与学生在 test 上表现不稳的主因。建议在论文中作为“更严苛协议”呈现，并报告 Macro‑F1/Pos‑F1 与 ECE，而非单纯 Acc。

---

## 6. 消融与失败经验（可写入“Ablation/Discussion”）
- WeFEND：尝试过 soft quota、fallback、正类强化、事件校准（Beta 先验）、弱教师 EMA/共识等；`event_reliable` 为最稳定收益，但提升幅度 ~+1.0pt，尚未达到 +1.5pt 目标。
- Fakeddit：在图像未补齐与事件门控偏紧时，蒸馏掩码长期为 0；放宽门限后 BERT‑base 仍有限。严格子板块划分暴露出验证/测试分布翻转问题。
- 详细过程与日志：
  - WeFEND：`dynamic_distill/experiments/wefend_failed_attempts.md`
  - Fakeddit：`dynamic_distill/experiments/fakeddit_failed_attempts.md`

---

## 7. 统计检验与校准（可写入“Evaluation Protocol”）
- 指标：Acc、Macro‑F1、Pos‑F1、ECE（`scripts/evaluate_model.py` / `expected_calibration_error`）。
- 建议：每个配置至少 3 个种子，报告均值±方差；对关键对比（teacher vs student）做成对 t‑test 或 Wilcoxon 检验；
- 置信度校准：温度缩放（`scripts/calibrate_temperature.py`）；阈值扫描（`calibrate_threshold.py`）。

---

## 8. 伦理与有效性威胁（可写入“Ethics/Threats to Validity”）
- 事件/时间泄露：本工作显式按事件/子板块/时间划分，避免同事件跨 split。
- 分布偏移：Fakeddit 子板块划分存在严重标签偏移，本工作予以披露并采用 Macro‑F1/Pos‑F1 评估；
- 可复现性：提供脚本/配置/数据处理代码与随机种子；HF 下载波动可重试。

---

## 9. 推荐图表与表格清单
- 表 1：数据集统计（样本数、正负比、事件数）——三数据集两协议（subreddit/time）。
- 表 2：主结果表（Teacher/Student/强基线），指标 Acc/Macro‑F1/Pos‑F1/ECE（含多 seed 平均±方差）。
- 表 3：消融（门控/弱教师/一致性/温度/δ 课程 等）。
- 图 1：方法框图（双分支 + EDL + 双向蒸馏 + 事件门控）。
- 图 2：训练曲线与校准曲线（ECE/reliability diagram）。
- 图 3：严格划分 vs i.i.d. 的性能对比与混淆矩阵。

---

## 10. 附：关键路径速查
- 训练脚本：`dynamic_distill/scripts/train_mvp.py`
- 事件门控/Trainer：`dynamic_distill/src/training/trainer.py`
- 模型组件：`dynamic_distill/src/models/`
- 数据读取：`dynamic_distill/src/data/`
- 配置示例：`dynamic_distill/configs/`
- 批处理脚本：`scripts/run_with_free_gpus.sh`
- 结果目录：`paper_results/`
- 失败记录：`dynamic_distill/experiments/*.md`

> 如需将本文档导入到另一 AI/编辑器，可直接作为论文大纲与补充材料使用；所有路径均为仓库相对路径，可一键定位。
