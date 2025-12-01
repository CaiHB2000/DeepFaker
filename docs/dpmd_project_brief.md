# DPMD 项目全景与论文写作资料包（投前交接版）

本文面向“另一位AI撰写论文”的交接。内容覆盖：代码/模块结构、方法技术路线、核心参数与默认值、数据与划分原则、实验结果与结论、复现实验与目录对照表。将本文件交给写作代理即可形成论文初稿与补实验清单。

---

## 0. 术语与目标

- DPMD：Dynamic Priority Multi-modal Distillation（动态优先级多模态蒸馏）。
- 目标：在严格、无泄漏的事件/账号感知划分上，构建多模态文本+图像的虚假信息检测方法；以不确定性和事件可靠度为核心信号，动态决定蒸馏方向与时机；在多数据集上超过未蒸馏基线与公开方法，且具备可复现与可信评测。
- 设备约束：GPU 0/1 为他人保留，不使用；训练前以 `nvidia-smi` 检查可用的 2~6 号卡。

---

## 1. 仓库结构（与论文章节映射）

根目录仅列关键子树（更多见 `rg --files`）。

- `dynamic_distill/`（论文方法主体）
  - `src/` 方法与训练核心
    - `training/trainer.py`：训练总控，包含所有“动态蒸馏/事件门控/不确定性EMA/融合蒸馏/δ课程/事件重加权/fallback”逻辑。
    - `losses/distillation.py`：蒸馏损失（KL/MSE），支持不确定性加权与覆写掩码。
    - `models/`（未逐文件列举）：`DynamicModalDistillationModel` 及 `TextEncoder`/`VisionEncoder`。文本默认 `bert-base-chinese`，图像默认 `google/vit-base-patch16-224-in21k`。
    - `data/`：数据集装载器
      - `wefend.py`：WeFEND（WeChat）数据，事件由公众号名映射；输出 `event_id` 与 `event_size`。
      - `fakeddit.py`：Fakeddit 多模态（文本+图）实现（已集成）。
      - 统一由 `src/data/builder.py`（在 `build_datasets` 中）调度（由 `scripts/train_mvp.py` 调用）。
    - `utils/`：ECE 计算、warmup 等。
  - `configs/`（论文附录/补充材料列参照）
    - WeFEND 关键配置：
      - `wefend_uncertainty_fusion_v2.yaml`（教师/未蒸馏基线）；
      - `wefend_dynamic_distill_teacher_wcls_event_reliable.yaml`（事件可靠度门控，当前最佳思路之一）；
      - `wefend_dynamic_distill_teacher_wcls_event_reliable_{uncertainty,studentfocus,curriculum,reweight,hybrid}.yaml`（一系列方案探索）；
      - `wefend_dynamic_distill_teacher_wcls_event_reliable_fallback.yaml` 与 `..._fallback_soft.yaml`（蒸馏样本不足时的安全兜底）。
    - Fakeddit 关键配置：`fakeddit_{uncertainty_fusion,dynamic_distill,_teacher}.yaml`。
  - `scripts/`（训练/评估/校准）
    - `train_mvp.py`：主训练脚本（方法、基线与教师蒸馏均在此运行）。
    - `evaluate_model.py`：离线评估，导出预测CSV与指标。
    - `calibrate_temperature.py`、`calibrate_threshold.py`：后处理校准。
    - `run_experiments.py`：批量实验入口（可作为论文附带脚本）。
  - `data_prep/`：
    - `wefend_make_split.py`、`wefend_prepare.py`：WeFEND 去重、账号事件映射与严格划分。
    - `fakeddit_prepare.py`、`download_images.py`：Fakeddit 资源准备。
  - `experiments/wefend_failed_attempts.md`：WeFEND 全部失败尝试与超参记录（论文可做附录“负结果与经验”）。

- `paper_results/`（实验资产与可视化）
  - `wefend_*`、`fakeddit_*` 子目录：每次运行的 `config.yaml`、`summary.json`、`train/val_metrics.csv`、`test_predictions.csv` 等。
  - `tables/`：如 `fakeddit_wefend_summary.csv`（跨seed汇总）等。
  - `figures/`：生成的图表（若存在）。

- `datasets/`（原始/处理后数据概览）
  - `wechat/processed_split/wefend_{train,val,test}.csv`（WeFEND 严格划分CSV）。
  - `fakeddit/processed/`（Fakeddit 处理数据，文本与图像路径）。
  - 其余 Twitter/Weibo 目录仅用于早期或旁路研究，主线当前聚焦 WeFEND 与 Fakeddit；Twitter 明确不再继续。

---

## 2. 方法技术原理（论文方法部分直用）

### 2.1 模型与不确定性
- 多模态分类器 `DynamicModalDistillationModel`：文本编码器（BERT）与视觉编码器（ViT）分别输出 `logits`/`penultimate` 与“evidence α”（用于Dirichlet evidential loss）。融合头输出 `fusion_logits`。
- 不确定性估计：基于 Dirichlet evidential loss（`src/losses/__init__.py`）得到每模态的证据与不确定性，训练中每步可得 `text_modal.uncertainty` 与 `vision_modal.uncertainty`。
- 不确定性 EMA：在蒸馏触发判定时使用滑动平均的不确定性，降低瞬时噪声（`TrainerConfig.use_uncertainty_ema`）。

### 2.2 动态蒸馏（核心）
- 核心函数：`losses/distillation.py::compute_dynamic_distillation`。
- 蒸馏方向由“模态间不确定性差”决定：若 `u_text + δ < u_vision`，文本为教师，视觉为学生；反之亦然。
- δ 课程（delta schedule）：随 epoch 线性收缩 δ，提高中后期触发率（`distillation.delta_schedule`）。
- 置信门控与一致性：
  - 教师 logits 经 softmax 的Top‑2差值 ≥ margin 方可触发（`confidence_gate`）。
  - 可要求学生当前预测错误或置信更低（`require_student_mistake` 与 `agreement_confidence_gap`）。
- 事件感知门控：
  - `event_filter`：为每个“事件”（WeFEND中对应公众号）维护滚动统计（教师在该事件的精度与平均置信），低于门槛则禁止在该事件触发蒸馏。
  - `positive_event_gate`：正样本事件的专用门；可按事件规模、教师/学生置信阈值进一步筛选，支持 `only` 将正类样本单独限定触发。
- 不确定性权重：KL/MSE 蒸馏项按“教师‑学生不确定性差”的幂函数放大（`uncertainty_weight_{enabled,scale,power,clip}`）。
- 融合→单模态蒸馏：当融合教师可信时，以小权重引导单模态学生（`lambda_fusion_to_text/vision`）。
- 事件/学生聚焦（可选）：
  - `student_focus`：若某事件上学生（或学生正类召回）低于阈值，则对该事件优先蒸馏。
  - `event_reweight`：事件级分类损失重加权，聚焦难事件或正类事件。
- Fallback 蒸馏（样本不足兜底）：若本批次蒸馏样本占比低于 `min_pairs_fraction`，则在延迟启用后（`fallback_start_fraction`）以更高温度/更小 KL 权重、且可按（教师‑学生）置信差动态缩放 KL，对额外样本进行“软蒸馏”。

### 2.3 损失函数
总损失：
```
L = CE(fusion) + α·[CE(text)+CE(vision)] + β·[EVI(text)+EVI(vision)] + γ·[KL/MSE 蒸馏（含可选融合→单模态）]
```
- 交叉熵可带类别权重（应对类别不平衡）。
- Evidential loss 用于不确定性学习与校准。
- 蒸馏项含：模态间KL、可选特征MSE、可选融合→模态KL；支持样本/事件加权与不确定性/置信差缩放。

---

## 3. 关键参数总表（按配置节）

以下以 WeFEND 最佳/重要配置为例（其余配置语义一致）：

### 3.1 训练与优化
- `training.epochs`：常用 14~18（WeFEND 多为16）。
- `training.batch_size`：32。
- `optim.lr`：1.7e‑5 ~ 2.0e‑5；`weight_decay`：0.01。
- `scheduler`：`cosine`，`warmup_steps` 600~720。
- `training.max_steps_per_epoch`：用以加速迭代（WeFEND 常设 220~240）。
- `early_stopping_patience`：4~6。

### 3.2 模型与数据
- 文本编码：`bert-base-chinese`，`tokenizer.max_length`=160，`word_dropout`=0.05~0.1。
- 视觉编码：`google/vit-base-patch16-224-in21k`，`image_size`=224；训练时常开 `augment`（翻转/亮度/随机裁剪）。
- 类别权重（WeFEND 示例）：`loss.class_weights=[0.635, 2.3597]`（负/正）。

### 3.3 动态蒸馏
- 不确定性阈 `delta`：0.07 起始，课程收缩至 0.03~0.045。
- 蒸馏温度 `temperature`：2.4~2.6；`adaptive_temperature`: base 2.4~2.5, coeff 1.3~1.6。
- KL/特征权重：`lambda_kl` 1.05~1.25；`lambda_feat` 0.1~0.12。
- 触发时窗：`start_fraction` 0.18~0.48；`end_fraction` 0.58~0.70（正类阶段性 boost 可在 0.33~0.48 启用）。
- 置信门控：`confidence_gate.margin` 0.17；`agreement_confidence_gap` 0.03。
- 正类增强：`positive_distill_boost` 1.4~1.55；`positive_student_conf_margin` 0.88~0.90；`positive_stage_boost` 1.85~2.2。
- 事件可靠度：`event_filter.min_size=2`；`teacher_min_acc` 0.84~0.9；`teacher_min_conf` 0.9~0.92；`warmup_steps` 3~5。
- 不确定性权重：`uncertainty_weight.{enabled,scale(1.1~2.0),power(1.0~1.25),clip(2.5~4.2)}`。
- 融合→单模态KL：`lambda_fusion_to_text/vision` 0.12~0.18；`fusion_confidence` 0.16~0.22。
- 学生聚焦：`student_focus.{enabled,warmup,threshold,mode(pos_recall/acc_below)}`。
- 事件重加权：`loss.event_reweight.{enabled,scale(1.5~2.3),power(1.0~1.2),focus_positive,clip}`。
- Fallback：`fallback.{enabled,min_pairs_fraction(0.12~0.25),start_fraction(0.32~0.35),temperature(2.8~3.0),lambda_scale(0.2~0.35),confidence_scale(≤1.8)}`。

---

## 4. 数据与划分（无泄漏规则）

### 4.1 WeFEND（WeChat）
- 位置：`datasets/wechat`，处理后CSV：`processed_split/wefend_{train,val,test}.csv`；图片目录：`images/`（文件名为原始guid或哈希）。
- 事件定义：公众号（`Ofiicial Account Name`）映射为 `event_id`；`event_size` 为同一公众号下样本计数；缺失记为 ‑1/0。
- 划分原则：
  - 同一公众号（事件）不跨 split；
  - 文本去重、图片存在性校验；
  - 长度与图像可读性简单过滤（`train_mvp.py` Collate 内有容错与黑底占位）。

### 4.2 Fakeddit
- 位置：`datasets/fakeddit`；处理脚本：`dynamic_distill/data_prep/fakeddit_prepare.py`；图像下载 `download_images.py`。
- 我们采用官方层次标签中的二分类版本（真实/虚假），并保证训练/验证/测试按 submission/story 级别去重，避免同帖/同事件泄漏。

### 4.3 Weibo（仅保留历史结果与脚本，不作本轮主线）
- 位置：`datasets/weibo*`；历史运行在 `paper_results/weibo_*` 中有汇总表与多seed结果。主线不再对Twitter与Weibo做大规模新实验（避免精力分散）。

---

## 5. 实验与结果（可直接引用/制图）

说明：所有单次运行目录均含：`config.yaml`、`summary.json`（含 train/val/test 指标）、`train_metrics.csv`、`val_metrics.csv`、`test_predictions.csv`（若评估时导出）。

### 5.1 WeFEND（主数据集）

- 教师/未蒸馏基线（uncertainty_fusion_v2）
  - 示例：`paper_results/wefend/wefend_uncertainty_fusion_wcls_seed00/summary.json`
  - test：acc≈0.9315 / macro‑F1≈0.8987 / pos‑F1≈0.8410 / ECE≈0.0569（seed=0 例）

- 事件可靠度门控（event_reliable）
  - 路径：`paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2/wefend_dynamic_distill_teacher_wcls_event_reliable_seed00/summary.json`
  - test：acc≈0.9425 / macro‑F1≈0.9141 / pos‑F1≈0.8646 / ECE≈0.0414（seed=0 例）

- 多配置对比（均 seed=0 样例）
  - `..._event_reliable_uncertainty`：0.9348 / 0.9041 / 0.8499（覆盖率偏低）。
  - `..._studentfocus`：0.9392 / 0.9082 / 0.8549（触发仍稀缺）。
  - `..._curriculum`：0.9403 / 0.9108 / 0.8594（较稳但未超老师+1.5pt）。
  - `..._reweight`：0.9348 / 0.9034 / 0.8483（仅重加权不足）。
  - `..._hybrid`：0.9315 / 0.8975 / 0.8385（不稳定，早停）。
  - `..._fallback`：首版 0.9337 / 0.9027 / 0.8477，安全版 `fallback_soft` 0.9392 / 0.9086 / 0.8556（KL受控，验证更稳但仍未达 +1.5pt）。

- 跨seed汇总（示例表）：`paper_results/tables/fakeddit_wefend_summary.csv`
  - WeFEND baseline（mean±std）：acc 0.9285±0.0032，macro‑F1 0.8929±0.0024。
  - WeFEND dynamic_distill（mean±std）：acc 0.9359±0.0016，macro‑F1 0.9047±0.0034。

结论（WeFEND）：事件可靠度门控最有潜力；安全fallback可作为覆盖率兜底，但需更细粒度/置信差自适应以避免噪声放大。整体领先幅度仍需提升至 ≥+1.5pt（论文主目标）。

### 5.2 Fakeddit（主补充数据集）
- 子集快速跑（20k/5k/5k）
  - baseline BERT+ViT：acc≈0.82，macro‑F1≈0.819。
  - 动态蒸馏（长训+多教师）最优约 acc≈0.846（+1.7pt 左右）。
  - 运行目录：`paper_results/fakeddit*`（含 teacher、安全方法 SAFE、EANN 对比）。

### 5.3 Weibo（历史）
- 多seed均值 acc≈0.9215，ECE≈0.0577（见 `paper_results/weibo_*`）。

---

## 6. 资源对照表（论文“数据与代码可用性”）

- 代码入口
  - 主训练：`dynamic_distill/scripts/train_mvp.py`（`--config` 加载YAML，`--disable-distill` 切换到基线）。
  - 模型：`dynamic_distill/src/models/*`，训练总控：`src/training/trainer.py`，损失：`src/losses/distillation.py`。
  - 数据准备：`dynamic_distill/data_prep/*`。
  - 评估/校准：`dynamic_distill/scripts/{evaluate_model,calibrate_temperature,calibrate_threshold}.py`。

- 数据与划分
- WeFEND：`datasets/wechat/processed_split/wefend_{train,val,test}.csv`；图像 `datasets/wechat/images/`。
- Fakeddit：`datasets/fakeddit/*`（脚本参见 `data_prep/`）。
- PHEME（新增，2025-11-05 下载）：`datasets/pheme/all-rnr-annotated-threads/`（事件→[rumours|non-rumours]→tweet tree）；原始归档保存在 `datasets/pheme/raw/PHEME_veracity.tar.bz2`，MD5=`11530d4c0c7127fc78bbc1e46f2498f8`。

- 结果与图表
  - 单次运行：`paper_results/<run_name>_seedXX/`，内含 `config.yaml`、`summary.json`、各阶段CSV 与预测。
  - 汇总表：`paper_results/tables/*.csv`；可视图：`paper_results/figures/*`。
  - WeFEND 失败经验：`dynamic_distill/experiments/wefend_failed_attempts.md`（按日期与策略枚举）。

---

## 7. 复现实验与命令（论文“可复现”小节可直接付诸）

注意：仅使用 GPU 2~6；`conda activate ddistill` 后运行。若首次运行需下载模型，确保网络畅通。

### 7.1 WeFEND 基线（未蒸馏教师）
```
CUDA_VISIBLE_DEVICES=2 \
python dynamic_distill/scripts/train_mvp.py \
  --config dynamic_distill/configs/wefend_uncertainty_fusion_v2.yaml \
  --seed 0 --progress
```

### 7.2 WeFEND 事件可靠度门控（当前最有潜力）
```
CUDA_VISIBLE_DEVICES=2 \
python dynamic_distill/scripts/train_mvp.py \
  --config dynamic_distill/configs/wefend_dynamic_distill_teacher_wcls_event_reliable.yaml \
  --seed 0 --progress
```

### 7.3 WeFEND 安全Fallback（覆盖不足兜底）
```
CUDA_VISIBLE_DEVICES=4 \
python dynamic_distill/scripts/train_mvp.py \
  --config dynamic_distill/configs/wefend_dynamic_distill_teacher_wcls_event_reliable_fallback_soft.yaml \
  --seed 0 --progress
```

### 7.4 Fakeddit（快速子集）
```
CUDA_VISIBLE_DEVICES=6 \
python dynamic_distill/scripts/train_mvp.py \
  --config dynamic_distill/configs/fakeddit_dynamic_distill_teacher.yaml \
  --seed 0 --progress
```

### 7.5 评估与导出预测
```
python dynamic_distill/scripts/evaluate_model.py \
  --config <同训练配置> \
  --seed 0
```

### 7.6 后处理校准（可选）
```
python dynamic_distill/scripts/calibrate_temperature.py --run <run_dir>
python dynamic_distill/scripts/calibrate_threshold.py   --run <run_dir>
```

---

## 8. 论文写作建议与材料组织

### 8.1 论文结构建议
- 摘要：问题场景（多模态虚假信息/内容安全）、数据泄漏风险、我们提出事件可靠度与不确定性交织的动态蒸馏、主要结果（Fakeddit +1.7pt、WeFEND 正在逼近 +1.5pt 目标）、可复现与开源。
- 相关工作：多模态虚假新闻（EANN/SAFE 等）、知识蒸馏、不确定性学习与校准、事件/账号划分的必要性。
- 方法：
  - 动态教师选择（不确定性差+δ课程）
  - 事件可靠度门控与正类事件门控
  - 置信与一致性门控、融合→模态辅蒸馏
  - 不确定性加权、事件/学生聚焦与重加权
  - fallback 软蒸馏（安全阈）
- 数据与协议：事件/账号感知划分、无泄漏原则、评测指标（Acc/Macro‑F1/Pos‑F1/ECE）。
- 实验：
  - 主结果（表1 WeFEND/Fakeddit 对比）
  - 消融与触发覆盖率可视化（建议绘制 `num_pairs/batch` 与事件分布）
  - 校准曲线（ECE、温度/阈值）
  - 失败经验（附录）：`wefend_failed_attempts.md` 小结。
- 结论与限制：WeFEND 仍需提升 ≥+1.5pt；fallback 在教师噪声时会放大损失；未来工作含自适应权重上限与更稳健的事件可靠度估计。

### 8.2 图表与附录素材
- 主表：`paper_results/tables/fakeddit_wefend_summary.csv`（均值±方差）。
- 单跑曲线：各 `*_seedXX/val_metrics.csv`（可绘制 val_f1 / val_ece vs epoch）。
- 校准曲线：`*_temp_calibration.json` 与 `*_threshold_calibration.json`。
- 失败经验摘录：`dynamic_distill/experiments/wefend_failed_attempts.md`。

---

## 9. 现存问题清单（供写作时如实陈述）

1) WeFEND 的优势尚未稳定达到 +1.5pt；最佳配置为 `event_reliable`，fallback_soft 改善波动但提升有限。  
2) 蒸馏覆盖率与教师噪声存在张力：门槛严格→覆盖低；放宽→KL 放大且易不稳。  
3) 需要补齐跨seed统计与显著性检验（bootstrap/AR test）以满足顶会严苛性。  
4) 需要补充“融合→模态蒸馏”与“学生聚焦/事件重加权”的联合消融图，展示各模块的边际收益。  

---

## 10. 快速FAQ（给写作代理）

- Q：最应该优先写哪种方法变体？
  - A：`event_reliable`（事件可靠度门控），它是目前最稳、最有说服力的改进来源；其次以 `fallback_soft` 作为覆盖兜底的补充实验。

- Q：论文中如何描述“无泄漏划分”？
  - A：WeFEND 以公众号为事件ID，严格事件不交叉；Fakeddit 以 submission/story 为单位去重分割；文本与图像双重去重。

- Q：可用图表有哪些？
  - A：`paper_results/figures/` 与各运行 `val_metrics.csv` 可快速绘图；汇总表在 `tables/` 下。

- Q：如何一键找回某次配置与结果？
  - A：进入对应 `paper_results/<run_name>_seedXX/`，查 `config.yaml` 与 `summary.json`。写作时将配置段直接摘入附录即可。

---

（完）
