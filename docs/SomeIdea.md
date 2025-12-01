太好了！我把你列的几类论文想法“拆成可落地的改造点”，并按你现有仓库结构（`p2p/`, `DeepFakeBench/`）给出**可执行的实现方案**与**本周能完成的小任务**。整体分 6 组：**不确定性融合、溯源增强、传播建模、缺失模态鲁棒、知识对齐扩展、评估与策略 OPE**。每组都包含：要改什么 → 放哪 → 怎么做 → 本周可做的实验与数据产出。

为了方便后续学术研究与写作，已经把核心论文与设计文档统一收录到 `docs/` 目录，并新增了知识库入口 `docs/knowledge_base.md`。可以先在知识库里查到论文概要与项目映射，再回到本路线图落实实现细节。

## 文献快照（更新于 2025-10-24）

- [s1.pdf](./s1.pdf)：MUGCL 框架强调“多模态传播链 + 不确定性对比学习”。对应这里的第 1 组和第 3 组，可直接复用其传播链构造与置信度加权思路。
- [s2.pdf](./s2.pdf)：Assess and Guide 使用 Dirichlet 证据理论做模态不确定性建模，为第 1 组的不确定性融合与第 4 组的缺失模态鲁棒提供可落地的基线。
- [knowledge_base.md](./knowledge_base.md)：整理了上述论文的重点结论、对仓库模块的启发以及使用建议，后续新增资料时请同步更新。

---

# 1) 不确定性感知的三源融合（借鉴 ECCV’22/AAAI’24 思路）

**目标**：把 Ĉ（内容一致性）、Ŝ（源可信）、P̂（传播）三源分数统一到一个**带不确定性**的融合器里，支持“弃权/人审”与策略阈值更稳。

**改哪里**

* 模块：`p2p/aggregator/`、`p2p/analysis/`
* 新表：`aggregate_scores.csv` 增加 `*_var`/`*_aleatoric`/`*_epistemic` 字段与 `abstain` 标记

**怎么做（简化版）**

* 对每个源输出 mean+var：

  * Ĉ：用 DeepFakeBench 多检测器（多头/多裁剪/多次前向）→ 估计方差；
  * Ŝ：对 provenance 子探针（PDQ, vPDQ, C2PA, TLS/RDAP 特征打分）做**子模型堆叠 + 温度缩放**得到 mean/var；
  * P̂：传播早期窗口（如 0–60min 分桶）用轻量模型（logistic/GBDT）输出概率与方差（可用 MC dropout/bootstrapping）。
* 融合器：
  [
  r = \sigma!\left(\alpha\cdot\frac{Ĉ}{\sqrt{Var_C + \epsilon}} + \beta\cdot\frac{Ŝ}{\sqrt{Var_S+\epsilon}} + \gamma\cdot\frac{P̂}{\sqrt{Var_P+\epsilon}}\right)
  ]
  并输出总不确定性 (U=) 方差加权和；若 (U) 超阈值 → `abstain=1`。

**本周可做**

* 在 `p2p/aggregator/fusion_uncertainty.py` 加一个**“方差加权 sigmoid 融合器”**；
* 在 `analysis/` 生成 `calibration_reliability.png`（可靠度曲线）与 `risk_vs_uncertainty.png`；
* 报告里展示“加不确定性后策略触发的误伤率下降”。

---

# 2) 溯源（Provenance）增强：签名 + 水印 + 稳定性（借鉴 RealSeal/C2PA）

**目标**：让 Ŝ 不再只是“有没有 C2PA/指纹”，而是**多探针证据融合**的稳健评分。

**改哪里**

* 模块：`p2p/provenance/`、`p2p/tools/quality_filters.py`
* 新表：`provenance_media.csv` 增 `c2pa_chain_depth, watermark_conf, tls_age_days, rdap_domain_age, phash_stability_*`

**怎么做**

* **C2PA**：解析 claim 链条（issuer/path/创建时间），做 **chain_depth** 与 **时间一致性检查**（媒体 EXIF/抓取时间/claim 时间是否合理）。
* **水印/可见度**：引入简单的盲水印探针（可先用占位符/第三方脚本），输出 `watermark_conf`。
* **TLS/RDAP**：补充 `tls_age_days`、`rdap_domain_age`、`issuer_cn` one-hot；
* **稳定性探针**：现有 pHash 稳定性（JPEG Q、模糊、色偏）已在跑 → 归一化后进 Ŝ 子模型。
* **Ŝ 学习器**：在 `p2p/provenance/model_prov.py` 加一个轻量 **Logit/GAM**，输出 `S_mean, S_var`（可用 bootstrap）。

**本周可做**

* 在 `run_media_provenance.py` 串接新字段，更新 `provenance_media.csv`；
* 做一张“Ŝ 与 Ĉ 的相关性热力图”（看看证据冲突的分布）。

---

# 3) 传播（Propagation）建模升级：早期特征 → Hawkes/Gompertz

**目标**：让 P̂ 不再是“静态分数”，而是**早期传播动力学**预测的概率与不确定性。

**改哪里**

* 模块：`p2p/propagation/`、`p2p/tools/aggregate_posts.py`
* 新表：`prop_timeseries.csv`（posting_id, t_bucket, views/likes/comments），`prop_features.csv`

**怎么做**

* **最小时序采样**：把 Reddit 采样到的互动量按 5/15/60min 桶聚合，补全缺失为 0；
* **早期特征**：增速、倍增时间 `doubling_time`、爆发度 `burst_z`、首小时留存率、首条评论延迟等；
* **拟合**：

  * 快速版：用早期 60min 特征 → 逻辑回归预测“是否将达到高传播阈值（如上四分位）”；
  * 研究版：加 Hawkes/Gompertz 拟合，取参数（基础强度/自激系数/增长率）作为 P̂ 特征。
* **不确定性**：bagging 多子样本或 MC dropout 得到 `P_mean, P_var`。

**本周可做**

* 新增 `p2p/propagation/feats.py` 提取特征并落盘；
* 画 `early_feature_importance.png` 与 `roc_propagation.png`。

---

# 4) 缺失模态鲁棒（借鉴 “incomplete modality” 论文）

**目标**：面对“没有图”“没有文本”“没有 claim/EXIF”“没有时序”等不完整数据，系统**不会崩**，还能给**置信度合理**的输出。

**改哪里**

* 模块：`p2p/aggregator/`、`p2p/tools/dedupe_cluster.py`
* 表：在 `*_scores.csv` 增 `mask_*` 字段（1=可用/0=缺失），融合器使用 **gating**。

**怎么做**

* **遮罩感知融合**：若某源缺失，用剩余源分数按**置信度重分配**；
* **知识蒸馏/教师-学生**（可选后续）：在 `DeepFakeBench` 端用完整样本训练教师，对不完整样本蒸馏到学生（提升 Ĉ 在缺失模态下的稳健性）。
* **评估**：分层报告（全模态 / 缺文本 / 缺媒体 / 缺时序）的 AUC/Calib。

**本周可做**

* 融合器里加入 `mask_C, mask_S, mask_P`；
* `analysis/` 出一张 **“分模态可用性分层的可靠度曲线”**。

---

# 5) 知识对齐与结构增强（借鉴 KGAlign）

**目标**：把内容一致性从“像素/文本相似”提升到**实体/事实结构一致**，尤其适合**标题-图像-实体**不一致型伪造。

**改哪里**

* 模块：`p2p/content/`、`DeepFakeBench/`
* 新表：`content_entities.csv`（posting_id, entities, rels, kb_hits）

**怎么做**

* **轻量版**：

  * 用现成 NER/实体链接（如 spaCy/flashtext + 词典）抽取文本实体；
  * 对图片用 CLIP + 物体标签器（轻量）生成候选标签；
  * 构造 **“标题实体 ↔ 图像标签”一致性分**（Jaccard/点互信息/CLIP 对齐分）；
* **结构一致性**：若能命中公共知识（如维基实体）则计算“事实组装一致性”（比如“城市-国家”是否匹配）。
* 把该分数作为 **Ĉ_knowledge** 子分支，并纳入融合器。

**本周可做**

* 先做文本实体 ↔ 图像标签的简单一致性分，落到 `dfbench/detector_outputs.csv` 的附加列 `content_kg_align`；
* 出一张“加 `content_kg_align` 前后，Ĉ AUC/Calib 的提升”图。

---

# 6) 评估与策略 OPE（离线策略效果评估）

**目标**：让“限流/标注/隔离”策略的收益-代价**可量化**，并能在报告中**说服**听众。

**改哪里**

* 模块：`p2p/policy/`、`p2p/analysis/`
* 新表：`policy_decision_log.csv`（阈值、触发时刻、弃权原因、不确定性）

**怎么做**

* **OPE 协议**：

  * 用早期时序拟合“无干预”扩散曲线（Hawkes/Gompertz），
  * 在 t*=策略触发阈值时刻套用“限流倍率 m”生成**反事实曲线**；
  * 统计总曝光减少 vs 误伤率、延迟触发成本。
* **报告**：产出“收益–代价前沿曲线”“策略触发分布（含弃权）”。
