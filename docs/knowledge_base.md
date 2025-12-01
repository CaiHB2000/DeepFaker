# DeepFaker 学术知识库

## 目标
- 为团队提供围绕多模态假新闻检测、传播建模与不确定性融合的快速参考。
- 关联仓库内的设计笔记与外部论文，支撑方案落地与学术写作。

## 快速索引
- [README.md](./README.md) — 文档索引与导航
- [s1.pdf](./s1.pdf) — *Towards real-world multimodal propagation networks: Multimodal uncertainty graph contrastive learning for fake news detection*（MUGCL，2025，JIIS）
- [s2.pdf](./s2.pdf) — *Assess and Guide: Multi-modal Fake News Detection via Decision Uncertainty*（A&G，2024，ACM MIS Workshop）
- [SomeIdea.md](./SomeIdea.md) — 项目路线图与可执行改造清单
- [research_progress_2025-10-24.md](./research_progress_2025-10-24.md) — 最新运行与分析快照

## 核心论文速览

### MUGCL（s1.pdf）
- 将文本、视觉各自的传播链建模，并在对比学习框架下进行跨链交互，加强多模态与传播之间的互补性。
- 引入不确定性感知的对比损失，对不同模态和传播分支的置信度进行自适应加权。
- 数据集设置更贴近真实世界，强调广泛传播情境下的鲁棒性与泛化能力。

**项目启发**
- 支撑 `p2p/propagation/` 中“早期传播建模+对比学习”方向，可复用其双传播链抽样策略。
- 为 `p2p/aggregator/` 的不确定性融合提供理论依据，尤其是置信度加权与弃权策略。

### Assess and Guide（s2.pdf）
- 基于 Dirichlet 分布的证据理论建模各模态分类器输出，显式分离置信度与不确定性。
- 在融合阶段通过不确定性惩罚“信息失衡”的模态，缓解模态冲突带来的误判。
- 实验覆盖 Weibo/Twitter，提供多模态不确定性估计的实证基线。

**项目启发**
- 为 `p2p/aggregator/fusion_uncertainty.py` 中“方差加权 sigmoid 融合器”提供替代实现参考（Dirichlet 证据推理）。
- 可指导 `DeepFakeBench` 端 Detector/Feature 输出来区分 `evidence`, `belief`, `uncertainty` 三类量。

## 与仓库文档的联动
- [SomeIdea.md](./SomeIdea.md) 已将上述论文洞见映射到六大工作包（不确定性融合、溯源增强、传播建模、缺失模态鲁棒、知识对齐、策略评估）。
- `p2p_pipeline_overview.md`、`propagation_tracker.md` 提供管线与追踪指标，可与论文中的指标（如传播链长度、模态不确定度）对齐。

## 使用建议
- 写作或做学术汇报时，先查阅 `SomeIdea.md` 获取仓库内的任务落点，再回到对应论文章节获取理论支撑。
- 若实现新模块，记录与论文方法对齐的假设、指标和实验设置，便于日后迭代与投稿复现。
- 按需补充新的论文或实验报告，可在本文件新增条目并在 `SomeIdea.md` 引用，维持知识库的一致性。

## 近期研究进展（2025-10-24）
- **深度不确定性融合**：在“三源”（内容 Ĉ、溯源 Ŝ、传播 P̂）不确定性表征上叠加轻量深度层（多头注意或小型 GNN），学习非线性互补关系，同时保留 Dirichlet 证据输出的可解释置信度；可先在公开平台（YouTube、新闻图库、DFDC、FaceForensics++ 等）预训练，再迁移到只读监测场景以规避 Reddit 版权限制。
- **数据合规策略**：仅使用具公开授权的数据源；对含人脸的样本做脱敏（裁剪、模糊或向量化特征）；公开模型/统计而非原始帧，确保隐私与版权合规。
- **传播链 + 人脸特征切入点**：借 MUGCL 的双传播链，对人脸媒体构建“视觉传播链”，并结合 Assess & Guide 的模态不确定性导向，分析“传播-内容-溯源”三源交互如何影响早期识别与弃权触发。
- **目标强结论（拟）**：“我们识别出人脸假新闻的关键在于‘传播-内容-溯源’三源信息的不确定性交互，提出的深度证据融合器能够在真实平台的早期阶段将误判率压低至传统融合的一半，同时对高风险案例自动触发弃权。” 后续实验应围绕该论断提供量化支撑。

## 运行快照（2025-10-24 晚间）
- watch_reddit_faces 连续运行 9 轮后，`posts_status_all.csv` 共收录 2,682 条帖子，其中 1,497 条具备可用人脸裁剪（手动筛选比例约 55.8%），全部写入 `posts_status_faces.csv` 做后续建模。
- 人脸子集在互动指标上明显占优：得分中位数 229（全量 213）、评论数中位数 28（全量 27）、score>100 占比 65.3%（全量 63.8%），说明“含人脸”样本自然强化了高传播段。
- 子集用户分布偏向高关注度板块：`news`（中位分 4,667）、`CombatFootage`（402）、`technology`（340）、`worldnews`（200）等，提供应用场景对齐的语料池。
- 最新 20 条跟踪帖子中既有超高爆款（score>1,000、评论>100），也捕捉到低互动但包含真实人脸的边缘案例，平均可用面部裁剪约 1.0 张/帖，为后续“人脸质量 → 传播”耦合实验提供输入特征。
- 状态文件 `state/watch_reddit_faces.json` 同步保存 tracked/all 双维度，默认只刷新人脸帖子，保证 API/算力消耗集中在研究核心样本。
- watcher 新增 `--max-post-age-hours`（默认 6h）过滤器，仅跟踪新近帖子，为 5/15/60 分钟传播窗口采样创造条件。
- [p2p/analysis/compute_face_propagation.py](../p2p/analysis/compute_face_propagation.py) 可生成 `face_propagation_features.csv`：含整体增速、首条评论延迟以及 5/15/60 分钟窗口可用性（当前波次尚未捕获首小时，`has_window_* = 0`，需提升采样及时性）。

## 实验筹备：Dirichlet 融合
- 输入准备：使用 `face_propagation_features.csv`（传播特征）、`posts_status_faces.csv`（互动基线）与溯源数据（待接入 `provenance_media.csv`）构成 Ĉ/Ŝ/P̂ 三路特征；可引入启发式标签（如 score>1000 或评论>100）作为早期“高风险” proxy。
- 融合框架：构建 Dirichlet 证据头，将每一路输出拆解为 `alpha = evidence + 1`（K=2 情况下可视为 Beta 分布）；采用轻量注意层学习跨路互补后输出融合 belief/uncertainty，并保留 `abstain` 阈值。
- 评估指标：可靠度曲线 (ECE)、风险-覆盖度、弃权触发率；结合 `face_propagation_features.csv` 中的 `score_growth_per_hour` 与 `first_comment_delay_hours` 进行错误分析，验证“误判率减半 + 自动弃权”目标。
