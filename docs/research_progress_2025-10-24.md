# 2025-10-24 研究进展记录

## 数据快照
- 监控管线已连续运行 9 轮，`posts_status_all.csv` 共 2,682 条帖子，其中 1,497 条具备可用人脸裁剪（筛选比例 55.8%），对应条目写入 `posts_status_faces.csv`。
- 人脸子集互动显著高于全量：score 中位数 229 vs. 213，num_comments 中位数 28 vs. 27，score>100 占比 65.3% vs. 63.8%。
- 子集集中于高传播子版块：`news`（score 中位 4,667）、`CombatFootage`（402）、`technology`（340）、`worldnews`（200）、`politics`（187）。
- 最新 20 条记录覆盖从爆款（score>1,000、comments>100）到边缘低互动样本，平均可用人脸裁剪 1.02 张/帖。
- watcher 新增 `--max-post-age-hours`（默认 6h）过滤器，后续运行仅跟踪新近发布的帖子，为首小时传播特征留出观测空间。

## 传播特征抽取
- 通过脚本 [p2p/analysis/compute_face_propagation.py](../p2p/analysis/compute_face_propagation.py) 从 `propagation_timeseries.csv` 生成 `face_propagation_features.csv`，涵盖 1,497 条人脸帖子的传播动力学：整体增速（score/comments per hour）、首条评论延迟、以及 5/15/60 分钟窗口（目前因采样起点滞后而未命中，`has_window_* = 0`）。
- 中位 score_growth_per_hour ≈ 0.094，说明在被监测到时多数帖子增长已趋稳；首条评论出现时间中位数约 107 小时，提示需要更早触发 watcher 才能记录早期互动。
- 增速排名前列的帖子集中于 `politics`/`worldnews`（每小时增分 > 80），验证“高风险政治语境 + 人脸内容”是重点监测对象。

## 研究方向进展
- **深度不确定性融合**：已有数据支撑在 Ĉ/Ŝ/P̂ 三路上叠加轻量注意层；首小时的 `early_score`、`score_growth_per_hour` 等指标可作为传播子模态的动态特征，配合人脸质量（face_crops 数量）、溯源标签构建 Dirichlet 证据输入。
- **目标强结论**：初步量化结果表明，把焦点放在人脸帖子可显著降低误判风险。下一步将用 `face_propagation_features.csv` + 溯源表训练/比较（1）传统加权融合；（2）Dirichlet 证据融合；验证“误判率减半 + 自动弃权”。
- **合规落地**：保持 Reddit 只读监测，后续模型训练在 DFDC/FaceForensics++/油管公开集上完成，再迁移到当前人脸样本的推理与策略评估，符合知识库中的隐私与版权原则。

## TODO（按优先级）
1. **传播建模**：已补充 5/15/60 分钟窗口及评论延迟特征，下一步需优化 watcher 启动/采样频率以覆盖真实首小时数据，并追加爆发度指标 `burst_z`。
2. **证据融合实验**：实现 Dirichlet 证据头 + 轻量注意融合；对比传统 sigmoid 融合的校准曲线、弃权触发率与误伤率。
3. **策略评估**：将早期传播阈值与弃权策略输入 OPE 模块，生成“收益-代价前沿”图表；挑选若干高风险案例撰写定性分析，用于论文/汇报。
