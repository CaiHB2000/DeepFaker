# P2P 管线技术说明（2025-10-24）

## 1. 目标与愿景
- 面向真实社交平台，统一完成“采集 → 元数据 → 指纹/人脸 → 传播 → DeepFakeBench 检测 → 不确定性融合”的流程。
- 为多模态伪造识别提供可追溯的数据资产，支撑 MUGCL/Assess & Guide 式的三源不确定性研究与策略模拟。
- 在保持 Reddit 只读合规的前提下，后续可将训练迁移至 YouTube/微博/公开数据集。

## 2. 目录结构概览

| 目录 | 说明 |
| --- | --- |
| `p2p/runners/` | 各平台抓取脚本（`run_reddit_seed.py` 等） |
| `p2p/tools/` | 聚合、去重、质量筛选、传播监控、特征抽取等工具 |
| `p2p/analysis/` | 特征加工与评估脚本（如 `compute_face_propagation.py`） |
| `tmp/<source>/` | 单个数据源的全部中间产物、日志、图表 |
| `docs/` | 文档索引、研究进展、技术指南 |

> 推荐将不同采集策略（慢速全量 / 快速新帖）放在独立的 `tmp/` 子目录，避免互相覆盖。

## 3. 端到端流程

1. **帖子发现与下载**
   - 慢速：`python -m p2p.runners.run_reddit_seed --subs pics,news ...` 产生 `reddit_posts.jsonl`, `media_manifest.csv`。
   - 快速新帖监控：`python -m p2p.tools.watch_reddit_faces --out-dir tmp/reddit_fastlane --max-post-age-hours 1 --max-refresh-batch 40 --interval 120`。
     - Watcher 自动下载媒体、刷新互动指标，并根据年龄动态调度刷新频率（新帖 2–3 分钟，旧帖逐步延长）。
     - 产物：`posts_status_all.csv`, `posts_status_faces.csv`, `media/`, `face_crops/`, `state/watch_reddit_faces.json`。

2. **人脸检测与裁剪**
   - `p2p/tools/filter_faces.py` 现默认使用 **RetinaFace**（score ≥0.9、最短边 ≥60px），未命中时退回调高阈值的 Haar cascade（`minNeighbors≥6`、`minSize=80`）。
   - 输出：`face_crops/rdtC_xxx.png`（256×256）、`kept_content_faces.csv`、`face_detection_summary.csv`、`no_face_content.csv`。
   - 若需重新裁剪：`python -m p2p.tools.filter_faces --work-dir tmp/reddit_fastlane --out-kept kept_content_faces.csv`.

3. **指纹与溯源探测**
   - `python -m p2p.runners.run_media_provenance --work-dir tmp/<source>`。
   - 输出：`provenance_media.csv/.jsonl`（C2PA、PDQ/vPDQ、TLS/RDAP 等）。

4. **聚合与质量筛选**
   - `python -m p2p.tools.aggregate_posts --out-dir tmp/<source>` → `posts_summary.csv`, `media_inventory.csv`.
   - 去重：`python -m p2p.tools.dedupe_cluster --work-dir tmp/<source>` → `content_id_map.csv`, `content_canonical.csv`.
   - 质量过滤：`python -m p2p.tools.quality_filters` 生成 `kept_content.csv`。

5. **DeepFakeBench 检测**
   - `python -m p2p.runners.run_dfbench_multi --work-dir tmp/<source> --join-posts-summary posts_summary.csv`。
   - 输出：`dfbench/dfbench_multi_content_scores.csv`, `posts_summary_with_dfbench_multi.csv`。

6. **传播时间序列与特征**
   - 持续刷新：`watch_reddit_faces` 已在步骤 1 输出 `propagation_timeseries.csv`。
   - 特征抽取：`python -m p2p.analysis.compute_face_propagation --watch-dir tmp/<source>` → `face_propagation_features.csv`（整体增速、首评论延迟、时间窗口命中情况）。
   - 传统工具：`p2p.tools.propagation_features` / `p2p.tools.analyze_correlations` 仍可用于批量数据回放。

7. **分析与报告**
   - `python -m p2p.tools.analyze_correlations --posts-summary posts_summary_with_dfbench_multi.csv ...` 产出 `analysis/` 图表。
   - 文档与报告存于 `docs/`（如 `report_2025-10-24.md`、`research_progress_2025-10-24.md`）。

## 4. Watcher 调度策略

| 参数 | 说明 |
| --- | --- |
| `--max-post-age-hours` | 仅抓取创建时间距当前不超过指定小时的新帖，避免旧帖占用配额。 |
| `--max-refresh-batch` | 单轮最多刷新多少帖子，默认 300；可针对 fastlane 设置为几十。 |
| 动态刷新 | 每个帖子在 state 中维护 `next_refresh_utc`：新帖 ~180s，6–24 小时内 900s，24–72 小时 3600s，最终稳定为 4 小时。 |
| 状态持久化 | `state/watch_reddit_faces.json` 保存 tracked/all 双表，可随时中断/恢复。 |

> 建议同时运行两个实例：slow lane（完整队列）+ fast lane（新帖），并根据需要合并 `propagation_timeseries.csv` 与特征表。

## 5. 人脸裁剪注意事项

1. 若早期数据存在假阳性（风景被判做人脸），请删除旧的 `face_detection_summary.csv` 和 `face_crops/` 后重新运行 `filter_faces.py`。
2. RetinaFace 依赖 GPU 驱动/TensorFlow；若环境不支持会自动退回 Haar cascade。可在 `filter_faces.py` 中调整 `retinaface_score` 或 `min_size`。
3. 所有裁剪统一为 256×256 PNG，方便直接输入 DFBench 检测器。

## 6. 传播特征与不确定性融合

- `face_propagation_features.csv` 目前包含：整体增速（score/comments per hour）、首评论延迟、6 小时窗口命中情况。随着 fast lane 数据积累，将新增 5/15/60 分钟窗口数值，用作传播子模态特征。
- 三源不确定性融合（规划中）：
  1. 内容 Ĉ：DFBench 多检测器输出 + 置信度；
  2. 溯源 Ŝ：`provenance_media.csv` 中的 C2PA/指纹/TLS 字段；
  3. 传播 P̂：`face_propagation_features.csv` 中的增速、窗口特征、评论延迟。
- 目标是在 `p2p/aggregator/fusion_uncertainty.py` 引入 Dirichlet 证据头 + 轻量注意机制，输出风险评分与弃权策略。

## 7. 维护规范

- 每个数据源使用独立的 `tmp/<source>/`，保持文件命名一致；运行脚本时总是显式设置 `--work-dir`/`--out-dir`。
- 重大变更（例如引入 RetinaFace 或 Watcher 逻辑）须同步更新 `docs/` 与 `p2p/README.md`。
- 定期归档 `analysis/` 图表、`posts_status*.csv`、`face_propagation_features.csv`，作为实验重复与论文附录的基础。
- 训练数据与只读监测严格区分：Reddit 数据仅用于分析，不入训练集。

## 8. 快速命令清单

```bash
# 1. 快速监控新帖
python -m p2p.tools.watch_reddit_faces --out-dir tmp/reddit_fastlane \
  --subs pics,news,worldnews --max-post-age-hours 1 --max-refresh-batch 40 \
  --interval 120 --bootstrap-existing

# 2. 重裁剪人脸
python -m p2p.tools.filter_faces --work-dir tmp/reddit_fastlane \
  --out-kept kept_content_faces.csv

# 3. 更新传播特征
python -m p2p.analysis.compute_face_propagation --watch-dir tmp/reddit_fastlane

# 4. 运行 DFBench 检测
python -m p2p.runners.run_dfbench_multi --work-dir tmp/reddit_fastlane \
  --join-posts-summary tmp/reddit_fastlane/posts_summary.csv

# 5. 综合分析
python -m p2p.tools.analyze_correlations \
  --posts-summary tmp/reddit_fastlane/posts_summary_with_dfbench_multi.csv \
  --provenance tmp/reddit_fastlane/provenance_media.csv \
  --inventory tmp/reddit_fastlane/media_inventory.csv \
  --propagation tmp/reddit_fastlane/face_propagation_features.csv \
  --out-dir tmp/reddit_fastlane/analysis
```

## 9. 后续计划

- **传播窗口补齐**：fast lane 运行数小时后验证 5/15/60 分钟字段是否填充，必要时进一步降低 `max_post_age_hours` 或改用 `first_seen_utc` 判断年龄。
- **不确定性融合实验**：基于更新后的三源特征，搭建 Dirichlet 证据模型并评估风险-覆盖曲线。
- **策略模拟**：将融合模型输出接入离线传播模拟（OPE），给出“误判率减半 + 自动弃权”数据证据。

> 若发现文档与代码不一致，请同步更新 `docs/` 与对应脚本，确保团队成员能按此说明复现实验。
