# 传播监控与特征提取指南（更新 2025-10-24）

P2P 模块提供两类工具：
1. 传统抓取 → `track_propagation` / `watch_propagation`；
2. 新版“人脸优先”Watcher → `watch_reddit_faces`（同时输出人脸裁剪与传播轨迹）。

以下说明涵盖环境准备、命令示例、特征抽取及自动化建议。

---

## 1. 环境准备
- Reddit API：
  ```bash
  export REDDIT_CLIENT_ID=xxxx
  export REDDIT_CLIENT_SECRET=yyyy
  export REDDIT_USER_AGENT="deepfaker-p2p/0.1"
  ```
- 推荐为不同流程使用独立目录：
  - `tmp/reddit_slowlane`：遍历历史帖子；
  - `tmp/reddit_fastlane`：仅监控新帖（≤1 h），快速捕获前 60 分钟数据。

---

## 2. 快速监控新帖（`watch_reddit_faces`）
```bash
python -m p2p.tools.watch_reddit_faces \
  --out-dir tmp/reddit_fastlane \
  --subs pics,news,worldnews \
  --max-post-age-hours 1 \
  --max-refresh-batch 40 \
  --interval 120 \
  --bootstrap-existing
```

产物：
- `posts_status_all.csv` / `posts_status_faces.csv`：帖子摘要、互动指标、是否含人脸；
- `media/` & `face_crops/`：原始媒体与 256×256 人脸裁剪（RetinaFace 优先）；
- `propagation_timeseries.csv`：持续追加的时间序列；
- `state/watch_reddit_faces.json`：保存 tracked/all 双集合与下一次刷新时间。

> Watcher 自动根据帖子“年龄”设定刷新频率：新帖约 180s/次，24 小时后逐渐降频至 4 小时/次。`--max-refresh-batch` 可限制单轮刷新条目，保证新帖优先。

---

## 3. 传统单次/慢速刷新
- 单次抓取：
  ```bash
  python -m p2p.tools.track_propagation \
    --posts-jsonl tmp/reddit_seed_run_mass/reddit_posts.jsonl \
    --out-csv tmp/reddit_seed_run_mass/propagation_timeseries.csv
  ```
- 持续轮询（兼容旧流程）：
  ```bash
  python -m p2p.tools.watch_propagation \
    --posts-jsonl tmp/reddit_seed_run_mass/reddit_posts.jsonl \
    --out-csv tmp/reddit_seed_run_mass/propagation_timeseries.csv \
    --interval 900 --jitter 60 --per-request-sleep 1.0 --refresh-targets
  ```
  主要参数与说明保持不变，适合对历史列表做低频刷新。

---

## 4. 生成传播特征
新版脚本会从 Watcher 输出中直接计算动态指标。
```bash
python -m p2p.analysis.compute_face_propagation \
  --watch-dir tmp/reddit_fastlane \
  --out-csv tmp/reddit_fastlane/face_propagation_features.csv
```

字段示例：
- `score_growth_per_hour` / `comments_growth_per_hour`：整体增速；
- `first_comment_delay_hours`：首条评论出现时间；
- `has_window_{5m,15m,60m,6h}`：时间窗口是否命中（fastlane 运行数小时后会填充数值）；
- `score_5m`, `score_15m`, `score_60m`：窗口末尾得分（需首小时数据）；
- `points_6h`：6 小时内采样点数量。

旧工具 `p2p.tools.propagation_features` 仍可用于历史数据，但推荐逐步迁移到上述脚本。

---

## 5. 可视化与分析
- 查看单帖时间序列：
  ```bash
  python -m p2p.tools.plot_timeseries \
    --timeseries tmp/reddit_fastlane/propagation_timeseries.csv \
    --posting-id reddit:1oed8q9 \
    --out-dir tmp/reddit_fastlane/analysis/plots
  ```
- 联合分析（内容 + 溯源 + 传播）：
  ```bash
  python -m p2p.tools.analyze_correlations \
    --posts-summary tmp/reddit_fastlane/posts_summary_with_dfbench_multi.csv \
    --provenance tmp/reddit_fastlane/provenance_media.csv \
    --inventory tmp/reddit_fastlane/media_inventory.csv \
    --propagation tmp/reddit_fastlane/face_propagation_features.csv \
    --out-dir tmp/reddit_fastlane/analysis
  ```

---

## 6. 自动化建议
- 使用 `tmux` 或 `systemd --user` 持久运行 `watch_reddit_faces`；fastlane 与 slowlane 可开两个实例。
- 定时（例如每日）运行 `compute_face_propagation`、`analyze_correlations`，刷新图表与统计。
- 针对临近研究节点，可缩小 `--max-post-age-hours`（如 0.2 h）以重点捕获刚发布的帖子。
- `state/watch_reddit_faces.json` 可手动备份，实现断点恢复或迁移到其它机器。

---

## 7. 常见问题
1. **5/15/60 分钟窗口为空**：说明第一次快照距 `created_utc` 过久，请减少 `--max-post-age-hours` 并运行 fastlane watcher。
2. **人脸误判**：最新 `filter_faces.py` 已引入 RetinaFace；若历史数据仍包含风景误判，请删除旧的 `face_crops/` 与 `face_detection_summary.csv` 后重跑。
3. **API 限制**：保持 `--per-request-sleep ≥1s`，并设置 `REDDIT_USER_AGENT` 为唯一值；如遇 429，Watcher 会自动退避。

---

通过以上工具组合，可以在数小时内获得“新帖 → 人脸裁剪 → 传播曲线”的完整链路，为后续的不确定性融合与策略模拟提供可复现的特征数据。 若流程有新增功能（如队列调度、更多传播指标），请同步更新本指南与 `p2p/README.md`。
