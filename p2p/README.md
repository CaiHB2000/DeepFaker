# p2p 模块使用手册

## 目录概览

```
p2p/
├── runners/
│   ├── run_reddit_seed.py          # Reddit UGC 抓取
│   ├── run_wikimedia_feed.py       # Wikimedia Commons 策展数据
│   ├── run_flickr_feed.py          # Flickr CC 照片
│   └── run_dfbench_capsule.py 等   # DeepFakeBench 相关脚本
├── tools/
│   ├── aggregate_posts.py          # 聚合抓取元数据
│   ├── analyze_correlations.py     # 指标关联分析
│   ├── dedupe_cluster.py           # PDQ/vPDQ 去重
│   ├── quality_filters.py          # 质量筛选
│   ├── join_dfbench_scores.py      # DFBench 结果回写
│   └── watch_propagation.py        # 传播监视
└── provenance/
    └── fingerprint.py, probes/     # C2PA/PDQ 指纹分析
```

## 常用流程

### 1. 数据抓取 (Runners)

| Runner | 适用场景 | 关键参数 |
| --- | --- | --- |
| `run_reddit_seed.py` | Reddit UGC 内容 (新闻/生活) | `--subs`, `--min-score`, `--allow-post-hints`, `--require-media` |
| `run_wikimedia_feed.py` | Wikimedia Commons 策展照片 | `--categories`, `--start-date`, `--limit`, `--download-media` |
| `run_flickr_feed.py` | Flickr CC 照片 (新闻/纪实) | `--tags`, `--licenses`, `--min-width`, `--limit` |

运行示例（Wikimedia）：
```
pandoc tmp/report_analysis.md -o tmp/report_analysis.pdf --pdf-engine=wkhtmltopdf --pdf-engine-opt=--enable-local-file-access --metadata title="三源联合检测数据分析报告"
```

### 2. 人脸裁剪 (可选)

如使用 DeepFakeBench 人脸检测模型，建议先运行人脸提取：
```
python -m p2p.tools.filter_faces   --work-dir tmp/<source>   --out-kept kept_content_faces.csv   --face-dir face_crops
```
- 输出 `face_crops/`，`kept_content_faces.csv`，`no_face_content.csv`。
- 后续步骤可将 `--kept-csv` 指向 `kept_content_faces.csv`，仅对有人脸的样本进行检测。

### 3. 指纹、聚合与筛选


抓取完成后，按以下顺序运行：
```
python -m p2p.runners.run_media_provenance   --manifest tmp/<source>/media_manifest.csv   --media-root tmp/<source>/media   --out-provenance tmp/<source>/provenance_media.csv   --emit-jsonl

python -m p2p.tools.aggregate_posts   --posts_jsonl tmp/<source>/reddit_posts.jsonl   --manifest tmp/<source>/media_manifest.csv   --prov tmp/<source>/provenance_media.csv   --out_dir tmp/<source>   --media-root tmp/<source>/media

python -m p2p.tools.dedupe_cluster   --inventory tmp/<source>/media_inventory.csv   --provenance-jsonl tmp/<source>/provenance_media.jsonl   --out_dir tmp/<source>

python -m p2p.tools.quality_filters   --inventory tmp/<source>/media_inventory.csv   --content-map tmp/<source>/content_id_map.csv   --canonical tmp/<source>/content_canonical.csv   --media-root tmp/<source>   --out_dir tmp/<source>
```
输出包括 `kept_content.csv`、`posts_summary.csv` 等。

### 4. DeepFakeBench 多检测器评估

```
python -m p2p.runners.run_dfbench_multi   --work-dir tmp/<source>   --dfb-root DeepFakeBench   --detectors all   --prob-threshold 0.6   --min-positive-detectors 3
```
生成 `dfbench/dfbench_multi_content_scores.csv`、`posts_summary_with_dfbench_multi.csv`。

### 5. 指标分析

```
python -m p2p.tools.join_dfbench_scores --posts-summary tmp/<source>/posts_summary.csv   --content-map tmp/<source>/content_id_map.csv   --content-scores tmp/<source>/dfbench/dfbench_multi_content_scores.csv   --out-path tmp/<source>/posts_summary_with_dfbench_multi.csv

python -m p2p.tools.analyze_correlations   --posts-summary tmp/<source>/posts_summary_with_dfbench_multi.csv   --provenance tmp/<source>/provenance_media.csv   --inventory tmp/<source>/media_inventory.csv   --propagation tmp/<source>/propagation_features.csv   --out-dir tmp/<source>/analysis
```
分析结果位于 `tmp/<source>/analysis/`。

### 6. 传播监视 (Reddit)

```
python -m p2p.tools.watch_propagation   --posts-summary tmp/reddit_seed_realnews/posts_summary_with_dfbench_multi.csv   --out-csv tmp/reddit_seed_realnews/propagation_timeseries.csv   --status-json tmp/reddit_seed_realnews/propagation_watch_status.json   --interval 300
```
累积的传播快照可用于 CAPR 指标计算。

## 指标与建模建议

- **CAPR (Conflict-Aware Propagation Risk)**：`CI × (Δscore/Δt)`，用于传播预警。回测数据见 `tmp/analysis_capr_metrics.csv`。
- **SAC (Source-Aware Calibration)**：依据数据来源设置权重调节判假阈值，示例结果见 `tmp/sac_adjustment_summary.csv`。

## 常见输出

| 文件 | 说明 |
| --- | --- |
| `posts_summary.csv` | 聚合后的帖子元数据 |
| `kept_content.csv` | 质量筛选后的内容清单 |
| `dfbench_multi_content_scores.csv` | 多检测器概率、判假及冲突度 |
| `posts_with_features.csv` | 帖子 + 检测 + 指纹特征汇总 |
| `analysis/` | CAPR、SAC、传播四分位统计等 |

## 维护建议

1. 对新增数据源（Flickr/GDELT 等）沿用同一目录结构：`tmp/<source>/` 下包含抓取、指纹、质量、分析组件。
2. 所有分析脚本输出 MarkDown 和图表放置在 `tmp/report_figures/`，生成总结报告 `tmp/report_analysis.md`。
3. 根据 CAPR/SAC 指标加强模型设计，并在论文中附上初步实验结果与案例。

