# 数据采集与标注流水线（面向个人研究者）

_适用资源：公开平台 API/爬虫、自动化脚本、GPT API、有限人工核查_

## 1. 事件驱动采集

1. **事件池构建**
   - 每日从事实核查 RSS（IFCN、腾讯较真、辨真）拉条目，写入 `events.csv`（字段：`event_id`, `title`, `keywords`, `fact_check_url`, `category`, `language`, `published_at`）。
   - 结合新闻 API（Bing News、360 搜索）抓取正确信息，记录 `reference_urls`。
2. **关键词派生**
   - 自动生成 5–10 个关键词组合（原文关键词 + 同义词 + 表情/俗称）；可调用 GPT 生成候选，再人工删选。

## 2. 平台采集模块

| 平台 | 工具 | 核心要点 |
| --- | --- | --- |
| 微博 | `snscrape` + 关键词 + 时间窗 | 输出 JSON 行，保留贴link、文本、媒体 URL、转发/评论数。 |
| 公众号 | `wechat-articles-spider` 或 Playwright | 抓正文、封面图、阅读量；同步保存 PDF 截图。 |
| 短视频/图文平台（抖音/快手/小红书/B站） | `yt-dlp` (封面/字幕) + Selenium 截图 | 若无法下载视频，至少截图 + 描述文本。 |
| Reddit/Telegram 等 | 若已有抓取进程，可将结果映射到事件关键词，用作多语对照或噪声负样本。 |

输出统一写入 `raw_posts/{platform}/{event_id}.jsonl`。

## 3. 模态生命周期生成

1. **full_bundle**：采集时保存所有模态（文本、图像、视频文件或下载链接、外链 HTML）。  
2. **observed_view 采样**：
   - 自动检测：若某模态下载失败（HTTP 非 200、文件缺失），记录 `missing_reason` 并生成 `observed_view` 条目。
   - 人工模拟：使用脚本随机“丢模态”或转换为截图，仅保留文本/封面，用以模拟用户分享时的劣化。
   - 平台转发链：抓取转发/引用内容（比如微博长截图），归为额外 `observed_view`。
3. **证据存档**：所有视图写入 `views/{event_id}/{post_id}/{view_id}.{json|png|mp4}`，并保存 Sha256，方便后续验证。

## 4. 自动化标注

1. **检索对齐**
   - 调用 Bing Web Search / Google Fact Check API，根据帖文本、图片（调用反向图片搜索 API）检索候选证据。
   - 写入 `retrieval.jsonl`（每条帖对应若干证据 URL + 片段摘要）。
2. **LLM 结构化判断**
   - 模板中包含：帖文本、OCR 结果（若只有截图）、媒体描述、检索证据摘要、事实核查原文。
   - 模型输出 JSON：`veracity`, `confidence`, `manipulation_types`, `evidence_used`, `needs_human_review`.
   - 推荐模型：GPT-4o（中英双语 + 图像）、Claude 3.5 Sonnet（图像+文本），成本可控；若需要纯本地可用 Qwen-VL/Qwen2.5-VL 做初筛。
3. **冲突检测**
   - 若 `confidence < 0.6` 或 `evidence_used` 为空 → 标记待人工。
   - 若自动标签与 `fact_check_url` 结论不一致 → 强制人工。

## 5. 人工核查策略

| 类型 | 触发条件 | 操作 |
| --- | --- | --- |
| 必查样本 | 公共安全/医疗/政治敏感关键词；LLM 低置信；检索缺失 | 手动阅读原帖 + 事实核查文章，确认标签并记录理由。 |
| 抽检样本 | 每事件随机抽 10% | 快速验证 LLM 输出；若发现系统误差，调整提示或重新标注整批。 |
| 争议样本 | LLM 给出 `partial` 或 `unknown` | 尝试追加证据，必要时标记为 `unknown` 并说明。 |

人工审核输出 `human_review.jsonl`，字段包括审核人、时间、结论、备注。

## 6. 数据清洗与发布

1. **脱敏**：哈希 `author_id`、裁剪敏感图片区域、移除私信。
2. **一致性校验**：脚本检查 `posts`、`views`、`labels` 三方 ID 对齐；缺失项写入 `validation_report.md`。
3. **生成平衡/原始子集**：
   - `balanced_train`: 事件内按真/假/部分配比抽样。
   - `raw_eval`: 保留真实分布，供鲁棒性评测。
4. **文档**：自动生成
   - `DATACARD.md`
   - `COLLECTION_LOG.md`（记录脚本、API key、调用频次）
   - `LABELING_LOG.md`（LLM 模型版本、费用、人工投入）

## 7. 自动化脚本清单（后续实现）

1. `scripts/events/build_event_seed.py`：拉取事实核查 RSS，更新 `events.csv`。
2. `scripts/collect/weibo_scrape.py`：根据 `events.csv` 自动抓取微博。
3. `scripts/collect/wechat_fetch.py`：抓公众号文章并截图。
4. `scripts/views/generate_observed.py`：根据下载失败或随机策略生成缺模视图。
5. `scripts/label/retrieve_and_prompt.py`：调用检索 + GPT API 输出结构化标签。
6. `scripts/label/human_review_helper.py`：输出待人工列表、记录复核结论。
7. `scripts/export/build_release.py`：汇总数据、生成 Data Card。

上述脚本可逐步实现，每一步都可以单独运行/调试，方便个人研究者按优先级推进。***
