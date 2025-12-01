# DPMD 数据集执行清单（Step-by-Step）

> 适用于个人研究者。每一级任务尽量独立，可完成后打勾并记录日志。

## Phase 0 — 准备
1. [ ] 创建专用目录结构 `datasets/dpmd_modalities/{events,raw_posts,views,labels,logs}`。
2. [ ] 申请/确认 API：Bing Web Search (或 SerpAPI)、事实核查 RSS、GPT-4o key。
3. [ ] 建立 `.env`（API key）、`configs/dataset.yaml`（平台开关、限速、路径）。

## Phase 1 — 事件池
1. [ ] `scripts/events/build_event_seed.py`：抓取 IFCN/Tencent RSS → `events.csv`。
2. [ ] `scripts/events/expand_keywords.py`：用 GPT 生成关键词、存 `events_keywords.json`.
3. [ ] 人工审阅前 50 个事件，剔除重复或无法公开的数据源。

## Phase 2 — 平台抓取
1. [ ] `scripts/collect/weibo_scrape.py`：输入 `events_keywords.json`，输出 `raw_posts/weibo/*.jsonl`。
2. [ ] `scripts/collect/wechat_fetch.py`：Playwright 抓公众号正文 + 截图。
3. [ ] `scripts/collect/video_snapshot.py`：对抖音/B站链接用 `yt-dlp` 拉封面 + 字幕。
4. [ ] `scripts/collect/reddit_mapper.py`（可选）：复用现有 Reddit 监控，将帖子按关键词映射到事件。
5. [ ] `scripts/collect/deduplicate.py`：对同事件文本/链接做 SimHash 去重。

## Phase 3 — 模态视图生成
1. [ ] `scripts/views/generate_full_bundle.py`：将下载到的原始内容整理入 `views/.../full_bundle`.
2. [ ] `scripts/views/detect_missing.py`：检测下载失败/平台屏蔽 → 自动生成 `observed_view` + `missing_reason`.
3. [ ] `scripts/views/simulate_dropout.py`：随机屏蔽模态，生成额外 `observed_view`（记录 `simulated=true`）。

## Phase 4 — 检索 & LLM 标注
1. [ ] `scripts/label/retrieve.py`：针对每条帖调用 Bing/Google Fact Check/反向图片，保存 `retrieval.jsonl`。
2. [ ] `scripts/label/prompt_llm.py`：调用 GPT-4o，多模态输入→输出结构化标签 (`labels_auto.jsonl`)。
3. [ ] 自动一致性检查脚本：比对 `fact_check_url` 与 LLM 输出，标记 `needs_review`.

## Phase 5 — 人工核查
1. [ ] `scripts/label/human_queue.py`：生成待人工列表（高风险/随机抽检）。
2. [ ] 进行人工核查，写入 `labels_human.jsonl`，包括理由、使用的证据。
3. [ ] 合并 `labels_auto` 与 `labels_human`，产出最终 `labels_final.jsonl`。

## Phase 6 — 数据清洗 & 发布
1. [ ] `scripts/export/validate_alignment.py`：确保 `events`, `posts`, `views`, `labels` ID 对齐。
2. [ ] `scripts/export/anonymize.py`：哈希用户标识、模糊敏感图像。
3. [ ] 生成 `balanced_train.jsonl` 与 `raw_eval.jsonl`。
4. [ ] 自动生成 `DATACARD.md`、`COLLECTION_LOG.md`、`LABELING_LOG.md`。
5. [ ] 打包 `v0.1` 版本，上传至 Zenodo/OSF，并撰写发布说明。

---

> 建议每完成一项，立即在 `logs/dataset_build/YYYYMMDD.md` 记录遇到的问题、耗时、改进建议，方便后续迭代。***
