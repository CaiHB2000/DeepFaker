# Modality-Lifecycle Fake News Dataset (Draft Spec)

_Last updated: 2025-11-18_

## 1. 目标与差异化亮点

| 项目 | 说明 |
| --- | --- |
| 核心用途 | 评估与训练在模态缺失、内容退化情况下仍需保持鲁棒性的虚假新闻检测模型（DPMD）。 |
| 主要受众 | 多模态谣言检测研究者、事实核查工具开发者，以及需要部署在真实社交平台的生产系统。 |
| 差异化亮点 | **模态生命周期标注**：对每条事件提供“完整视图”(full bundle) 与若干“真实可见视图”(observed views)，详细记录图像/视频/链接丢失或被替换的原因。现有公开数据集中尚无此类成对样本。 |

## 2. 数据单元设计

- **事件 (event_id)**：围绕同一谣言或辟谣主题的集合，来自事实核查平台或权威新闻。
  - 字段：`event_id`, `topic`, `language`, `seed_fact_check_url`, `time_span`.
- **帖子 (post_id)**：社交平台上的具体内容（文本/图像/视频/外链），每条属于一个事件。
  - 需要至少保存：`platform`, `author_hash`, `timestamp`, `raw_text`, `media_urls`, `engagement`。
- **视图 (view_id)**：同一帖子的不同“可见版本”。
  - 类型：`full_bundle`（采集当下完整模态）与 `observed_view`（平台转发或抓取时缺失模态）。
  - 元数据：
    - `modalities_available`: 例 `["text","image"]`
    - `missing_modalities`: 例 `["video"]`
    - `missing_reason`: `link_dead`, `platform_strip`, `user_deleted`, `screenshot_only`, 等。
    - `evidence_ref`: 快照截图/HTTP 状态/平台响应 JSON。

## 3. 标签体系

1. **真值标签**：`veracity ∈ {true, false, partial, unknown}`
2. **误导手法**（可多选）：`old_image`, `fabricated_quote`, `ai_generated_visual`, `context_omission`, `satire_misused`, …
3. **证据引用**：至少包含一个事实核查或权威报道 URL。
4. **模态可用性标签**：见第 2 节。

> 自动化建议：先由 GPT-4o/Claude 完成结构化判断，再由人工抽检并记录 `human_verified ∈ {none, sampled, full}`。

## 4. 采集范围（MVP）

| 平台 | 访问方式 | 备注 |
| --- | --- | --- |
| 微博公开帖 | snscrape / 官方 API | 关键词=事件标题；需去除敏感个人信息。 |
| 公众号文章 | 手动或半自动抓取 | 保存正文 + 封面，注意版权。 |
| 抖音/快手截屏 | 浏览器自动化 + 截图 | 视频原文件可选，至少保存截图哈希。 |
| B 站稿件 | `yt-dlp` 抓取封面与字幕 | 仅公开稿件。 |
| 事实核查源 | 腾讯较真、辨真、IFCN RSS | 作为 `seed_fact_check_url`。 |

## 5. 数据量目标（个人可完成规模）

- 事件数：30–50（覆盖健康、公共安全、政治、社会、娱乐等主题）
- 每事件帖数：≥40（真/假/辟谣/截图混合）
- 视图数：`full_bundle` 1 条 + `observed_view` 至少 2 条
- 总样本量：约 1,500–2,500 视图条目（适合 Zenodo/OSF 发布）

## 6. 发布内容

1. `events.csv`：事件级元数据。
2. `posts.jsonl`：帖原文 + 平台字段。
3. `views/`：保存文本、截图、媒体文件及 YAML 元数据。
4. `labels.jsonl`：LLM 预测 + 人工复核记录。
5. `scripts/`：采集、清洗、模态缺失生成脚本。
6. `DATACARD.md`：数据卡（含采样策略、偏差、许可、隐私处理）。

## 7. 安全与隐私

- 对用户 ID 做哈希并移除私信内容。
- 图片/视频若涉及个人敏感信息，需要模糊处理或提供 URL + 哈希但不直接发布文件。
- 在 Data Card 中明确许可用途（研究/教学）、免责声明、以及撤稿流程。

---

下一步：根据本草案，细化采集与标注流水线，并列出可执行脚本/任务。***
