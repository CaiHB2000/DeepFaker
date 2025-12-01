# 项目进展速记（2025-11-06）

## 数据准备与基线
- 制定了 Weibo / WeFEND / Fakeddit 官方拆分方案，并补充事件级去重、图像哈希等清洗脚本，确保训练与测试互不泄漏。
- 建立了三套公开方法对比：不确定融合教师、EANN、SAFE 等，形成跨数据集基线参考。

## 当前最佳结果
- **Weibo**：`weibo_uncertainty_fusion_seedseed02` — Acc 0.9515 / Macro-F1 0.9515。
- **WeFEND**：`wefend_dynamic_distill_teacher_wcls_teacher_match_seed00`（自制 teacher-match 实现） — Acc 0.9481 / Macro-F1 0.9222。
- **Fakeddit**：`fakeddit_dynamic_distill_teacher_seed00` — Acc 0.8705 / Macro-F1 0.8705。
- 所有 summary.json、日志与失败记录位于 `paper_results/`、`logs/`、`dynamic_distill/experiments/wefend_failed_attempts.md`。

## 关键探索
- Teacher-match 框架：长程训练 + 仅老师正确样本蒸馏可稳定获得 ~0.97pt 提升。
- 正类增强尝试：事件重加权、fallback、δ/温度微调、学生聚焦等组合均未突破差值瓶颈，易放大噪声事件。
- 可视化结果（`paper_results/figures/wefend_acc_timeline.png`）显示准确率已趋于平台期。

## 当前不足
- WeFEND 仍未达到 ≥+1.5pt 差值目标；正类事件覆盖不足且容易引入噪声。
- Fakeddit 多模态蒸馏策略缺乏系统优化，尚未体现显著优势。
- 三个数据集尚未形成统一、可迁移的蒸馏方案以支撑投稿论点。

## 下一步方向
1. 事件分层蒸馏：在 `trainer` 中实现按事件可靠度的动态权重，让低可靠事件以低强度蒸馏。
2. 双阶段调度：宽松阶段补齐难事件，再回到严格门控微调，重点监控正类召回。
3. 多 Seed 验证：对候选最优方案在至少 3 个种子上复现差值，确保统计可信。
4. 扩展到 Fakeddit：复用改进后的策略，提升图文融合鲁棒性，并更新跨数据集对比表。
