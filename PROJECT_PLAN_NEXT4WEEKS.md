# Project Plan — ARR/ACL 2026 (updated)

## 已完成（资料与结果全部落盘）
### 写作与结构
- 设定已收紧：多分支（text/image/fusion）teacher + 显式事件 ID 的监督多模态谣言检测，非通用 KD。
- 主文精简：仅方法框架图 + 主结果表 + 消融表，正文不含可靠性/门控占位图。
- 话术降档：贡献强调“系统化组合与评估”；MiRAGeNews/Fakeddit 如实说明；校准表述为“保持不变或小幅提升”。
- 附录已齐：错误案例（无 PostID）、per-class ECE/NLL、WeFEND 条形图、分歧 vs 错误图、松阈值 gate 覆盖图、噪声鲁棒性小表。
- 数据溯源与简报：`DATA_SOURCES.md`、`PROJECT_BRIEF_WRITING.md`、`PROJECT_BRIEF_ENGINEERING.md`。

### 实验（可直接入文）
- 主结果（Table 1 已更新）：
  - Weibo：DMPD 0.948 / 0.948 / 0.040；Noisy-Student 0.941 / 0.941 / 0.056。
  - WeFEND：DMPD 0.939 / 0.913 / 0.030；Noisy-Student 0.939 / 0.908 / 0.057。
  - MiRAGeNews OOD：Teacher 0.883 / 0.882 / 0.057；DMPD 0.902 / 0.902 / 0.053；Noisy-Student 0.901 / 0.901 / 0.062（3 seeds）。
- 消融（Table 2）：去 gate / 去 event / 去 evidential 均降低 MF1、提高 ECE。
- Baseline 全覆盖：Simple-Conf-KD、Selector ablations、Noisy-Student、Text/Image-only、SAFE/EANN、fusion teacher。
- 分歧 vs 错误：图 `figures/disagreement_error.pdf`，数据 `paper_results/{weibo,wefend}_disagreement_error.csv`。
- Gate 覆盖：Weibo loosegate v2（coverage>0，Test≈0.947/0.947/ECE 0.046），WeFEND loosegate（coverage>0，Test≈0.935/0.904/ECE 0.056），图已入附录。
- 噪声鲁棒性（Weibo，1 seed）：noise 0.1/0.2/0.3，DMPD≈Noisy-Student，轻噪 ECE 略低；CSV `paper_results/weibo_noise_comparison.csv`。
- MiRAGeNews Noisy-Student 3 seeds OOD：Acc 0.9013 ± 0.0081 / MF1 0.9009 ± 0.0081 / ECE 0.0622 ± 0.0080，已写入主表。

## 待完成（按优先级拆分）
### P1 —— 文本与呈现（低风险，立即做）
1) Experiments 段补一句：MiRAGeNews Noisy-Student（3 seeds）表现很强，DMPD 接近且优于 teacher，ECE 更低。
2) Appendix 加一句噪声结论：重噪下 DMPD≈全量 KD，轻噪 ECE 略低。
3) Analysis/正文补一句：跨模态分歧 decile 越高，学生错误率显著上升（有现成图）。
4) Gate 覆盖在正文一句话点到即止：放松阈值时 coverage>0，性能/ECE 基本不变；详细图留附录。

### P2 —— 可选小实验（看时间）
5) MiRAGeNews DMPD 轻调参（1–2 seeds，温度/增广）以缩小与 Noisy-Student 差距；无收益则保留现表。
6) WeFEND τ/δ 小 sweep（1–2 点）展示 coverage–性能权衡，结果放附录。

### P3 —— 高投入，下一轮（非本轮投稿硬性项）
7) VLM teacher 试验（BLIP-2/Qwen-VL 等）。
8) 更系统的合成噪声 + coverage 曲线 / 风险分解。

## 备注
- 当前 7 张 GPU 空闲（loosegate v2 已完毕）。
- 主文保持 8 页；新增内容优先放附录，正文仅留概括句。
- 危险图已移除，避免再加入占位或噪声大的可视化。
