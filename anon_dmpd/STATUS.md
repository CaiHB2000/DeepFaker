### 实验现状（2025-11-27）
- **Weibo 主结果**：seed01 Acc 0.9495 / Macro-F1 0.9495 / ECE 0.0323；seed00 0.9468 / seed02 0.9481 作为补充。strict 协议学生 Acc 0.880 / Macro-F1 0.878。
- **Weibo 消融 (seed00)**：
  - no-event: Macro-F1 0.9474 / ECE 0.0408；
  - no-gate: 0.9447 / ECE 0.0367；
  - no-evi: 0.9440 / ECE 0.0474。
  → 去 event/gate/evi 均降分，gate+event 贡献最明显，evidential 有助校准。
- **WeFEND 主结果**：softquota_seed00 Macro-F1 0.9186 / ECE 0.0378；dualgate_reweight_seed00 Macro-F1 0.9174 / ECE 0.0361；教师 Macro-F1 0.9015 / ECE 0.0431，提升约 +1.7pt。
- **WeFEND 消融 (seed00)**：no-event 0.9088 / ECE 0.0532；no-gate 0.8938 / ECE 0.0504；no-evi 0.8991 / ECE 0.0485。事件过滤与 gate 作用显著，evi 也有贡献。
- **Fakeddit（时间划分 2-way）**：学生 gate/softquota/reweight seed00/01 Macro-F1 ≤0.53，未超教师 (0.51) 和强基线 SAFE/EANN (~0.92)。需重设计策略或转用 SDML/MIMoE/SpotFake+ 基线，正文暂不使用该结果。

待办（论文一周内收尾）：
1) Fakeddit：尝试“强制蒸馏/无门控”或直接跑 SDML/MIMoE/SpotFake+ 基线占位；若仍无提升，移至附录/未来工作。
2) 图表与检验：Weibo/WeFEND 生成可靠性图、gate 覆盖、事件分布；Bootstrap + McNemar 对最强基线。
3) 数据集统计表：样本/事件/类比例等脚本化生成。
4) 论文正文仅保 Weibo + WeFEND 主表与消融；Fakeddit/其它英文集移附录或声明未来工作。

## 2025-11-15 Weibo baseline updates
- 补跑 `dynamic_distill/configs/weibo_teacher_baseline.yaml`（无蒸馏）。测试集：Acc 0.9433 / Macro-F1 0.9433 / ECE 0.0355。
- 动态蒸馏最优 `weibo_dynamic_distill_seed01`：Acc 0.9495 / Macro-F1 0.9495 / ECE 0.0323。Teacher→Student 提升 +0.62pt Acc / +0.61pt F1，同时 ECE ↓0.003。
- `evaluate_model.py` + `plot_reliability.py` 已产出可靠性曲线：
  - `paper_results/weibo_teacher_baseline/weibo_teacher_baseline_seed00/reliability.png`
  - `paper_results/weibo_dynamic_distill/weibo_dynamic_distill_seed01/reliability.png`
- 额外基线：EANN (seed0) 复现完成 → Acc 0.937 / Macro-F1 0.937 / ECE 0.052（config dynamic_distill/configs/weibo_eann.yaml；输出 paper_results/weibo_eann/weibo_eann_seed00）。
- Weibo 原论文协议（strict-id：train/test 不交叉）已跑：
  - Teacher baseline strict：Acc 0.877 / Macro-F1 0.874 / ECE 0.105（paper_results/weibo_teacher_baseline_strict/...）。
  - Student strict：Acc 0.880 / Macro-F1 0.878 / ECE 0.098（paper_results/weibo_dynamic_distill_strict/...）。
  - EANN strict：Acc 0.858 / Macro-F1 0.855 / ECE 0.132（paper_results/weibo_eann_strict/...）。
  - 对应 reliability png 已生成在各自目录下。
- WeFEND 主协议：
  - Teacher baseline (gamma=0): Acc 0.9425 / Macro-F1 0.9134 / ECE 0.0463 （paper_results/wefend_teacher_baseline/...）。
  - Student (event_consensus_balanced_seed00): Acc 0.9436 / Macro-F1 0.9152 / ECE 0.0506 （paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2/...）。
  - EANN baseline: Acc 0.9304 / Macro-F1 0.8984 / ECE 0.0476 （paper_results/wefend_eann/...）。
  - 已生成 reliability 图：teacher（.../reliability.png）、student（.../reliability.png）。
