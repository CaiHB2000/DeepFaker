# WEFEND 批量迭代实验策略（提醒）

> 目标：在多个 GPU（2–6）上并行探索多种参数组合，类似遗传/网格搜索，提高 WeFEND 主结果。

## 核心流程

1. **基准配置**：`dynamic_distill/configs/wefend_dynamic_distill_teacher_wcls_event_reliable.yaml`。
2. **自动生成多样配置**：
   ```bash
   python dynamic_distill/scripts/gen_wefend_batches.py \
     --base dynamic_distill/configs/wefend_dynamic_distill_teacher_wcls_event_reliable.yaml \
     --out-dir dynamic_distill/configs/auto_wefend \
     --count 12 --seeds 0 --random-seed 2025
   ```
   - 输出 `auto_wefend/wefend_auto_*.yaml`；参数随机采样自合理范围（温度、λ_KL、正类权重、门控阈值等）。
   - 生成运行脚本 `scripts/run_wefend_auto.sh`（分配 GPU 2–6，日志写入 `logs/wefend_batch/`）。

3. **批量运行**：
   ```bash
   bash scripts/run_wefend_auto.sh
   ```
   - 监控：`tail -f logs/wefend_batch/<name>.log`
   - 运行结束后：检查对应 `paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2/<run>/summary.json`。

4. **记录结果**：
   - 将每次实验的指标（Acc / Macro-F1 / Pos-F1 / ECE）写入 `dynamic_distill/experiments/wefend_failed_attempts.md` 或新建成功小节。
   - 保留最优配置 -> 回写手工命名，如 `event_reliable_poscost_v1.yaml`。

5. **循环迭代**：
   - 从表现最佳的 2–3 个配置出发，收紧参数范围再生成一批（类似遗传算法“精英保留+变异”）。
   - 对关键机制（正类代价敏感、门控阈值、事件可靠度等）做有针对性的微调。

## 注意事项

- 禁用 GPU 0/1（保留给他人），`gen_wefend_batches.py` 默认分配 2–6。
- 每轮结束务必更新失败/成功记录，避免重复踩坑。
- 若 GPU 资源不足，可调整 `--count` 或修改脚本中的 GPU 列表。
- 若需不同 seeds，可添加 `--seeds 0 1`（脚本会自动为每个 seed 生成命令）。

> 记住：始终保持 “生成配置 → 批量运行 → 汇总分析 → 精英保留再迭代” 的循环，直到 WeFEND 指标达到目标差值 ≥ +1.5pp。

## 2025-11-05 新增：Teacher-Match 精英批次

- 基于 `wefend_dynamic_distill_teacher_wcls_teacher_match.yaml` 人工设计 13 条策略（目录：`dynamic_distill/configs/wefend_teacher_match_batch/`）。
- 运行脚本：`bash scripts/run_wefend_teacher_match_batch.sh`
  - 利用 GPU 2–6 分批执行（每 5 个任务一轮并 `wait`），避免在单卡上堆叠过多作业。
  - 日志输出到 `logs/wefend_teacher_match_batch/`，保证后续分析有迹可循。
- 策略覆盖：delta 课程、正类聚焦、代价敏感、事件重加权、学生聚焦、温度调节、长程训练等多方向。
- 每轮结束后务必同步更新 `dynamic_distill/experiments/wefend_failed_attempts.md` 或成功记录，筛选能稳定领先 ≥1.5pt 的方案继续迭代。

## 2025-11-06 新增：Teacher-Match 精炼批次

- 以 `wefend_teacher_match_longtrain.yaml` 为基准，进一步组合 delta 微调、温度/λₖₗ 下调、正类权重、事件重加权等策略（目录：`dynamic_distill/configs/wefend_teacher_match_refine/`）。
- 生成命令：`python dynamic_distill/scripts/build_wefend_teacher_match_refine.py`，对应运行脚本 `bash scripts/run_wefend_teacher_match_refine.sh`。
- 日志目录：`logs/wefend_teacher_match_refine/`，按 5 卡一批自动 `wait`，便于“见缝插针”地在空闲 GPU 上运行。
- 目标：在保持 teacher-match 框架稳定性的同时，探索能否将 Macro‑F1 拉升至 ≥老师 +1.5pt；每次完成后照例更新 `wefend_failed_attempts.md` 汇总经验。

## 2025-11-06 新增：Pos-Balanced 增强批次

- 基线：`wefend_longtrain_pos_balanced.yaml`（目录：`dynamic_distill/configs/wefend_teacher_match_refine/`），在此基础上组合事件重加权、δ/温度微调、学生聚焦、软 fallback 等策略，生成配置存放于 `dynamic_distill/configs/wefend_teacher_match_enhanced/`。
- 生成命令：`python dynamic_distill/scripts/build_wefend_teacher_match_enhanced.py`；运行脚本 `bash scripts/run_wefend_teacher_match_enhanced.sh`（默认使用 GPU 0/3/4/5/6，日志写入 `logs/wefend_teacher_match_enhanced/`）。
- 若需要在前台保持长时间运行，可显式设置较大的 `timeout_ms` 或配合 `nohup … &`，避免命令 2 分钟超时导致批处理中断。
