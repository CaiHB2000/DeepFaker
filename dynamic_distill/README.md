# Dynamic Modal Priority Distillation

This package implements the training pipeline described in `docs/dynamic_modal_priority_distillation.md`.
It is scoped to **multimodal fake news detection** with dynamic teacher selection based on evidential uncertainty.

Status: **bootstrap**. The code currently targets the MVP (non-graph) variant and provides:

- backbone wrappers for HuggingFace text/vision encoders,
- classification和Dirichlet证据头,
- dynamic teacher/student 路由工具,
- 训练调度器与损失组合,
- Weibo/Twitter 多模态数据集读取与预处理,
- 默认配置与 CLI 训练脚本。

Next steps:

1. 并入图结构版（设计文档 B 方案）。
2. 对接实验追踪（TensorBoard/W&B）与系统化评估脚本。
3. 加入 EMA-teacher、动量门控等可选稳定策略。

## Layout

```
dynamic_distill/
├── configs/             # YAML configuration presets
├── scripts/             # CLI entrypoints (train/eval/export)
├── src/
│   ├── models/          # Encoders, heads, fusion, model assembly
│   ├── losses/          # Evidential and distillation losses
│   ├── training/        # Trainer orchestration
│   └── utils/           # Shared utilities (metrics, scheduling)
└── tests/               # PyTest-based sanity checks
```

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.1, torchvision ≥ 0.16
- transformers ≥ 4.40, huggingface-hub
- pandas, Pillow, pyyaml
- matplotlib (生成指标曲线/可靠性图)
- accelerate (optional, multi-GPU)

> 默认配置使用 `bert-base-chinese` 与 `google/vit-base-patch16-224-in21k`；首次运行会自动下载。

安装示例：

```bash
pip install -r requirements.txt
pip install -e dynamic_distill
```

## Quick Start

1. 下载并解压多模态数据至 `datasets/weibo/` 与 `datasets/twitter/`（参见仓库根目录 `datasets/`）。
2. 运行 Weibo MVP 训练（减配 quick-run）：

   ```bash
   python dynamic_distill/scripts/train_mvp.py \
     --config dynamic_distill/configs/default_mvp.yaml
   ```

   - `training.max_steps_per_epoch` 控制每 epoch 的样本数（默认 200，用于验证流程）。
3. 切换 Twitter 数据集：

   ```bash
   python dynamic_distill/scripts/train_mvp.py \
     --config dynamic_distill/configs/twitter_mvp.yaml
   ```

训练脚本会在每个 epoch 结束后输出验证集指标，并在收尾对测试集做一次评估。

### 批量实验 & 可视化

1. **消融实验批跑**：

   ```bash
   python dynamic_distill/scripts/run_experiments.py \
     --config dynamic_distill/configs/default_mvp.yaml \
     --experiments dynamic_full no_distill no_feature \
     --epochs 1 --max-steps 100
   ```

   结果 `summary.json`、`train/val_metrics.csv`、`test_predictions.csv` 会保存于 `runs/<timestamp>_<exp>/`。

2. **生成统计图**（需要 `matplotlib`）：

   ```bash
   python dynamic_distill/tools/plot_results.py \
     runs/20251027-130357_dynamic_full runs/... --output plots/latest
   ```

   将输出验证 F1 曲线、测试指标柱状图、校准曲线图。
