# ARR（ACL Rolling Review）投递包（准备版）

更新时间：2025-11-12（美国时区）

- 官方入口与指南：
  - Authors Guidelines（投稿说明，含 Responsible NLP Checklist 与日期）：https://aclrollingreview.org/authors
  - 多投/伦理政策：https://www.aclweb.org/portal/content/acl-rolling-review
  - Word 模板停用公告（自 2026-03 起 Word 稿件将被直接 desk-reject）：https://aclrollingreview.org/discontinuation-word-template
- 模板与格式：
  - 使用官方 Overleaf LaTeX 模板（推荐）并遵守 *ACL 样式；本仓库的 `main.tex` 仅作结构占位，建议在 Overleaf 直接迁移到官方模板再编译。
- 盲审：
  - 双盲；文中避免自引用可溯源描述；arXiv 预印本允许但需匿名化引用策略按 ARR 说明处理。
- 频率与要求：
  - 每两个月一个批次；所有作者需在提交后 48 小时内注册为 ARR 审稿人（政策自 2025-05 起生效）。
  - 提交平台：OpenReview；需填写 Responsible NLP Checklist。

## 结构与写作建议
- 遵循 *ACL 长文结构：Abstract、Introduction、Related Work、Method、Experiments（含设置/消融/统计检验/校准）、Analysis（含错误案例与伦理考量）、Conclusion、Limitations、Ethics Statement、Reproducibility。
- 本目录下提供：
  - `main.tex`：稿件骨架（占位，迁移至官方模板编译）。
  - `refs.bib`：参考文献样例与占位。
  - `sections/`：分章节占位文件，便于多人协作。
  - `checklists/responsible_nlp_checklist.md`：清单占位，提交时需在 OpenReview 表单中完整填写。

## 编译与迁移
- 已内置 `acl.sty` 与 `acl_natbib.bst`，`main.tex` 使用 `\usepackage[review]{acl}`，可直接在本地或 Overleaf 编译。
- Overleaf 迁移：
  1) 打包上传 `papers/arr_rolling_review/` 整个目录（含 `acl.sty` / `acl_natbib.bst`）。
  2) 选择 `main.tex` 为主文件并编译（review 模式、自动行号与匿名）。
  3) 如需官方模板工程，也可在 Overleaf 先创建 ARR 工程，再将本目录 `sections/*.tex`、`refs.bib`、`macros.tex` 覆盖进去。
- 本地编译：若已安装 `tectonic` 或 TeXLive，执行 `./build.sh`（优先使用 `tectonic`）。

## 盲审自查
- 去除作者名/单位/可溯源链接；补充匿名化附录与仓库链接（可匿名 GitHub），在 rebuttal 期前不要暴露真实身份。

## 结果占位与复现实验
- 结果与脚本请从仓库的 `paper_results/`（指标）与 `dynamic_distill/configs/`、`scripts/`（命令、YAML）引用；在附录提供一键运行命令与 seeds。

### 一键生成论文表格与指标宏

当你完成一轮实验后，可运行：

```bash
python papers/arr_rolling_review/scripts/gen_tables_from_results.py \
  --root paper_results/wefend_dynamic_distill_teacher_wcls_selective_v2 \
  --out papers/arr_rolling_review
```

- 生成 `papers/arr_rolling_review/tables/{main_results,ablations}.tex` 并自动写入 `metrics_macros.tex`（供摘要/正文引用最佳分数）。
- 若不想展示当前数值，只需删除 `metrics_macros.tex`，宏将回退为 `macros.tex` 中的 `TBD`。
