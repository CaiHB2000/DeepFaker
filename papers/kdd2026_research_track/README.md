# KDD 2026 Research Track 投递包（准备版）

更新时间：2025-11-12

- 官方 CfP（以 Research Track 为准）：见 KDD 2026 官网 Research Track 页面。
- 模板：ACM `acmart`（`sigconf,anonymous,review`），提交版内容页限 **8 页（不含参考文献）**；可附不计页的附录与补充材料；相机版通常允许 12 页总长、内容 9 页（以官网为准）。
- 盲审：双盲；遵守 ACM 政策；允许带匿名的补充材料与代码链接。
- 重现性：填写/附带 Reproducibility Checklist，提供脚本、随机种子、环境说明。
- 多投政策：严格禁止一稿多投；遵守冲突声明与伦理。

> 注：请以 KDD 2026 Research Track 官方页面为最终权威；若与此处不一致，以官网为准。

## 目录结构
- `main.tex`：基于 `acmart` 的稿件骨架（匿名审稿模式）。
- `refs.bib`：BibTeX 参考文献占位。
- `sections/`：章节占位。
- `figs/`：图片占位。

## 编译
- 本地需要安装 ACM `acmart` 宏包；或将本目录上传到 Overleaf 并切换 `acmart` 模板编译。

## 写作要点
- 侧重“严格划分协议 + 不确定性蒸馏 + 多数据集强基线对比 + 统计检验”。
- 主表建议：WeFEND（学生≥+1.5pt）、Fakeddit-time（teacher≥0.6，学生≥+1.5pt）、强基线（≥2）。
- 压力测试：Fakeddit-sub（报告 Macro-F1/Pos-F1/ECE 与校准/拒识）。
