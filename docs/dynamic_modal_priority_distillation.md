好！你选的“方案4：**动态模态优先蒸馏**（谁更确定谁当老师）”我给成一套可直接落地的实现清单。下面分两版：**A. 无图版（快跑 MVP）**和**B. 图结构版（并入 MUGCL）**。先把 A 做通，再把蒸馏钩到图模型里就是 B。

---

# A. 无图版（T-FND 基础上的动态蒸馏）

## 0) 目标与总览

* 两个单模态分支：Text、Vision，各自产生分类 logits 与**Dirichlet 证据**（不确定性）。
* 每个样本上**自动挑不确定性更低的模态当 teacher**；另一模态当 student。
* 蒸馏信号可以是**logits-KL**（主）+ **特征 MSE**（辅）；teacher 用 **stop-grad** 或 EMA。
* 融合头照常训练（可用不确定性加权），但不参与 teacher 选择。

## 1) 模块搭建

* 文本编码：BERT-base（或你现有 text encoder）→ 得到 `x_t`（[B, D]）。
* 图像编码：ViT-B/16（或现有 vision backbone）→ 得到 `x_v`（[B, D]）。
* 各自接两个头：

  * 分类头 `head_cls_*`: `MLP(D → C)` 输出 `logits_*`（C=2 或多类）
  * 证据头 `head_evi_*`: `MLP(D → C)` + **softplus** 激活，输出 `evidence_*`，令 `alpha_* = evidence_* + 1`
* 不确定性：`S_* = alpha_*.sum(dim=-1, keepdim=True)`；`u_* = C / S_*`（逐样本不确定性标量）

> 小贴士：证据头与分类头**共享 backbone**但**不共享**最后的 MLP；证据训练走证据损失（见 §4）。

## 2) Teacher/Student 动态选择（逐样本）

* 对 batch 中每个样本 i：

  * 若 `u_t[i] + δ < u_v[i]`，**Text→Teacher，Vision→Student**；
  * 若 `u_v[i] + δ < u_t[i]`，**Vision→Teacher，Text→Student**；
  * 否则（两者接近）：**跳过蒸馏**（避免用不可靠 teacher）。
* `δ` 是**最小置信差**阈值（如 0.05–0.1），防止抖动。

## 3) 蒸馏损失（主 + 辅）

* **主：Logits-KL（带温度 T）**

  * `p = softmax(logits_teacher / T) (detach)`
  * `q = log_softmax(logits_student / T)`
  * `L_KL = mean(sum(p * (log(p) - q))) * T^2`
* **辅：特征对齐（可选）**

  * 取 teacher 与 student 的**倒数第二层**或投影层 `z_teacher, z_student`（[B, d]）
  * `L_feat = MSE(z_student, z_teacher.detach())`
* 蒸馏总损失：`L_distill = λ_KL * L_KL + λ_feat * L_feat`（只对**被选中**的 student 样本求和）

> 经验值：`T=2` 或 `3`；`λ_KL=1.0`，`λ_feat=0~0.2`。前期以 KL 为主。

## 4) 证据学习（不确定性头）

* 每个模态各自有**证据损失**（可直接沿用 Evidential DL）

  * 交叉熵的**Dirichlet 期望** + **逐步加权的 KL 先验正则**
  * 训练前 20–30% 的 epoch **线性拉升** KL 系数（先学判别，再校准）
* 文本与图像证据损失分别为 `L_evi_t, L_evi_v`，并入总损失（`λ_evi` 可设 0.1–0.5）

## 5) 融合分支（可选，但建议保留）

* 做一次轻量的 cross-attention 得到交互特征 `m_f`，并保留 `m_t, m_v`（线性投影）。
* 用不确定性加权拼接：`m = [(1-u_v)*m_v, (1-u_t)*m_t, (u_t+u_v)*m_f]` → `MLP → logits_fuse`
* 融合支路走常规交叉熵 `L_fuse`（不参与 teacher 选择；但最后评测主要看融合输出）。

## 6) 训练总损失与流程

* 单模态分类：`L_cls_t, L_cls_v`（普通 CE）
* 证据损失：`L_evi_t, L_evi_v`
* 融合分类：`L_fuse`
* 蒸馏损失：`L_distill`（只对被判为 student 的样本求和）
* **总损失**：

  ```
  L = L_fuse
    + α * (L_cls_t + L_cls_v)
    + β * (L_evi_t + L_evi_v)
    + γ * L_distill
  ```

  推荐起始：`α=0.5, β=0.3, γ=1.0`
* **训练伪代码（关键处）**：

  ```python
  # x_t, x_v from encoders; logits_t/v, alpha_t/v; u_t/v computed.

  # 1) 基本分类/证据/融合损失
  L_fuse = CE(logits_fuse, y)
  L_cls_t = CE(logits_t, y);   L_cls_v = CE(logits_v, y)
  L_evi_t = evidential_loss(alpha_t, y, step)  # 含KL拉升
  L_evi_v = evidential_loss(alpha_v, y, step)

  # 2) 动态蒸馏挑选
  with torch.no_grad():
      choose_t_teacher = (u_t + delta < u_v)
      choose_v_teacher = (u_v + delta < u_t)

  L_KL, L_feat = 0., 0.
  if choose_t_teacher.any():
      p = softmax((logits_t / T).detach())
      q = log_softmax(logits_v / T)
      L_KL += kl_div(p, q, reduce_over=choose_t_teacher) * (T*T)
      L_feat += mse(z_v[choose_t_teacher], z_t[choose_t_teacher].detach())

  if choose_v_teacher.any():
      p = softmax((logits_v / T).detach())
      q = log_softmax(logits_t / T)
      L_KL += kl_div(p, q, reduce_over=choose_v_teacher) * (T*T)
      L_feat += mse(z_t[choose_v_teacher], z_v[choose_v_teacher].detach())

  L_distill = lambda_KL*L_KL + lambda_feat*L_feat

  # 3) 总损失
  L = L_fuse + α*(L_cls_t+L_cls_v) + β*(L_evi_t+L_evi_v) + γ*L_distill
  L.backward(); optimizer.step()
  ```
* **训练细节**：AdamW；bs=32；lr=2e-5（encoder）/1e-4（头）；warmup 5%；epoch 20–30；早停看 `F1` 与 `ECE`。

## 7) 关键开关 & 稳定技巧

* **停梯度**：teacher 侧必须 `detach()`；可选 **EMA-teacher**（Polyak 平均）替代实时 teacher，收敛更稳。
* **阈值 δ**：0.05–0.1；加**动量门控**防抖（如 EMA(u) 决策）。
* **冷启动**：前 3–5 个 epoch **不做蒸馏**（γ=0），等证据与分类先学稳。
* **不对称样本**：如只有文本没有图像，直接跳过“图像为 student”的蒸馏分支。

---

# B. 图结构版（并入 MUGCL）

在 A 跑通后，把“动态蒸馏”接到 MUGCL 的**图根节点级**与**图级表征**上：

## 1) 接入位置

* **节点级**：Text-Graph 的根节点表示 `h_t_root` 与 Vision-Graph 的根节点 `h_v_root`。
* **图级**：池化后的 `s_t`、`s_v`（MUGCL 里做图级对比的那个表示）。

## 2) 不确定性来源

* 文本/图像分支仍各自有**证据头**（同 A），产生 `u_t,u_v`（针对**新闻级样本**）。
* 用它来**选择 teacher**：对本条新闻的**整张图**蒸馏（而不是评论节点逐个蒸）。

## 3) 蒸馏目标

* **logits-KL**：图级分类头（基于 `s_t` 与 `s_v` 的各自 logits）做 KL 蒸馏（同 A）。
* **特征 MSE**：在节点级或图级表征上做 `MSE(s_student, s_teacher.detach())`。
* **与对比损失并存**：保留 MUGCL 的节点/图级 InfoNCE；蒸馏只是**额外项**，不会打破对齐学习。

## 4) 总损失（示例）

```
L = L_graph_cls
  + λN * L_node_InfoNCE
  + λG * L_graph_InfoNCE
  + β * (L_evi_t + L_evi_v)
  + γ * L_distill_graph   # 动态蒸馏（teacher 由 u_t/u_v 决）
```

* `L_distill_graph = λ_KL * KL(logits_teacher, logits_student) + λ_feat * MSE(s_student, s_teacher.detach())`
* **注意**：如果你在 MUGCL 里已有“跨模态不确定性策略（CUL）”，可保留；动态蒸馏是互补的。

## 5) 训练策略

* 同样使用**冷启动**（先训对比 + 分类，再开蒸馏）。
* **温度与权重**与 A 同；epoch 更长（30–50），学习率更小（图模型更敏感）。

---

# 实验设计与你要的“小而有效提升”

1. **主表**：

   * 基线1：原 T-FND / 原 MUGCL
   * ours-A：+ 动态蒸馏（无图）
   * ours-B：+ 动态蒸馏（并入图）
     指标：Accuracy/F1/AUC + **ECE**（校准）+ **NLL/Brier**（可说明不确定性收益）

2. **消融**：

   * 去蒸馏（γ=0）
   * 固定单边蒸馏（永远 Text→Vision 或 Vision→Text）
   * 随机 teacher（检验动态选择的价值）
   * 只 KL / 只 MSE / KL+MSE

3. **鲁棒性**：

   * 模态缺失（只文本/只图像）
   * 加噪（文本同义改写、图像 JPEG 压缩/裁剪）
   * 早期样本（只保留少量评论/短传播）

> 预期现象：
>
> * **MVP（无图）**版本即可在 Weibo/Twitter/Fakeddit 上对基线有**2–4pt F1 提升**，且 **ECE 明显下降**；
> * 图版在评论噪声多/图像质量参差的数据上提升更稳定。

---

# 常见坑位 & 规避

* **teacher 不稳定**：用 EMA-teacher；或设更高 δ、推迟开启蒸馏。
* **蒸馏压制多样性**：λ_feat 别太大（≤0.2），以 KL 为主。
* **证据发散**：KL 先验**缓慢拉升**；证据头学习率比主干小一档。
* **计算开销**：只在被选中样本上计算蒸馏；其余样本不做 KL/MSE。

---

如果你把现有代码结构（encoder/heads/forward/loss 调度函数）发我一段目录与关键函数名，我可以把**可直接粘贴的 PyTorch 代码片段**（证据损失、动态选择、蒸馏汇总）写到对应位置，帮你一次性打通 A→B 的升级路径。
