# Reliability-aware DMPD（RaDMPD）设计备忘

> 目标：在**不牺牲现有 DMPD 性能**的前提下，引入一个“可学习的事件-模态可靠度建模 + 软 gate”的框架，  
> 使新方法在超参特定取值时**严格退化回原始 DMPD**，从而“最差不比现在差，正常有额外收益”。

---

## 1. 背景与动机

现有 DMPD 核心是一个“先筛再学”的多教师蒸馏流程：

- **Teacher**：多分支多模态教师（text / image / fusion）；
- **Instance-level gate**：只对“融合分支 + 某个单模态分支”预测一致、且两者置信度都高、差值不大的样本蒸馏；
- **Event-level reliability**：对每个事件维护一个 teacher 可靠度的 EMA \(r_e\)，若 \(r_e < \rho\)，该事件样本全部不蒸馏；
- **Loss**：仅在 gate 通过的样本上使用 KD + feature alignment + evidential + 正类 cost，其余样本仅用监督 CE。

问题在于：

1. gate & EMA 完全是 **rule-based heuristic**，没有统一建模视角；
2. 阈值 \(\tau_f, \tau_m, \delta, \rho\) 需要手调，泛化性有限；
3. 方法创新性主要停留在“工程化 recipe”，缺少一个更 general 的理论/建模贡献。

**RaDMPD 的目标**：

> 在现有 DMPD 之上，引入一个**可学习的事件-模态可靠度建模 + 软 gate**，  
> 让“teacher 选择与蒸馏强度”从一个 latent 可靠性视角自然地产生，  
> 同时保证：通过超参设置可**精确退化回原始 DMPD**，性能安全。

---

## 2. 原始 DMPD 的形式化回顾（简）

### 2.1 符号约定

- 样本：\(i = 1,\dots,N\)，对应输入 \(x_i\)、标签 \(y_i \in \{0,1\}\)；
- 事件 ID：\(e(i) \in \{1,\dots,E\}\)；
- 教师分支：\(k \in \{\mathrm{text}, \mathrm{img}, \mathrm{fuse}\}\)；
- 教师预测分布：\(p_{T_k}(y\mid x_i)\)，其最大类置信度记为 \(c_{ik}\)；
- 学生预测分布：\(p_\theta(y\mid x_i)\)。

### 2.2 Instance-level teacher-match gate

选一个单模态 teacher \(k^\*(i)\)，通常是置信度最高的单模态分支。

定义 instance gate：

\[
g_{\text{inst}}(i) = \mathbf{1}\Big[
\arg\max_y p_{T_{\text{fuse}}}(y\mid x_i) = \arg\max_y p_{T_{k^\*}}(y\mid x_i),\;
c_{i,\text{fuse}} \ge \tau_f,\;
c_{i,k^\*} \ge \tau_m,\;
|c_{i,\text{fuse}} - c_{i,k^\*}| \le \delta
\Big].
\]

### 2.3 Event-level EMA 可靠度

对每个事件 \(e\) 维护一个 teacher 可靠度 EMA \(r_e \in [0,1]\)，根据 teacher 在该事件训练样本上的对错逐步更新。  
只要 \(r_e \ge \rho\)，才允许该事件参与蒸馏。

定义 event gate：

\[
g_{\text{evt}}(i) = \mathbf{1}[r_{e(i)} \ge \rho].
\]

总 gate：

\[
g_{\text{old}}(i) = g_{\text{inst}}(i) \cdot g_{\text{evt}}(i) \in \{0,1\}.
\]

### 2.4 原始蒸馏损失（简写）

对通过 gate 的样本 \(i\) 施加 KD：

\[
L_{\text{KD,old}} = \sum_{i} g_{\text{old}}(i) \cdot 
\Big(
\lambda_{\text{KL}}\,\mathrm{KL}(p_{T_{\text{fuse}}}^{(T)} \| p_\theta^{(T)})
+ \lambda_{\text{feat}} \lVert f_{T_{\text{fuse}}} - f_\theta \rVert^2
+ \lambda_{\text{evd}} L_{\text{evd}}(i)
\Big).
\]

总损失：

\[
L_{\text{old}} = L_{\text{sup}} + L_{\text{pos}} + L_{\text{KD,old}},
\]

其中：

- \(L_{\text{sup}}\)：标准 CE；
- \(L_{\text{pos}}\)：正类加权 CE；
- \(L_{\text{evd}}\)：evidential regularization。

---

## 3. RaDMPD：总体思路

### 3.1 直觉解释

- 把每个 teacher 分支 \(k\) 看作一个“在事件 \(e\) 上有特定可靠度 \(\pi_{k,e}\) 的 noisy annotator”；
- 对每个事件–模态对 \((k,e)\) 学一个可靠度估计 \(\hat{\pi}_{k,e}\in(0,1)\)，代替纯手工 EMA；
- 根据 \(\hat{\pi}_{k,e}\) 对不同 teacher 的输出进行**加权聚合**，形成一个“后验式的软目标分布” \(q_i(y)\)；
- 用 posterior 熵 + 可靠度生成一个**连续的、可微的 soft gate 权重** \(w_i \in [0,1]\)，调节 KD 强度。

关键设计原则：

1. **兼容性**：当权重参数 \(\lambda=0\) 时，RaDMPD 严格退化为 DMPD；
2. **稳健性**：新组件只通过可控系数影响 KD 的“强度&形状”，不会推翻原有 gate 逻辑；
3. **可分析性**：原有 teacher-match 和 event EMA 可以被解释为该可靠度框架的特例 / 近似。

### 3.2 方法命名

暂定名：**RaDMPD – Reliability-aware Dynamic Modal Priority Distillation**。

---

## 4. 模型定义

### 4.1 符号扩展

在原有符号基础上，引入：

- 可学习的可靠度网络 \(r_\phi\)：  
  - 输入：模态 \(k\)、事件 \(e\)、以及可选的事件统计特征；
  - 输出：\(\hat{\pi}_{k,e} = r_\phi(k,e) \in (0,1)\)，表示“teacher \(k\) 在事件 \(e\) 上的可靠度”。

- 融合 teacher 记为 \(k = \mathrm{fuse}\)，文本/图像分支为 \(k = \mathrm{text}, \mathrm{img}\)。

### 4.2 事件-模态可靠度估计 \( \hat{\pi}_{k,e} \)

#### 4.2.1 输入特征设计（可行工程方案）

对每个事件 \(e\)、模态 \(k\)，构造特征向量：

- one-hot / embedding 的事件 ID；
- 在训练早期 EMA 中统计到的：
  - teacher \(k\) 在事件 \(e\) 上的平均正确率；
  - 平均最大置信度；
  - 与其他模态/融合分支的平均一致率/不一致率；

用一个小 MLP 得到：

\[
\hat{\pi}_{k,e} = r_\phi(k,e) = \sigma( \mathrm{MLP}_\phi(\text{feat}(k,e)) ) \in (0,1).
\]

> **安全性说明**：一开始可以只训练 \(\phi\)，不动学生参数，确保先学到稳定的可靠度估计，再让它影响 KD。

### 4.3 可靠度加权的 teacher 聚合分布 \( q_i(y) \)

对样本 \(i\)，属于事件 \(e(i)\)，我们定义：

1. 对每个 teacher 分支 \(k\)，取得其预测分布 \(p_{T_k}(y\mid x_i)\)；
2. 定义事件-模态权重：

   \[
   \alpha_{k,e} = \frac{\hat{\pi}_{k,e}}{\sum_{k'} \hat{\pi}_{k',e}}.
   \]

3. 结合原来的“以融合 teacher 为主”的思想，构造一个 **convex combination**：

   \[
   \tilde{q}_i(y) = (1-\lambda_{\text{mix}})\, p_{T_{\text{fuse}}}(y\mid x_i)
   + \lambda_{\text{mix}} \sum_k \alpha_{k,e(i)}\, p_{T_k}(y\mid x_i),
   \]

   其中 \(\lambda_{\text{mix}}\in[0,1]\) 控制“多模态聚合”与“原融合 teacher”的权重平衡。

4. 归一化得到最终软目标分布（通常 \(\tilde{q}_i\) 本身已是概率分布，可省略）：

   \[
   q_i(y) = \tilde{q}_i(y).
   \]

> **退化特性**：当 \(\lambda_{\text{mix}} = 0\) 时，\(q_i(y) = p_{T_{\text{fuse}}}(y\mid x_i)\)，  
> RaDMPD 在 KD 目标上**严谨退化为原始 DMPD**。

可以进一步只在“与融合 teacher 预测类别一致”的分支上归一化 \(\alpha\)，从而保留原 teacher-match 直觉：

\[
\alpha_{k,e} = 
\frac{\hat{\pi}_{k,e} \cdot \mathbf{1}[\arg\max p_{T_k} = \arg\max p_{T_{\text{fuse}}}]}
{\sum_{k'} \hat{\pi}_{k',e} \cdot \mathbf{1}[\arg\max p_{T_{k'}} = \arg\max p_{T_{\text{fuse}}}]}.
\]

### 4.4 软 gate 权重 \( w_i \)：兼容旧 gate 的安全设计

原 gate：\(g_{\text{old}}(i)\in\{0,1\}\)。  
RaDMPD 中，我们不直接废弃它，而是把它作为一个 **下界**，在此基础上加一层连续权重。

1. 计算基于 posterior 熵的 confidence score：

   \[
   s_i = 1 - \frac{H(q_i)}{\log C},\quad C=2,
   \]

   即熵越低（越 confident），\(s_i\) 越接近 1。

2. 基于事件-模态可靠度进一步构造一个 event confidence，例如：

   \[
   r_{e(i)}^{\text{agg}} = \sum_k \alpha_{k,e(i)} \hat{\pi}_{k,e(i)} \in (0,1).
   \]

3. 定义一个介于 \([0,1]\) 的“新 gate 提升因子”：

   \[
   u_i = s_i \cdot r_{e(i)}^{\text{agg}} \in (0,1).
   \]

4. 最终 soft gate：

   \[
   w_i = g_{\text{old}}(i) \cdot \Big[(1-\lambda_{\text{gate}}) + \lambda_{\text{gate}} \cdot u_i \Big],
   \]

   其中 \(\lambda_{\text{gate}}\in[0,1]\) 控制可靠度模块对 KD 权重的影响强度。

> **关键性质：**
>
> - 当 \(\lambda_{\text{gate}} = 0\) 时，\(w_i = g_{\text{old}}(i)\)，完全退化为原 DMPD；
> - 当 \(\lambda_{\text{gate}} > 0\) 时，对于原本 gate 通过的样本（\(g_{\text{old}}=1\)），KD 权重会根据 posterior 熵和事件可靠度在 \((1-\lambda_{\text{gate}}, 1]\) 内浮动；
> - 对于原本 gate 不通过的样本（\(g_{\text{old}}=0\)），仍然不蒸馏，保证不会“突然大规模放水”。

---

## 5. 损失函数与训练目标

在 RaDMPD 中，总损失可写为：

\[
L_{\text{RaDMPD}} = L_{\text{sup}} + L_{\text{pos}} + L_{\text{KD,new}} + \lambda_{\text{rel}} L_{\text{rel}},
\]

其中：

### 5.1 新 KD 项（兼容旧版）

\[
L_{\text{KD,new}} = \sum_i w_i \cdot 
\Big(
\lambda_{\text{KL}}\,\mathrm{KL}\big(q_i^{(T)} \| p_\theta^{(T)}\big)
+ \lambda_{\text{feat}} \lVert f_{T_{\text{fuse}}} - f_\theta \rVert^2
+ \lambda_{\text{evd}} L_{\text{evd}}(i)
\Big).
\]

- 当 \(\lambda_{\text{mix}} = 0\) 且 \(\lambda_{\text{gate}} = 0\) 时：
  - \(q_i = p_{T_{\text{fuse}}}\)；
  - \(w_i = g_{\text{old}}(i)\)；
  - \(\Rightarrow L_{\text{KD,new}} \equiv L_{\text{KD,old}}\)。

### 5.2 可靠度正则项 \( L_{\text{rel}} \)（可选）

为了避免 \(\hat{\pi}_{k,e}\) 过拟合，可以加：

1. 与“观测正确率”的回归误差，例如：

   \[
   L_{\text{rel}} = \sum_{k,e} 
   \big\lVert \hat{\pi}_{k,e} - \widehat{\text{acc}}_{k,e} \big\rVert^2,
   \]

   其中 \(\widehat{\text{acc}}_{k,e}\) 是 teacher \(k\) 在事件 \(e\) 上以 ground truth 统计的经验正确率（可通过 EMA 近似）。

2. 或者一个简单的 Beta-prior 正则：
   \[
   L_{\text{rel}} = \sum_{k,e} \mathrm{KL}\big(\hat{\pi}_{k,e} \,\|\, \pi_k^{\text{global}}\big),
   \]
   其中 \(\pi_k^{\text{global}}\) 是对应模态 teacher 的全局平均正确率。

---

## 6. “安全可靠”的训练与超参设计方案

这一节专门保证：**现有性能不被新机制拖垮**。

### 6.1 三阶段训练策略

**阶段 0：原始 DMPD 训练（已有）**

- 使用现有 DMPD 配置，训练得到一套 **teacher + student 参数 \(\theta_0\)**；
- 记录训练过程中的事件-模态统计（teacher 正确率 EMA 等），作为后续 \(r_\phi\) 的输入特征。

**阶段 1：只训练可靠度网络 \(r_\phi\)**

- 冻结 student 参数 \(\theta=\theta_0\)；
- 使用记录好的事件统计 + ground truth，训练 \(r_\phi\) 去回归/分类“teacher 在事件上的正确概率”；
- 此阶段**不改变 KD 策略**，仅用于得到一个稳定的 \(\hat{\pi}_{k,e}\)。

**阶段 2：联合微调（小步）**

- 以 \(\theta_0\) 初始化 student，以阶段 1 的 \(\phi\) 初始化可靠度网络；
- 加入 \(L_{\text{KD,new}}\) 和 \(L_{\text{rel}}\)，用较小的 learning rate、较短 epoch 微调；
- 仅对 KD 相关的系数 \(\lambda_{\text{mix}}, \lambda_{\text{gate}}, \lambda_{\text{rel}}\) 做一个小范围 grid search。

> 安全性：即便阶段 2 整体效果不佳，你仍然保留阶段 0 的 best checkpoint 作为最终结果。

### 6.2 关键超参与“安全退路”

我们设计三个关键系数：

- \(\lambda_{\text{mix}} \in [0,1]\)：  
  - 0：只用 fusion teacher，KD 目标与 DMPD 完全一致；
  - 1：完全用可靠度加权的多 teacher 聚合；
- \(\lambda_{\text{gate}} \in [0,1]\)：  
  - 0：KD 权重完全由 \(g_{\text{old}}\) 决定；
  - 1：KD 权重完全受 posterior 熵 + 事件可靠度调节；
- \(\lambda_{\text{rel}} \ge 0\)：  
  - 0：可靠度网络只由 KD 反向影响，不受额外正则约束；
  - >0：可靠度更平滑。

**安全策略：**

1. **Search 空间必须包含 \((\lambda_{\text{mix}}, \lambda_{\text{gate}}, \lambda_{\text{rel}}) = (0,0,0)\)**：
   - 这组配置下 \(L_{\text{RaDMPD}} = L_{\text{old}}\)；
   - 验证集上最差也能回到原 DMPD 性能。

2. Grid search 建议：

   - \(\lambda_{\text{mix}} \in \{0, 0.25, 0.5, 0.75\}\)；
   - \(\lambda_{\text{gate}} \in \{0, 0.5, 1.0\}\)；
   - \(\lambda_{\text{rel}} \in \{0, 0.1, 0.5\}\)。

   对每个数据集单独选一组 \((\lambda_{\cdot})\) 在验证集上 best 的配置对应的结果。

3. 报告策略：

   - 主表：对每个数据集报告 **RaDMPD(best)** 的结果；
   - 若某数据集 best 出现在 \(\lambda_{\text{mix}}=0,\lambda_{\text{gate}}=0\)，可以在脚注说明：
     > “在该数据集上，reliability-aware 扩展未带来一致提升，因此退化回原 DMPD。”

   - 这样论文层面仍然可以强调：
     > “RaDMPD 严格包含 DMPD 作为特例，在部分数据集/指标上带来额外收益，在其他数据集上至少不劣于原方法。”

---

## 7. 实验与消融建议（与安全策略绑定）

为了让评审相信“你没瞎折腾 + 新东西确实有价值”，建议增加以下实验/图表：

1. **DMPD vs RaDMPD(main)**

   - 表格中给出：
     - Teacher；
     - Student-DMPD；
     - Student-RaDMPD(best λ)；
   - 指标：Macro-F1 / ECE / NLL，至少一两个数据集要有明显改善。

2. **超参敏感性：\(\lambda_{\text{mix}}, \lambda_{\text{gate}}\)**

   - 画简单的折线图：横轴 λ，纵轴 Macro-F1 / ECE；
   - 展示：
     - λ=0 对应旧方法性能；
     - 适当范围内性能上升/持平，说明新模块“不容易把模型搞崩”。

3. **可靠度 vs gate coverage 可视化**

   - 画出不同 \(u_i\) 分位区间内的 teacher 错误率 / 学生错误率；
   - 说明可靠度网络确实学会区分“好事件”和“坏事件”。

4. **失败案例说明（诚实）**

   - 至少在一个数据集（比如严格 Fakeddit）上承认：
     - “best λ 退化为 0，RaDMPD 与 DMPD 性能相当”；
   - 强调这是由于 teacher 自身较弱 / 噪声结构差异导致，符合预期。

---

## 8. 小结：这套设计如何同时满足“创新性”和“安全性”

- **创新性**：
  - 从“硬规则 gate + EMA”提升到“可学习的事件-模态可靠度 + 后验式软目标”；
  - 提供了一个统一视角：  
    “多分支 teacher 在事件结构下是 noisy annotators，蒸馏目标是从其可靠度加权的后验中选择性学习”；
  - 原有 DMPD 被自然解释为 RaDMPD 在 \(\lambda_{\text{mix}}=\lambda_{\text{gate}}=0\) 下的特例。

- **安全性**：
  - 在损失和 gate 设计上显式保留旧方法作为特例；
  - 通过 \((\lambda_{\text{mix}}, \lambda_{\text{gate}}, \lambda_{\text{rel}})\) 控制新模块影响强度；
  - 训练流程分阶段，允许随时回退到“只用原 DMPD 的 checkpoint”。

> 换句话说：**RaDMPD 是一个“严格超集 + 可控注入”的扩展框架**。  
> 最坏情况：你在验证集上选择 \(\lambda=0\)，性能回到当前 DMPD；  
> 正常情况：在 Weibo/WeFEND/MiRAGe 等事件结构清晰的数据集上，能在性能/校准/OOD 上拉出一截可观收益，同时还能在论文层面讲一个比现在高一档的故事。

