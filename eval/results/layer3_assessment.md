# Layer 3 检索质量评估结论

**评估时间**：2026-05-09  
**原始数据**：`results/2026-05-09T00-35-25/layer3_retrieval.json`  
**Golden Query Set**：18 条（15条 in_scope / in_scope_multi，3条 out_of_scope）

---

## 一、总体指标

| 指标 | k=3 | k=5 |
|------|-----|-----|
| Context Precision | **0.578（FAIL）** | 0.52 |
| Context Recall | **0.967（PASS）** | 1.0 |
| 边界拒绝率 | 3/3（PASS） | — |
| out_of_scope L2 均值 | 1.061 | — |
| in_scope L2 均值 | 0.730 | — |
| L2 分数差（out - in） | **+0.331（PASS）** | — |

分数差 0.331 说明 embedding 能有效区分相关/不相关内容，向量化质量本身没问题。

---

## 二、失败 query 逐条分析

### q01「Z2训练的功率范围是多少」precision=0.33

召回了：
- #1 chunk\_01（训练区间应用，**相关**：含 Z2 描述）
- #2 chunk\_00（功率区间表，含 Z2 行，但 expected\_keywords 是 "56-75%"、"有氧耐力"，**相关**）
- #3 chunk\_02（FTP定义，**无关**）

实际 chunk\_00 和 chunk\_01 都召回了，但评分逻辑把 chunk\_02 计为无关。**根因**：expected\_keywords 设置过严（只列了 chunk\_01 的词），实际 chunk\_00 包含完整表格（含 "56-75%" 行），应标记为相关。**评估脚本问题，非检索问题**。

### q04「心率相比功率有什么局限性」precision=0.0

应该召回的 chunk\_01 含 "滞后于功率约30-60秒"，**完全没有被召回**。

召回的是：
- #1 chunk\_00（功率区间表，含"心率参考"列）
- #2 chunk\_03（TSS/IF/NP/EF，**无关**）
- #3 chunk\_02（FTP定义，**无关**）

根因：query 用了"心率"和"局限性"两个词，而 chunk\_01 在 embedding 空间里离 chunk\_00（含心率列的表格）更近，但 chunk\_00 排第一意味着 chunk\_01 被挤出了 k=3。**这是真实的检索失败**。心率滞后内容嵌在 chunk\_01 后半段，embedding 被前半段 Z2/Z4/Z5 的应用描述"稀释"了。

### q09「NP和平均功率有什么区别」precision=0.33

应召回 chunk\_03（含 NP 定义），实际排第一的是 chunk\_03，**正确**。但 #2 chunk\_02 和 #3 chunk\_00 均无关。根因同 q04——知识库太小（7个 chunk），k=3 约等于召回 43% 的知识库，噪声不可避免。

### q11「过度训练有哪些早期信号」precision=0.33

应召回 chunk\_05（含"早期信号"、"静息心率"），实际 #1 是 chunk\_05（**正确**），但 #2 是 chunk\_03（**无关**），#3 是 chunk\_06（含过度训练处理方式，可视为相关）。expected\_keywords 只包含了 chunk\_05 的词，评分低估了实际召回质量。**部分是评估脚本问题**。

### q13「训练后多长时间内补充营养效果最好」precision=0.33

应召回 chunk\_06（含"0-30分钟"），实际 #1 是 chunk\_06（**正确**），#2 是 chunk\_05（可视为相关），#3 是 chunk\_03（**无关**）。同样是 chunk\_03 占据了第三个噪声名额。

### q14「有氧能力的适应大概需要多少周」precision=0.33

应召回 chunk\_05（含"心肺适应：6-8周"），实际 #1 是 chunk\_05（**正确**），#2 是 chunk\_03（**无关**），#3 是 chunk\_06（含周期化内容，可视为相关）。

---

## 三、根本原因诊断

**核心问题：chunk\_03 是"中枢 chunk"（Hub Chunk）**

chunk\_03 的内容是 TSS参考值 + IF公式/含义 + NP定义 + EF定义，覆盖了 4 个不同概念。因为包含的术语最多，它的 embedding 在向量空间里是各方向都有分量，导致几乎对所有 query 都有一定相似度。在 15 条 in\_scope query 的 k=3 结果中，chunk\_03 出现了 **11 次**，但实际相关只有 5 次（涉及 IF/NP/EF/TSS 的 query）。

**次要问题：知识库规模导致结构性精度上限**

7 个 chunk，k=3 强制召回其中 43%。即使 embedding 完全精准，只要不相关的 chunk 数量超过知识库的 57%，精度就无法到达 1.0。这是一个知识库规模问题，不是 embedding 质量问题。

**验证**：out\_of\_scope query 的 L2 分数（均值 1.061）明显高于 in\_scope（0.730），说明 bge-small-zh 的区分能力是正常的。

---

## 四、评估脚本自身问题

以下 query 的低精度是 expected\_keywords 设置不准确导致的，不是真实检索问题：

- q01：chunk\_00 实际包含 Z2 的功率范围，但 keywords 没覆盖表格格式的数据
- q07：实际召回了 chunk\_06（含减量周，可视为相关），但 keywords 只设了 chunk\_05 的词
- q11、q13、q14：chunk\_06 通常包含可用信息，但没列入 keywords

修复方向：expected\_keywords 改为覆盖所有相关 chunk，或改用"expected\_chunk\_ids"直接指定。

---

## 五、结论与改进建议

### 检索质量评价

**向量化是健康的**，bge-small-zh 能正确区分领域相关与无关内容（L2 gap 0.331）。召回率 0.967 说明答案几乎都在 k=3 的结果里。精度问题不是 embedding 的问题，是 chunk 组织导致的。

### 优先修复

**修复 1（最高优先）：拆分 chunk\_03（中枢 chunk）**

把 ftp\_and\_metrics.md 中的 TSS参考值单独成一节，IF/NP/EF 各自独立小节，通过 markdown 结构让 MarkdownTextSplitter 把这 4 个概念分成独立 chunk（每个约 100-150 字）。预期 chunk 数从 7 增至 10-11，chunk\_03 的"中枢"效应消失，precision 预计提升至 0.7+。

**修复 2（中优先）：q04 检索失败问题**

"心率局限性"答案在 chunk\_01 后半段，被前半段内容稀释。可以在 training\_zones.md 中把"心率区间"单独成 `## 心率区间` 小节，让它有自己的 chunk。

**暂不修复**：k 值（k=3 合理）、embedding 模型（bge-small-zh 在这个任务上表现正常）。

### 评估脚本改进

Layer 3 下一版需要把 expected\_keywords 改为 expected\_chunk\_ids（在 Layer 1 跑完后填充），消除评分误差。
