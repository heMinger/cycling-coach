# Layer 1 切片质量评估结论

**评估时间**：2026-05-09  
**原始数据**：`results/2026-05-09T00-06-50/layer1_chunks.json`

---

## 一、切片方法说明

使用 `MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)`。

这是一种"结构感知"切分器，与普通字符切分的区别：
- 优先在 markdown 语义边界切分（`##` 标题、段落、列表块），而不是到字符数上限就硬切
- 只有当某个 markdown 块本身超过 500 字时，才会在该块内部切分
- chunk_overlap=50 的作用是：当必须在块内部切分时，下一个 chunk 重复前一个末尾的 50 字符

**实际表现**：知识库 3 个文件合计 156 行，远小于 chunk_size 上限，切分完全由 markdown 结构边界决定，overlap 从未生效。

---

## 二、实际切片结果（共 11 个 chunk）

| id  | 来源                         | 字符数 | 说明                                      |
|-----|------------------------------|--------|-------------------------------------------|
| #00 | training_zones.md            | 440    | Z1-Z6 表格 + 标题，表格完整               |
| #01 | training_zones.md            | 348    | Z2/Z4/Z5 应用描述 + 心率区间说明          |
| #02 | ftp_and_metrics.md           | 468    | FTP定义/测试方法/参考水平表格 + TSS公式   |
| #03 | ftp_and_metrics.md           | 464    | TSS参考值 + IF公式/含义 + NP/EF定义       |
| #04 | ftp_and_metrics.md           | 97     | HRV定义（极短，仅2行）                    |
| #05 | recovery_and_adaptation.md   | 437    | 超量恢复原理 + 训练适应时间线 + 过度训练信号 |
| #06 | recovery_and_adaptation.md   | 330    | 恢复策略 + 训练周期化                     |
| #07 | user_profile.md              | 478    | 空表格（应排除）                          |
| #08 | user_profile.md              | 353    | 空表格（应排除）                          |
| #09 | user_profile.md              | 463    | 空表格（应排除）                          |
| #10 | user_profile.md              | 454    | 空表格（应排除）                          |

知识库有效 chunk：7 个（#00-#06）；应排除 chunk：4 个（#07-#10）。

---

## 三、策略评估

### 好的方面

1. **表格完整性：全部保留**。training_zones.md 的功率区间表（6行×5列）完整在 #00 中，没有被切断。MarkdownTextSplitter 在这一点上表现好于字符切分器。

2. **chunk 大小合理**。7 个知识 chunk 平均约 370 字，全部在 300-500 字区间，不存在被硬截断的风险。

3. **语义边界对齐**。每个 chunk 基本对应一个完整的知识主题（区间表、区间应用、FTP、TSS/IF/NP、HRV、恢复原理、恢复策略）。

### 存在的问题

**问题 1（高风险）：`user_profile.md` 混入向量库**

4 个全是空表格的 chunk（#07-#10）进入了向量库。用户问训练状态时，这些 chunk 可能被召回但内容完全是空的，产生无效召回且占用 k=3 的名额。

根本原因：`rag.py` 的 `load_documents()` 同时加载了 `data/knowledge/` 和 `data/user_data/`，但用户数据已经通过 `_fetch_context()` 实时拉取，不应该进入静态向量库。

**修复**：从 `rag.py` 的 `load_documents()` 中移除 `user_dir` 参数，向量库只索引 `data/knowledge/`。删除现有 `chroma_db` 重建。

**问题 2（中风险）：overlap 实际未生效，产生跨 chunk 断裂**

TSS 的公式在 #02 末尾（`TSS = ...`），TSS 的参考值（`<150 恢复快`、`150-300 中等疲劳`）在 #03 开头。#03 以 `**参考值**：` 开头，没有"TSS"字样作为上下文。

影响：用户问"TSS超过300需要恢复几天"时，embedding 需要从 `**参考值**：- <150...` 这段文本推断出这是 TSS 相关内容，bge-small-zh 大概率能处理，但这是一个隐患。

更直接的影响：chunk #03 的开头缺少"TSS"这个词，如果用户用"TSS恢复"作为 query，#03 的 embedding 里没有 TSS 作为锚点词，召回得分可能低于预期。

**修复建议**：把 ftp_and_metrics.md 中的 TSS 小节合并成一个 chunk（可以微调文档结构，或者增大 chunk_size 到 700 让 #02 和 #03 合并）。

**问题 3（低风险）：chunk #04 极短（97字）**

HRV 小节只有 2 行，单独成一个 chunk，信息密度低。用户问 HRV 相关问题时只能召回这 97 个字。可接受，但不理想。

**修复建议**：在 `recovery_and_adaptation.md` 的过度训练信号部分加入 HRV 的应用描述，让两者合并到同一 chunk。

---

## 四、结论

当前策略的核心问题不是 chunk_size 或切分算法的选择，而是**向量库的索引范围设置错误**（混入了 user_profile.md）。排除后有效 chunk 从 11 个降到 7 个，知识库规模很小但覆盖范围完整。

chunk_size=500 + MarkdownTextSplitter 的组合对于这个规模的骑行知识库是合适的，不需要调整切分参数。

优先修复：1）排除 user_profile.md → 重建 chroma_db；2）考虑 ftp_and_metrics.md TSS 跨 chunk 问题对召回的实际影响（在 Layer 3 测 q07 时验证）。
