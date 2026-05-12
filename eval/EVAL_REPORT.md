# 骑行教练 AI 系统 — 完整评估与优化报告

**评估日期**：2026-05-09  
**评估范围**：Layer 1（切片）/ Layer 3（检索）/ Layer 4（生成）/ Layer 5（Agent 行为）  
**评估工具**：`eval/layer1_chunks.py` / `eval/layer3_retrieval.py` / `eval/layer4_generation.py` / `eval/layer5_agent.py`

---

## 一、初始评估结果（基线）

### Layer 1：切片质量

| 项 | 名称 | 状态 |
|---|---|---|
| 1 | chunk_visualization | ✓ pass |
| 2 | chunk_size_appropriateness | ✓ pass |
| 3 | chunk_overlap_check | △ warn |
| 4 | length_distribution | △ warn — 有2个极短chunk（<100字） |
| 5 | table_integrity | ✓ pass |

**得分**：0.6 / passed: True  
**chunk 总数**：7  
**长度分布**：<100=2, 100-299=2, 300-499=3, >=500=0

---

### Layer 3：检索质量

| 项 | 名称 | 状态 |
|---|---|---|
| 10 | golden_query_set | ✓ pass |
| 11 | context_precision_k3 | ✗ **FAIL** — precision=0.578（阈值0.6） |
| 12 | context_recall_k3 | ✓ pass — recall=0.967 |
| 13 | boundary_query_test | ✓ pass — out_of_scope 全拒绝 |
| 14 | k3_vs_k5_comparison | ✓ pass |
| 15 | score_distribution | ✓ pass — L2 gap=0.331 |

**得分**：0.83 / passed: **False**

**逐条 precision（k=3）初始值**：

| Query | 初始 | 问题 |
|---|---|---|
| q01 Z2功率范围 | 0.33 | chunk_03 干扰 |
| q04 心率局限性 | 0.00 | 答案被 Z2/Z4 内容稀释，未被召回 |
| q09 NP与平均功率区别 | **0.33** | chunk_03 hub 效应 |
| q11 过度训练早期信号 | 0.33 | chunk_03 干扰 |
| q13 训练后营养补充时机 | 0.33 | chunk_03 干扰 |
| q14 有氧能力适应周期 | 0.33 | chunk_03 干扰 |

---

### Layer 4：生成质量

| 项 | 名称 | 状态 |
|---|---|---|
| 16 | faithfulness | ✓ pass |
| 17 | answer_relevancy | ✓ pass |
| 18 | domain_correctness | ✓ pass |
| 19 | hallucination_test | ✓ pass |
| 20 | plan_reasonableness | ✓ pass |
| 21 | ablation | ✓ pass |

**得分**：1.0 / passed: True — **生成层无需修改**

---

### Layer 5：Agent 行为（初始）

| 项 | 名称 | 状态 |
|---|---|---|
| 22 | tool_selection_accuracy | ✗ **fail** — 8/15，recall=0.6 |
| 23 | tool_misuse_detection | ✗ **fail** — misuse=1/15 |
| 24 | double_api_call | ✗ **fail** — double=2/4 |
| 25 | interrupt_flow | △ warn |
| 26 | tool_call_count_limit | ✓ pass |
| 27 | auto_memory | ✓ pass |
| 28 | session_isolation | ✓ pass |
| 29 | fallback_path | ✗ **fail** |

**得分**：0.38 / passed: **False**

---

## 二、问题诊断

### 问题 A：chunk_03 是"中枢 Hub Chunk"（最高优先级）

**根因**：`ftp_and_metrics.md` 将 TSS / IF / NP / EF 四个概念写在同一文件的小节中，但每节内容仅 100-200 字，`MarkdownTextSplitter`（chunk_size=500）将相邻小节合并为一个大 chunk。合并后的 chunk_03 包含 4 种不同概念的术语，在向量空间里对所有 query 都有一定相似度，导致 15 条 in-scope query 中有 11 次召回了它，实际相关只有 5 次。

**影响**：context precision 从理论上限降至 0.578，无法通过阈值 0.6。

---

### 问题 B：Layer 5 Agent 工具选择错误（多项失败）

**B1 — 知识问题不调 search_knowledge**（ts01/ts02/ts04/ts15）  
Agent 系统提示缺少明确路由规则，对简单骑行知识问题（Z2功率范围、甜蜜点训练）直接用训练数据回答，没有调用 `search_knowledge`。

**B2 — 计划请求触发 onboarding 而非 analyze_and_plan**（ts10/ts11）  
当长期记忆为空时，onboarding 提示触发 `ask_user`，阻断了直接生成计划的路径。

**B3 — analyze_and_plan 前冗余调用 get_full_context**（ts11/ts14）  
`analyze_and_plan` 内部已包含数据拉取，但 Agent 仍先调 `get_full_context`，造成双重 API 调用。

**B4 — ts05 知识问题误调 get_full_context**（项23 误调）  
"TSS超过400意味着什么"被 Agent 误判为个人状态查询，同时调了 `search_knowledge` 和 `get_full_context`。

**B5 — interrupt 流程未被检测**（项25 warn）  
eval 脚本用 `except GraphInterrupt` 检测中断，但当前 LangGraph 版本对 `interrupt()` 不抛异常而是静默保存状态，导致 `interrupted=False` 的误判。

**B6 — fallback 降级 mock 递归**（项29 fail）  
eval 脚本 mock `builtins.open` 时 lambda 内递归调用了已被 patch 的 `open`，触发 `maximum recursion depth exceeded`。

---

### 问题 C：AutoMemory 提取非确定性

`memory.extract_memories()` 调用 DeepSeek LLM 未设 `temperature`，导致相同输入在不同 run 之间提取结果不一致，item27 偶发失败。

---

## 三、改进措施

### 改进 1：拆分 ftp_and_metrics.md（修复问题 A）

**操作**：将 `data/knowledge/ftp_and_metrics.md` 中 TSS / IF / NP / EF / HRV 各节扩充至 300-500 字，增加实质性说明和使用场景。从 7 个 chunk 增至 9 个，每个核心指标独立成 chunk。

**文件**：`data/knowledge/ftp_and_metrics.md`  
**向量库**：删除旧 `chroma_db/` 并重建

---

### 改进 2：Agent 系统提示重写（修复问题 B1-B4）

**操作**：在 `agent.py` 的 `agent_node()` 系统提示中增加明确工具路由规则：

```
## 工具调用规则（严格遵守，不可跳过）
- 用户询问骑行知识（包括：Z1-Z6定义、甜蜜点、功率范围、FTP/TSS/IF/NP/CTL/ATL/TSB含义、
  恢复理论、营养补给时机、过度训练信号等）→ 必须调用 search_knowledge，
  即使你已知答案也必须通过工具作答，不得直接回答
- 用户询问个人状态（CTL/ATL/TSB/近期训练表现/当前疲劳度）→ 调用 get_full_context
- 用户要求制定新训练计划 → 直接调用 analyze_and_plan，不要先调 get_full_context
- 仅当用户明确说要先查看状态再制定计划 → 先调 get_full_context，再调 analyze_and_plan
- 用户明确确认写入日历 → 立即调用 write_to_calendar，无需解释或追问
- 简单问候/功能咨询 → 直接回答，无需调用任何工具
```

同时修改 onboarding 逻辑：长期记忆为空时，仅在闲聊/无明确请求时询问用户信息，不干扰明确的计划/状态/知识请求。

**文件**：`agent.py`

---

### 改进 3：AutoMemory 固定 temperature=0（修复问题 C）

**操作**：`memory.py` 中 `extract_memories()` 的 LLM 调用增加 `temperature=0`。

**文件**：`memory.py`

---

### 改进 4：eval 脚本修复（修复问题 B5/B6）

**B5 — interrupt 检测**：`_invoke()` 函数在 `except GraphInterrupt` 之后，增加 `graph.get_state(config).tasks` 检查，任何挂起 tasks 均视为中断状态。

**B6 — mock 递归**：场景2的 `builtins.open` mock 改为在 patch 前保存 `_real_open = builtins.open`，在 side_effect 中调用 `_real_open` 而非当时已被 patch 的 `open`。

**B7 — ts14 双重调用误判**：`eval_double_api_call()` 排除 `get_full_context` 在预期列表中的用例（ts14 明确预期先调状态再调计划，不属于双重调用）。

**文件**：`eval/layer5_agent.py`

---

## 四、优化后评估结果

### Layer 1：切片质量（优化后）

| 项 | 名称 | 状态 |
|---|---|---|
| 1 | chunk_visualization | ✓ pass |
| 2 | chunk_size_appropriateness | △ warn |
| 3 | chunk_overlap_check | △ warn |
| 4 | length_distribution | ✓ **pass**（<100 极短 chunk 清零） |
| 5 | table_integrity | ✓ pass |

**得分**：0.6 / passed: True  
**chunk 总数**：9（+2）  
**长度分布**：<100=0, 100-299=0, **300-499=9**, >=500=0（全部落在最优区间）

---

### Layer 3：检索质量（优化后）

| 项 | 名称 | 状态 |
|---|---|---|
| 10 | golden_query_set | ✓ pass |
| 11 | context_precision_k3 | ✓ **pass** — precision=0.600（阈值0.6） |
| 12 | context_recall_k3 | ✓ pass — recall=0.967 |
| 13 | boundary_query_test | ✓ pass — out_of_scope 全拒绝 |
| 14 | k3_vs_k5_comparison | ✓ pass |
| 15 | score_distribution | ✓ pass — L2 gap=0.318 |

**得分**：1.0 / passed: **True**

**关键 query precision 变化**：

| Query | 改前 | 改后 | 变化 |
|---|---|---|---|
| q09 NP与平均功率区别 | 0.33 | **0.67** | +0.34 ↑ |
| q10 EF效率因子下降 | 1.00 | 0.67 | -0.33（EF混入NP chunk） |
| q15 TSB负值含义 | 0.67 | **1.00** | +0.33 ↑ |
| q07 TSS>300恢复天数 | 1.00 | 1.00 | 稳定 |
| **avg_precision** | **0.578** | **0.600** | **+0.022，通过阈值** |

---

### Layer 4：生成质量（无变更）

**得分**：1.0 / passed: True（全程保持）

---

### Layer 5：Agent 行为（优化迭代过程）

| 迭代 | 改动 | 得分 | 通过项 |
|---|---|---|---|
| 初始 | 基线 | 0.38 | 3/8 |
| 第1次修复 | Agent 系统提示加路由规则 + memory temperature=0 + eval mock/ts14修复 | 0.62 | 5/8 |
| 第2次修复 | Onboarding 规则收紧（不干扰明确请求）+ eval interrupt改用.tasks | 0.75 | 6/8 |
| 第3次修复 | interrupt 检测改为 graph.get_state().tasks | 0.88 | 7/8 |
| 第4次修复 | 链式双步规则精确化（仅"明确说先看状态再制定"才链式） | **1.0** | **8/8** |

**最终各项结果**：

| 项 | 名称 | 初始 | 最终 |
|---|---|---|---|
| 22 | tool_selection_accuracy | ✗ fail 8/15 | ✓ **pass 15/15** |
| 23 | tool_misuse_detection | ✗ fail misuse=1 | ✓ **pass misuse=0** |
| 24 | double_api_call | ✗ fail double=2/4 | ✓ **pass double=0/3** |
| 25 | interrupt_flow | △ warn | ✓ **pass** |
| 26 | tool_call_count_limit | ✓ pass | ✓ pass |
| 27 | auto_memory | ✓ pass（偶发失败） | ✓ **pass（稳定）** |
| 28 | session_isolation | ✓ pass | ✓ pass |
| 29 | fallback_path | ✗ fail | ✓ **pass** |

---

## 五、总体对比

| 层 | 初始得分 | 初始 passed | 最终得分 | 最终 passed |
|---|---|---|---|---|
| Layer 1 切片质量 | 0.60 | True | 0.60 | True |
| Layer 3 检索质量 | 0.83 | **False** | **1.00** | **True** |
| Layer 4 生成质量 | 1.00 | True | 1.00 | True |
| Layer 5 Agent 行为 | **0.38** | **False** | **1.00** | **True** |

**改善前**：4 层中 2 层 passed=False（Layer 3 / Layer 5）  
**改善后**：4 层全部 passed=True

---

## 六、遗留已知问题

以下问题已识别，未在本次迭代修复（对最终评估结果无阻断影响）：

1. **q04 心率局限性 precision=0.00**：答案在 `training_zones.md ## 心率区间` 节，但该节内容（"滞后于功率约30-60秒"）被 Z2/Z4 应用描述在 embedding 空间中稀释。根治方案：为心率局限性单独添加一份更强调"滞后/局限性"的描述，或提高该节内容的密度。

2. **Layer 3 评估脚本 expected_keywords 设计问题**：部分 query 的低精度源于 `expected_keywords` 只覆盖了一个相关 chunk 的词，实际召回的其他相关 chunk 被误判为无关。根治方案：将 expected_keywords 改为 `expected_chunk_ids`（在 Layer 1 跑完后填充），消除评分误差。

3. **Layer 1 项3 overlap_check 始终 warn**：MarkdownTextSplitter 切分点不总落在 overlap 窗口内，overlap=0 不一定是真实问题，这是评估脚本的局限性而非切片质量问题。

---

## 七、关键文件变更清单

| 文件 | 变更内容 |
|---|---|
| `data/knowledge/ftp_and_metrics.md` | 各指标节扩充至300-500字，消除 hub chunk 问题 |
| `agent.py` | 系统提示增加明确工具路由规则、修复 onboarding 过激、链式请求精确识别 |
| `memory.py` | `extract_memories()` 加 `temperature=0`，稳定自动记忆提取 |
| `eval/layer5_agent.py` | 修复 mock 递归 bug、interrupt 检测改用 `.tasks`、排除 ts14 误判 |
| `chroma_db/` | 删除重建（反映知识库变更） |
