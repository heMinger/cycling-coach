# 面试准备 — cycling-coach Agent 设计复习

## 一、项目概览

公路骑行 AI 教练，LangGraph Agent + RAG + 记忆系统。用户通过 Web UI 对话，Agent 调用工具获取训练数据、检索知识、生成计划、写入日历。

技术栈：LangGraph / LangChain / FastAPI / Chroma / SQLite / DeepSeek API / Strava API / Intervals.icu API

---

## 二、Agent 架构

```
START → agent_node → (有 tool_calls?) → custom_tool_node → agent_node → ...
                    → (无 tool_calls?) → END
```

- **单 Agent 多工具**，无多 Agent 通信
- **无显式路由节点**，LLM 通过 tool description 自主选择工具
- **Custom ToolNode**：拦截工具返回值，分流 `_display`（进 messages）和结构化数据（进 state）

---

## 三、6 个 Tools

| # | 工具 | 触发条件 | 核心机制 |
|---|------|---------|---------|
| `get_full_context` | 用户问个人状态（CTL/ATL/TSB/疲劳） | Strava + Intervals 并行拉取，三层降级：实时 → 缓存 → 静态档案 |
| `search_knowledge` | 用户问通用骑行知识（Z2定义、TSS含义等） | Chroma 向量检索，L2 距离阈值 0.85 过滤噪声 |
| `analyze_and_plan` | 用户要制定新训练计划 | 内部自包含（自动拉数据），两步 LLM：Step1 状态分析 → Step2 生成逐天计划 |
| `modify_plan` | 用户要修改已有计划 | 通过 InjectedState 读取 current_plan，精确修改而非重生成 |
| `write_to_calendar` | 用户确认写入日历 | interrupt() 二次确认后写入 Intervals.icu，不可逆 |
| `ask_user` | 新用户 onboarding / 请求模糊 | interrupt() 暂停等待用户输入 |

**Tool Description 设计模板**：功能一句话 → 适用场景 → 不适用场景 → 参数说明。Description 即路由，关键是"唯一正确途径"和"不适用"这种 guardrail 语言。

---

## 四、Context Engineering（五层结构）

```
第1层：System Prompt（固定）
  - 角色设定："专业公路骑行教练，风格简练直接"
  - 6条工具路由规则（精确到关键词的条件判断）
  - 输出约束（150字、不用 markdown）
  - Onboarding hint（动态，仅新用户）

第2层：Messages History
  - trim_messages，max 6000 tokens
  - 用字符数估算 token（不用 LLM token count，DeepSeek 不稳定）
  - strategy="last"（保留最近消息）

第3层：Tool 返回的实时数据（按需注入）
  - get_full_context → Strava + Intervals
  - search_knowledge → Chroma 知识库
  - analyze_and_plan → 结构化计划（_display 透传）

第4层：长期记忆（按需注入）
  - analyze_and_plan 和 get_full_context 内部读取
  - 四类：伤病/偏好/目标赛事/固定日程

第5层：Agent State（LLM 不可见，代码层用）
  - current_plan, state_analysis, tool_call_count
```

核心原则：**固定角色设定 + 按需动态注入，不给 LLM 塞冗余信息。**

---

## 五、Agent State 设计

只有 6 个字段，原则是"只放 messages 解决不了的字段"：

| 字段 | 类型 | 为什么需要 |
|------|------|-----------|
| messages | Annotated[list, add_messages] | LangGraph 核心，自动合并追加 |
| session_id | str | 线程隔离 |
| current_plan | Optional[dict] | 跨 turn 共享计划，modify_plan / write_to_calendar 依赖它 |
| state_analysis | Optional[dict] | 状态分析结果 |
| tool_call_count | int | 安全限制，超 8 次报错 |

AgentState 与 messages 的分工：messages 放 LLM 上下文，state 放跨 turn 的代码级数据，不重复。存入 state 的条件三条缺一不可：LLM 不需要看、跨 turn 持久化、工具需要读写。

---

## 六、Custom ToolNode — 结构化数据回流

核心架构创新。LangGraph 默认 ToolNode 只把返回值放入 messages，不更新 state。custom_tool_node 同时做两件事：

1. **提取结构化数据写入 state**：从 analyze_and_plan / modify_plan 的返回中提取 plan 和 state_analysis
2. **替换 LLM 可见内容**：用 `_display` 字段替换 ToolMessage.content，格式化文本直接透传给用户

`_parse_tool_result` 做了双重解析兜底（json.loads + ast.literal_eval），解决不同 LangGraph 版本序列化方式不同的问题。

---

## 七、Human-in-the-Loop

v1→v2 升级的核心驱动。两个 interrupt 场景：

1. **write_to_calendar（确认）**：`interrupt({"type": "confirm", "plan": ..., "message": ...})`
2. **ask_user（输入）**：`interrupt({"type": "input", "question": q})`

前端状态机：`idle → loading → message / awaiting_confirmation / awaiting_input`

**三种安全层**：
1. `recursion_limit=10`（每次 invoke config，不在 compile 里，会被 **kwargs 吞掉）
2. `tool_call_count >= 8`（agent_node 内）
3. `write_to_calendar` 代码层 interrupt()（不可逆操作的最后防线）

**关键 Bug 修复**：interrupt() 在工具内触发时，部分版本 invoke() 抛 GraphInterrupt 而非正常返回。解决：try/except 包裹，统一从 `graph.get_state(config).tasks` 读 interrupt payload。

---

## 八、Memory 系统（双轨制）

| 层级 | 存储 | 生命周期 | 用途 |
|------|------|---------|------|
| 短期 | conversations 表 | 30天过期 | 最近5轮对话原文 |
| 长期 | memories 表 | 持久 | LLM 自动提取的关键信息 |

**AutoMemory 触发**：关键词即时触发（"膝盖""受伤""比赛""偏好"等14个词）+ 每5轮批量触发。分类：injury / preference / goal / schedule。

**Onboarding hint**：`if not mem.get_all_memories()` → 动态注入引导指令，老用户不浪费 token。注意加了"如果用户明确要求制定计划，优先执行请求，不要先问问题"防止 LLM 拦截明确请求。

---

## 九、RAG 知识检索

- Embedding：BAAI/bge-small-zh-v1.5（中文优化，CPU 推理）
- Chunking：MarkdownTextSplitter，chunk_size=500，overlap=50
- 检索：相似度搜索 k=5，L2 距离阈值 0.85（做过 threshold scan，0.85 是 precision/recall 最优）
- 低于阈值的 chunk 不返回，明确告知 LLM"未找到"而非塞噪声

---

## 十、四层评估体系

### 覆盖范围

| Layer | 测什么 | 核心指标 |
|-------|--------|---------|
| L1 | 切片质量 | chunk 数量、长度分布、表完整性 |
| L3 | 检索质量 | precision / recall（k=3, k=5），L2 距离分布 |
| L4 | 生成质量 | faithfulness, relevancy, hallucination, plan reasonableness |
| L5 | Agent 行为 | 工具选择准确率、误调、双重 API 调用 |

### Layer 5 工具选择准确率变化（核心数据）

**15 条 golden test cases**，每条有 expected tools + anti tools。

**初始问题诊断**：
- B1：知识问题不调 search_knowledge（LLM 直接回答）
- B2：计划请求触发 onboarding（ask_user 拦截了 analyze_and_plan）
- B3：双重 API 调用（analyze_and_plan 前先调了 get_full_context）
- B4：工具误调（TSS 指标解读调了 get_full_context）

**迭代过程**：

| 迭代 | 改动 | 得分 | 通过项 |
|------|------|------|--------|
| 初始 | 基线 | 0.38 | 3/8 |
| 第1次 | Agent 系统提示加路由规则 + memory temperature=0 + eval mock 修复 | 0.62 | 5/8 |
| 第2次 | Onboarding 规则收紧 | 0.75 | 6/8 |
| 第3次 | interrupt 检测改用 .tasks | 0.88 | 7/8 |
| 第4次 | 链式双步规则精确化 | **1.00** | **8/8** |

**最终数据**：

| 指标 | 改前 | 改后 |
|------|------|------|
| 工具选择 recall | 0.6 (8/15) | 1.0 (15/15) |
| 工具误调 | 1 次 | 0 次 |
| 双重 API 调用 | 2/4 | 0/3 |

核心改动不是代码逻辑，而是 tools description + system prompt 的路由规则。通过 eval 量化了 description 优化对准确率的影响。测试用了 anti 字段做反向验证（不该调的工具绝对不调），比单纯测 recall 更严格。

---

## 十一、LangGraph vs LangChain

| 维度 | LangChain | LangGraph |
|------|-----------|-----------|
| 核心抽象 | Chain / Runnable（线性管道） | StateGraph（有向图，节点+边） |
| 流程控制 | 固定顺序，A → B → C | 条件路由，A → B 或 C，可循环回 A |
| 状态管理 | 无内置持久化 | 内置 State + Checkpointer，跨轮次持久化 |
| Human-in-the-Loop | 需自己实现 | interrupt() 原生支持，Command(resume=) 恢复 |
| 适用场景 | 单次问答、简单 RAG | 多步 Agent、需确认的操作、复杂工作流 |

不是替代关系。LangGraph 的节点内部用 LangChain 的 LLM/prompt/retriever，LangGraph 的 value add 是编排层。

---

## 十二、其他技术点

### 三层数据降级
实时 API（Strava + Intervals 并行）→ 缓存文件 → 静态档案。确保服务在任何情况下都能给出回应。

### 社团发现算法（NetworkX + Gephi）
- 使用 NetworkX 的 Louvain 实现做社团发现
- Louvain 通过最大化模块度（Modularity）划分社团，两阶段迭代：局部移动节点 + 压缩社团
- 对每个节点计算模块度贡献的 Z-score 着色，暖色=核心成员，冷色=边缘/桥梁节点
- 导出 GraphML 到 Gephi，用 Force Atlas 2 布局可视化

### 为什么选 Louvain
速度快（近乎线性），能处理大规模图；比标签传播（LPA）结果稳定；比 Girvan-Newman 能算动大图。

---

## 十三、Tool Description 迭代过程

三个版本：V1（0db386c）→ V2（ffd31fa）→ V3（324c4fb，当前版本）。

### V1 → V2：Description 语言策略升级

改动最大的一次，核心是 **加了"唯一途径"排除性语言**。

| 工具 | V1 描述 | V2 描述 |
|------|---------|---------|
| `get_full_context` | "对话开始时...调用一次" | "获取用户个人训练数据的**唯一途径**" |
| `search_knowledge` | "在知识库中检索相关知识" | "回答通用骑行训练知识问题的**唯一途径**" |
| `analyze_and_plan` | "分析状态并生成个性化训练计划" | "制定新计划的**唯一正确途径**" |
| `modify_plan` | "对已生成的计划进行精确修改" | "修改已有计划的**唯一正确途径**" |

**消除冗余调用**：`analyze_and_plan` 加了"内部自动获取实时数据，**无需提前调用 get_full_context**"。V1 没这句，eval 显示 double_api_call=2/4。

**`write_to_calendar` 参数变化**：V1 参数是 `plan: dict`（LLM 得自己传计划），V2 改成 `state: Annotated[dict, InjectedState]`（"无需传入任何参数，工具自动读取已生成的计划"）。把 LLM 的责任移到了代码层。

**新增 `_display` 机制**：`analyze_and_plan` 和 `modify_plan` 返回 dict 里加了 `_display` 字段（格式化后的中文文本），custom_tool_node 用这个替换 ToolMessage.content。LLM 收到的是可直接展示的文本而不是 JSON，直接透传给用户。

### V2 → V3：Description 没变，改实现

Description 本身不再改动，`search_knowledge` 的技术优化解决了"给了噪声"的问题：

- k=3 → k=5，加 L2 距离阈值 0.85（做过 threshold scan，0.85 是 precision/recall 最优）
- embedding 从每次调用加载改为模块级单例（避免重复加载模型）
- 低于阈值的 chunk 明确返回"知识库未找到"而不是塞噪声 chunk

### 总结

两次迭代解决了两类不同问题：

1. **V1→V2 解决"选错工具"** — 通过"唯一途径" guardrail 语言 + 负向路由（"不适用：xxx"）+ 消除冗余调用提示。工具选择 recall 从 0.6 → 1.0。

2. **V2→V3 解决"工具返回噪声"** — Description 没动，改检索质量：加阈值过滤 + 单例优化 + 明确未找到信号。

核心认知：**Description 决定 LLM 调不调对工具，阈值决定工具返回的东西干不干净。** 两者都是 context quality 的关键环节。
