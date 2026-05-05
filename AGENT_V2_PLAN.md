# cycling-coach Agent v2 设计方案

## 背景

将现有 RAG + 固定路由的 workflow 架构升级为 LangGraph Agent 架构。
核心驱动：write_to_calendar 需要 human-in-the-loop 确认，LangGraph interrupt() 原语原生支持。

---

## 架构决策（不可修改）

1. **框架：LangGraph** — 唯一原因是 interrupt() 原生支持
2. **路由：无显式路由节点** — LLM 通过 tool description 自主选择工具
3. **AgentState 极简** — 只放 messages 解决不了的字段
4. **工具数量：6个** — 无功能交集，遵循"更少但更好"原则
5. **安全：三层** — recursion_limit（invoke config）、tool_call_count、write 前代码层 interrupt

---

## 问题清单与修复方案

原始设计有 9 个问题，另发现 3 个真实 bug，1 个设计权衡。

### 原始 9 个问题（全部修复）

| # | 问题 | 修复方式 |
|---|---|---|
| 1 | interrupt_before=["write_to_calendar"] 拦截工具名无效 | 改用 interrupt() 原语在工具内部触发 |
| 2 | current_plan 永远不会写入 state | 自定义 ToolNode，捕获工具返回值更新 state |
| 3 | tool_call_count 只检查不递增 | agent_node 统计每次响应的 tool_calls 数并累加 |
| 4 | user_confirmed 与 interrupt() 机制重复 | 删掉该字段，confirm/cancel 走 Command(resume=...) |
| 5 | /confirm 用 graph.invoke(None, config) | 改为 graph.invoke(Command(resume=value), config) |
| 6 | /ask 每次传 tool_call_count: 0 覆盖 checkpoint | 有 checkpoint 时只传新消息，不传初始值 |
| 7 | analyze_and_plan 描述自相矛盾 | 工具完全自包含，内部调用 _fetch_context()，删掉前置条件说明 |
| 8 | ask_user 只描述意图无实现 | 用 interrupt({"type": "input", "question": q}) 实现 |
| 9 | trim_messages(token_counter=llm) DeepSeek 不稳定 | 改为字符数估算函数 |

### 新发现 3 个真实 bug

**Bug A：ToolMessage 内容解析可能失败**

自定义 ToolNode 用 json.loads 解析工具返回值。langgraph 0.1.x 时用 str(dict) 存储（单引号），
json.loads 会抛异常。当前 0.2+ 版本用 json.dumps，但需要防御性处理。

修复：双重解析兜底
```python
def _parse_tool_result(content):
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(content)
        except:
            return None
```

**Bug B：graph.invoke() 在 interrupt() 触发时可能抛 GraphInterrupt 异常**

interrupt() 在工具内触发时，部分版本 invoke() 会抛 GraphInterrupt 而非正常返回。
不 catch 会导致 /ask 接口 500。

修复：用 try/except GraphInterrupt 包裹 invoke()，之后统一从 get_state() 读 interrupt 状态。

访问路径（已验证是官方 API）：
```python
snapshot = graph.get_state(config)
if snapshot.next:
    payload = snapshot.tasks[0].interrupts[0].value
```

**Bug C：recursion_limit 放在 compile() 里无效**

compile() 不接受 recursion_limit 参数，会被 **kwargs 静默吞掉，安全限制假设生效。

修复：移到每次 invoke() 的 config 里：
```python
config = {"configurable": {"thread_id": session_id}, "recursion_limit": 10}
```

### 设计权衡（接受，不改）

analyze_and_plan 内部调 _fetch_context()，get_full_context 也调。LLM 先问状态再制定计划时
会触发两次 API 调用。接受双重调用，数据最新，实现简单。如需优化，analyze_and_plan 可以
优先读 messages 历史里已有的 context，但增加的复杂度不值得。

---

## 文件变更一览

| 文件 | 操作 | 说明 |
|---|---|---|
| agent_state.py | 新建 | AgentState TypedDict |
| tools.py | 新建 | 6个工具 + 辅助函数 |
| agent.py | 新建 | LangGraph 图结构 |
| api.py | 完全重写 | 接口层，支持 interrupt 流程 |
| index.html | 更新 | 前端支持 awaiting_confirmation / awaiting_input |
| strava_client.py | 不动 | |
| intervals_client.py | 不动 | |
| memory.py | 不动 | |
| rag.py | 不动 | tools.py 复用其中逻辑 |
| data/, chroma_db/ | 不动 | |

---

## API 接口

### POST /ask
请求：`{"question": "...", "session_id": "..."}`

响应类型 1（普通回复）：
```json
{"type": "message", "answer": "..."}
```

响应类型 2（等待确认写入日历）：
```json
{"type": "awaiting_confirmation", "plan": {...}, "message": "..."}
```

响应类型 3（等待用户输入）：
```json
{"type": "awaiting_input", "question": "..."}
```

### POST /resume
请求：`{"session_id": "...", "value": true}` 或 `{"session_id": "...", "value": "用户回答文字"}`

响应：同 /ask 的三种类型之一（可能链式触发下一个 interrupt）

### GET /plan/current?session_id=xxx
响应：`{"plan": {...}}` 或 `{"plan": null}`

### GET /memory/list
### DELETE /memory/clear

---

## AgentState

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    current_plan: Optional[dict]      # custom_tool_node 在工具返回后写入
    state_analysis: Optional[dict]    # 同上
    tool_call_count: int              # agent_node 累加，超 8 次报错
    # user_confirmed 已删除
```

---

## 图结构

```
START → agent_node → (有 tool_calls?) → tools(custom) → agent_node → ...
                   → (无 tool_calls?) → END
```

tools 节点内部 interrupt() 触发时图暂停，客户端通过 /resume 恢复。

---

## 前端状态机

```
idle
  ↓ sendMessage / generatePlan (POST /ask)
loading
  ↓ 响应
  ├── type=message → 显示气泡 → idle
  ├── type=awaiting_confirmation → showConfirmCard() → pending_confirm
  └── type=awaiting_input → showInputPrompt() → pending_input

pending_confirm
  ↓ 用户点确认/取消 (POST /resume, value=true/false)
  → 处理响应（同 loading 的三个分支）

pending_input
  ↓ 用户输入提交 (POST /resume, value="文字")
  → 处理响应（同 loading 的三个分支）
```

---

## 服务启动

```bash
# 用 8001 端口，避免与 main branch 的 8000 冲突
uvicorn api:app --host 0.0.0.0 --port 8001

# 基本测试
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "我今天状态怎么样？", "session_id": "test-001"}'
```
