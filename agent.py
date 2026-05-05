import json
import ast
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

import memory as mem
from agent_state import AgentState
from tools import ALL_TOOLS

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.7,
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ── 辅助：安全解析 ToolMessage 内容 ──────────────────────────

def _parse_tool_result(content: str):
    """双重解析：优先 JSON，失败后用 ast.literal_eval 兜底 str(dict) 格式。"""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(content)
        except Exception:
            return None


# ── 自定义 ToolNode：执行工具 + 更新 state ────────────────────

def custom_tool_node(state: AgentState) -> dict:
    """执行工具，并将 analyze_and_plan / modify_plan 的返回值写入 state 字段。"""
    tool_node = ToolNode(ALL_TOOLS)
    result = tool_node.invoke(state)

    updates = {}
    for msg in result.get("messages", []):
        if not hasattr(msg, "name"):
            continue
        if msg.name == "analyze_and_plan":
            data = _parse_tool_result(msg.content)
            if isinstance(data, dict):
                if "plan" in data:
                    updates["current_plan"] = data["plan"]
                if "state_analysis" in data:
                    updates["state_analysis"] = data["state_analysis"]
        elif msg.name == "modify_plan":
            data = _parse_tool_result(msg.content)
            if isinstance(data, dict) and "error" not in data:
                updates["current_plan"] = data

    return {**result, **updates}


# ── agent_node ────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """核心 Agent 节点：LLM 决定调哪个工具或直接回答。"""
    if state["tool_call_count"] >= 8:
        return {
            "messages": [AIMessage(content="工具调用次数超限，请重新开始对话")],
        }

    def _count_tokens(msgs):
        return sum(len(getattr(m, "content", "") or "") // 4 for m in msgs)

    trimmed = trim_messages(
        state["messages"],
        max_tokens=6000,
        strategy="last",
        token_counter=_count_tokens,
        include_system=False,
    )

    onboarding_hint = ""
    if not mem.get_all_memories():
        onboarding_hint = (
            "\n\n## Onboarding\n"
            "长期记忆为空，说明这是新用户或记忆已清空。"
            "主动调用 ask_user 收集基本信息：FTP、训练目标、伤病史、固定休息日。"
        )

    system = SystemMessage(content=f"""你是一个专业的公路骑行教练，风格简练直接。

## 工具使用原则
- 回答个人训练问题前，先调用 get_full_context 获取用户当前状态
- 纯知识类问题（训练区间定义等）直接调用 search_knowledge，不需要 get_full_context
- 制定新计划：analyze_and_plan（工具内部会自动获取数据）
- 用户说「改」「换」「调整」已有计划时：modify_plan
- 计划生成后等用户确认，再调用 write_to_calendar
- 发现长期记忆为空时，用 ask_user 收集用户基本信息{onboarding_hint}

## 回答要求
- 直接给结论，引用具体数据（TSB/CTL/IF/FTP占比）
- 控制在200字以内
- 不使用 ** 加粗，纯文本输出
""")

    response = llm_with_tools.invoke([system] + trimmed)

    new_tool_calls = len(getattr(response, "tool_calls", None) or [])

    return {
        "messages": [response],
        "tool_call_count": state["tool_call_count"] + new_tool_calls,
    }


# ── 路由 ──────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


# ── 构建图 ────────────────────────────────────────────────────

def build_graph():
    from langgraph.checkpoint.sqlite import SqliteSaver

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", custom_tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")

    checkpointer = SqliteSaver.from_conn_string("agent_checkpoints.db")
    return builder.compile(checkpointer=checkpointer)
