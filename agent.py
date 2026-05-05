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


# ── 自定义 ToolNode：执行工具 + 更新 state + 替换 LLM 可见内容 ──

def custom_tool_node(state: AgentState) -> dict:
    """执行工具，提取 current_plan 写入 state，并将工具输出替换为对用户友好的文本。"""
    tool_node = ToolNode(ALL_TOOLS)
    result = tool_node.invoke(state)

    updates = {}
    new_messages = []

    for msg in result.get("messages", []):
        if not hasattr(msg, "name"):
            new_messages.append(msg)
            continue

        if msg.name in ("analyze_and_plan", "modify_plan"):
            data = _parse_tool_result(msg.content)
            if isinstance(data, dict) and "error" not in data:
                # 提取结构化数据写入 state
                if msg.name == "analyze_and_plan":
                    updates["current_plan"] = data.get("plan")
                    updates["state_analysis"] = data.get("state_analysis")
                else:
                    # modify_plan 直接返回 plan（含 summary + events）
                    plan = {k: v for k, v in data.items() if k != "_display"}
                    updates["current_plan"] = plan

                # 用 _display 字段替换 ToolMessage 内容，LLM 直接透传给用户
                display = data.get("_display", msg.content)
                msg = msg.model_copy(update={"content": display})

        new_messages.append(msg)

    result["messages"] = new_messages
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

工具调用后，将工具返回的内容原文展示给用户，不要改写或缩减。
普通回答控制在150字以内，不使用 markdown 格式或加粗符号。
引用具体数据支撑判断（CTL/ATL/TSB/FTP占比）。{onboarding_hint}
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
    import sqlite3
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

    # from_conn_string() 在 3.x 是 context manager，直接传连接对象替代
    conn = sqlite3.connect("agent_checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)
