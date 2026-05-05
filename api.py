import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import BaseModel

import memory as mem
from agent import build_graph

# ── 数据模型 ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    session_id: str = "default"

class ResumeRequest(BaseModel):
    session_id: str
    value: Any  # true/false 确认，或字符串（ask_user 的回答）

# ── 全局图实例 ────────────────────────────────────────────────

graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    mem.init_db()
    mem.clean_old_conversations()
    graph = build_graph()
    print("Agent 图初始化完成，服务就绪")
    yield
    print("服务关闭")

# ── FastAPI 实例 ──────────────────────────────────────────────

app = FastAPI(
    title="公路骑行 AI 教练",
    description="LangGraph Agent + 记忆系统的个性化骑行训练助手",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 辅助：读取 interrupt payload ──────────────────────────────

def _get_interrupt_payload(config: dict):
    """从 graph state 读取 interrupt 的 payload，返回 None 表示无 interrupt。"""
    snapshot = graph.get_state(config)
    if not snapshot.next:
        return None
    for task in (snapshot.tasks or []):
        for it in (getattr(task, "interrupts", None) or []):
            return it.value
    return None


def _build_interrupt_response(payload: dict) -> dict:
    if payload.get("type") == "confirm":
        return {
            "type": "awaiting_confirmation",
            "plan": payload.get("plan"),
            "message": payload.get("message", "请确认是否写入日历"),
        }
    return {
        "type": "awaiting_input",
        "question": payload.get("question", ""),
    }


# ── POST /ask ─────────────────────────────────────────────────

@app.post("/ask")
async def ask(request: AskRequest):
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = request.session_id
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 10}

    mem.get_or_create_session(session_id)
    mem.save_message(session_id, "user", request.question)

    # 有 checkpoint 则只追加新消息，避免覆盖 tool_call_count 等字段
    snapshot = graph.get_state(config)
    if snapshot.values:
        input_state = {"messages": [HumanMessage(content=request.question)]}
    else:
        input_state = {
            "messages": [HumanMessage(content=request.question)],
            "session_id": session_id,
            "tool_call_count": 0,
            "current_plan": None,
            "state_analysis": None,
        }

    try:
        result = await asyncio.to_thread(graph.invoke, input_state, config)
    except GraphInterrupt:
        result = None

    # 检查是否被 interrupt
    payload = _get_interrupt_payload(config)
    if payload is not None:
        return _build_interrupt_response(payload)

    # 正常响应
    last_message = result["messages"][-1].content
    mem.save_message(session_id, "assistant", last_message)

    turn_count = mem.increment_turn(session_id)
    asyncio.create_task(
        asyncio.to_thread(
            mem.process_memory_update, session_id, request.question, turn_count
        )
    )

    return {"type": "message", "answer": last_message}


# ── POST /resume ──────────────────────────────────────────────

@app.post("/resume")
async def resume(request: ResumeRequest):
    """用户确认或回答后，resume 被 interrupt 暂停的图。"""
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent 未初始化")

    session_id = request.session_id
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 10}

    try:
        result = await asyncio.to_thread(
            graph.invoke, Command(resume=request.value), config
        )
    except GraphInterrupt:
        result = None

    # 可能链式触发了下一个 interrupt
    payload = _get_interrupt_payload(config)
    if payload is not None:
        return _build_interrupt_response(payload)

    last_message = result["messages"][-1].content
    mem.save_message(session_id, "assistant", last_message)

    turn_count = mem.increment_turn(session_id)
    asyncio.create_task(
        asyncio.to_thread(
            mem.process_memory_update, session_id, "", turn_count
        )
    )

    return {"type": "message", "answer": last_message}


# ── GET /plan/current ─────────────────────────────────────────

@app.get("/plan/current")
async def get_current_plan(session_id: str = "default"):
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    if snapshot.values:
        return {"plan": snapshot.values.get("current_plan")}
    return {"plan": None}


# ── 记忆管理接口 ──────────────────────────────────────────────

@app.get("/memory/list")
def list_memories():
    return {"memories": mem.get_all_memories()}


@app.delete("/memory/clear")
def clear_memories():
    import sqlite3
    conn = sqlite3.connect(mem.DB_PATH)
    conn.execute("DELETE FROM memories")
    conn.commit()
    conn.close()
    return {"status": "cleared"}


# ── 静态文件 ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "service": "cycling-coach", "version": "2.0.0"}

@app.get("/ui")
def ui():
    return FileResponse("index.html")
