from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import json

from rag import (
    load_documents, split_documents, build_vectorstore, build_chain,
    route_request, run_plan_pipeline
)
import memory as mem

# ── 数据模型 ──────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    constraints: dict | None = None
    session_id: str = "default"     # 前端传入，用于关联历史

class AskResponse(BaseModel):
    question: str
    answer: str
    intent: str = "general"
    plan_result: dict | None = None

class PlanRequest(BaseModel):
    plan: dict
    constraints: dict

# ── 全局状态 ──────────────────────────────────────────────────
chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    print("正在初始化 RAG 系统...")

    # 初始化数据库
    mem.init_db()
    mem.clean_old_conversations()

    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    chain = build_chain(vectorstore)
    print("RAG 系统初始化完成，服务就绪")
    yield
    print("服务关闭")

# ── FastAPI 实例 ──────────────────────────────────────────────
app = FastAPI(
    title="公路骑行 AI 教练",
    description="基于 RAG + 记忆系统的个性化骑行训练系统",
    version="0.3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 接口 ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "service": "cycling-coach", "version": "0.3.0"}

@app.get("/ui")
def ui():
    return FileResponse("index.html")

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if chain is None:
        raise HTTPException(status_code=503, detail="RAG 系统未初始化")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = request.session_id
    question = request.question

    # 确保会话存在
    mem.get_or_create_session(session_id)

    # 获取短期历史（最近5轮）
    recent_messages = mem.get_recent_messages(session_id)
    short_term_history = mem.format_short_term_history(recent_messages)

    # 意图路由
    intent = await asyncio.to_thread(route_request, question)
    result = None

    if intent == "plan":
        try:
            result = await asyncio.to_thread(run_plan_pipeline, question, short_term_history, request.constraints)
            analysis = result["state_analysis"]
            plan = result["plan"]
            label = "计划草案" if plan["status"] == "draft" else "需要补充信息"
            answer = f"""状态分析：
{analysis.get('tsb_interpretation', '')}
本周类型：{analysis.get('week_type', '')}
{analysis.get('reasoning', '')}

{label}（共{len(plan.get('events', []))}天）：{plan.get('summary', '')}

请在「训练计划」标签查看草案、假设和待确认问题。"""
        except Exception as e:
            print(f"计划生成失败：{e}")
            answer = f"计划生成失败：{e}，请切换到「训练计划」标签重试。"
    else:
        # 注入短期历史
        answer = await chain.ainvoke({
            "question": question,
            "short_term_history": short_term_history
        })

    # 保存本轮对话到 SQLite
    mem.save_message(session_id, "user", question, intent)
    saved_answer = answer
    if result is not None:
        saved_answer += "\n训练草案（供后续调整）：" + json.dumps(result['plan'], ensure_ascii=False)
    mem.save_message(session_id, "assistant", saved_answer, intent)

    # 更新轮数，触发 AutoMemory（异步后台执行，不阻塞响应）
    turn_count = mem.increment_turn(session_id)
    asyncio.create_task(
        asyncio.to_thread(mem.process_memory_update, session_id, question, turn_count)
    )

    return AskResponse(question=question, answer=answer, intent=intent, plan_result=result)


@app.post("/plan/generate")
async def generate_plan(request: AskRequest):
    try:
        history = mem.format_short_term_history(mem.get_recent_messages(request.session_id))
        result = await asyncio.to_thread(run_plan_pipeline, request.question, history, request.constraints)
        mem.get_or_create_session(request.session_id)
        mem.save_message(request.session_id, "user", request.question, "plan")
        mem.save_message(request.session_id, "assistant", json.dumps(result['plan'], ensure_ascii=False), "plan")
        return result
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="计划生成服务暂不可用")


@app.post("/plan/create")
def create_plan(request: PlanRequest):
    from intervals_client import IntervalsClient
    from planning import normalize_constraints, validate_plan, evidence_bundle
    try:
        constraints = normalize_constraints(request.constraints)
        plan = validate_plan(request.plan, constraints, {c['id'] for c in evidence_bundle()})
        if plan['status'] != 'draft':
            raise ValueError('请先补充信息，当前没有可写入课程')
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    client = IntervalsClient()
    results = []
    for event in plan['events']:
        if event['kind'] == 'rest':
            continue
        result = client.create_event(
            date=event["date"],
            name=event["name"],
            description=event["description"],
            load_target=event.get("load_target")
        )
        results.append(result)
    return {"created": len(results), "events": results}


# ── 记忆管理接口（调试用）────────────────────────────────────
@app.get("/memory/list")
def list_memories():
    """查看当前所有长期记忆"""
    memories = mem.get_all_memories()
    return {"memories": memories}

@app.delete("/memory/clear")
def clear_memories():
    """清空所有长期记忆（调试用）"""
    import sqlite3
    conn = sqlite3.connect(mem.DB_PATH)
    conn.execute("DELETE FROM memories")
    conn.commit()
    conn.close()
    return {"status": "cleared"}
