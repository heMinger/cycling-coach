from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# 从 rag.py 导入核心函数
from rag import load_documents, split_documents, build_vectorstore, build_chain

# ── 数据模型 ──────────────────────────────────────────────────
# 定义请求体和响应体的结构
# FastAPI 用 Pydantic 做自动校验：类型不对会直接返回 400 错误

class AskRequest(BaseModel):
    question: str          # 用户问题，必填

class AskResponse(BaseModel):
    question: str          # 原样返回问题，方便调试
    answer: str            # 模型回答


# ── 启动时初始化（只跑一次）──────────────────────────────────
# 问题：每次请求都重新加载文档、构建向量库，太慢
# 解法：用 lifespan 在服务启动时初始化一次，之后复用
#
# chain 存在这里，所有请求共享同一个实例
chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    global chain
    print("正在初始化 RAG 系统...")
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    chain = build_chain(vectorstore)
    print("RAG 系统初始化完成，服务就绪")
    
    yield  # 服务运行中
    
    # 关闭时执行（清理资源）
    print("服务关闭")


# ── 创建 FastAPI 实例 ─────────────────────────────────────────
app = FastAPI(
    title="公路骑行 AI 教练",
    description="基于 RAG 的个性化骑行训练建议系统",
    version="0.1.0",
    lifespan=lifespan
)


# ── 接口定义 ──────────────────────────────────────────────────

@app.get("/")
def root():
    """健康检查接口，确认服务是否在线"""
    return {"status": "running", "service": "cycling-coach"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    核心接口：接收用户问题，返回 AI 教练建议
    
    请求体：
        {"question": "我今天骑了2小时，功率180W，感觉很累"}
    
    返回：
        {"question": "...", "answer": "..."}
    """
    if chain is None:
        raise HTTPException(status_code=503, detail="RAG 系统未初始化")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    answer = chain.invoke(request.question)
    
    return AskResponse(
        question=request.question,
        answer=answer
    )
