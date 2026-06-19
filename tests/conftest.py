"""pytest 共享 fixtures：FastAPI TestClient、模拟 LLM 响应。

测试策略：
- API 结构类测试 → 真实 TestClient，不依赖 LLM
- Agent 行为类测试 → mock agent_node，避免调用真实 DeepSeek API
"""
import pytest
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载真实 .env（让 FastAPI lifespan 能初始化 graph）
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from fastapi.testclient import TestClient
from api import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient，带 lifespan 初始化（加载真实 graph）。"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def fresh_session():
    """生成唯一 session_id，隔离测试。"""
    import uuid
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def ask(client, fresh_session):
    """快捷方法：向 /ask 发请求并返回 response dict。"""
    def _ask(question: str, session_id: str = None):
        sid = session_id or fresh_session
        r = client.post("/ask", json={"question": question, "session_id": sid})
        assert r.status_code == 200, f"/ask 返回 {r.status_code}: {r.text}"
        return r.json()
    return _ask
