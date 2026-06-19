"""API 端点测试：状态码、响应格式、错误处理。"""


class TestHealthEndpoint:
    def test_root_returns_status(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"
        assert data["service"] == "cycling-coach"
        assert "version" in data

    def test_ui_returns_html(self, client):
        r = client.get("/ui")
        assert r.status_code == 200
        assert "骑行" in r.text or "<html" in r.text.lower()


class TestAskEndpoint:
    def test_valid_question_returns_200(self, client, fresh_session):
        r = client.post("/ask", json={
            "question": "你好",
            "session_id": fresh_session,
        })
        assert r.status_code == 200
        data = r.json()
        assert "type" in data

    def test_missing_question_rejected(self, client, fresh_session):
        r = client.post("/ask", json={"session_id": fresh_session})
        assert r.status_code == 422  # FastAPI validation error

    def test_same_session_preserves_context(self, client, fresh_session):
        """同一 session 第二问应知道上下文。"""
        r1 = client.post("/ask", json={
            "question": "我今天状态怎么样？",
            "session_id": fresh_session,
        })
        assert r1.status_code == 200

        r2 = client.post("/ask", json={
            "question": "能帮我制定训练计划吗？",
            "session_id": fresh_session,
        })
        assert r2.status_code == 200
        data2 = r2.json()
        # 可能在 awaiting_* 状态（agent 需要确认或更多信息）
        assert data2["type"] in ("message", "awaiting_confirmation", "awaiting_input")


class TestMemoryEndpoints:
    def test_list_memories(self, client):
        r = client.get("/memory/list")
        assert r.status_code == 200
        assert "memories" in r.json()

    def test_clear_memories(self, client):
        r = client.delete("/memory/clear")
        assert r.status_code == 200
        assert r.json()["status"] == "cleared"


class TestPlanEndpoint:
    def test_get_plan_no_session(self, client):
        r = client.get("/plan/current", params={"session_id": "nonexistent"})
        assert r.status_code == 200
        assert r.json() == {"plan": None}
