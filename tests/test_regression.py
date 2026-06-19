"""回归测试：之前修过的 bug 不应再复现。

每修一个 bug，在这里加一条测试。测试失败意味着旧问题复现。
"""


class TestDateNotStale:
    """Bug：agent 不知道当前日期，回答 2025 年或 5 月。已修复 2026-06-19。"""

    def test_date_is_current_year(self, ask):
        from datetime import datetime
        resp = ask("今天是哪一天？")
        answer = resp.get("answer", "")
        current_year = str(datetime.now().year)
        assert current_year in answer, \
            f"回归：回答 '{answer}' 不包含当前年份 {current_year}"


class TestEmptyQuestion:
    """Bug：空问题或纯空格不应触发 Agent 调用。"""

    def test_whitespace_rejected(self, client, fresh_session):
        r = client.post("/ask", json={"question": "   ", "session_id": fresh_session})
        assert r.status_code == 400


class TestResponseHasAnswer:
    """Bug：某些情况下 /ask 返回了 type=message 但没有 answer 字段。"""

    def test_message_type_has_answer(self, ask):
        resp = ask("你好")
        if resp.get("type") == "message":
            assert "answer" in resp
            assert len(resp["answer"]) > 0
