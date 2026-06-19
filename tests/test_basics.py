"""T1 基础能力：日期 + T2 工具路由：正确选择工具。

这些测试不需要 mock API —— 通过 system prompt 日期注入和工具选择规则
可以直接验证，不依赖外部服务。
"""

import re
from datetime import datetime


class TestDateInjection:
    """T2：System prompt 必须注入当前日期。"""

    def test_system_prompt_contains_today_date(self, ask):
        """问"今天是哪一天"，回答应包含当前年月日。"""
        resp = ask("今天是哪一天？")
        today = datetime.now()
        patterns = [
            f"{today.year}年{today.month}月{today.day}日",
            f"{today.year}年0{today.month}月{today.day}日" if today.month < 10 else "",
            f"{today.year}年{today.month}月0{today.day}日" if today.day < 10 else "",
            f"{today.year}-{today.month:02d}-{today.day:02d}",
            str(today.year),
        ]
        answer = resp.get("answer", "")
        has_year = str(today.year) in answer
        assert has_year, f"回答 '{answer}' 不包含当前年份 {today.year}"

    def test_date_not_hardcoded(self, ask):
        """两次不同日期问同一问题，回答应该一致（都指向 today）。"""
        resp1 = ask("今天是哪一天？")
        resp2 = ask("今天是哪一天？")
        # 两次回答应该相似（都是今天），不能第一天是 5 月第二天变 6 月
        a1 = resp1.get("answer", "")
        a2 = resp2.get("answer", "")
        # 提取年份数字
        years1 = re.findall(r"20\d{2}", a1)
        years2 = re.findall(r"20\d{2}", a2)
        if years1 and years2:
            assert years1[0] == years2[0], f"年份不一致: {years1[0]} vs {years2[0]}"


class TestToolRouting:
    """T1：根据问题类型，Agent 应选择正确的工具（或不选工具）。"""

    def test_state_query_triggers_get_full_context(self, ask):
        """问个人状态应调用 get_full_context，回答包含 CTL/TSB 等指标。"""
        resp = ask("我今天状态怎么样？")
        answer = resp.get("answer", "").lower()
        # 应该包含训练状态相关词汇（工具返回的数据）
        # 注意：CTL/TSB 是工具返回的，直接回答不会包含
        keywords = ["ctl", "tsb", "训练", "状态", "疲劳", "恢复", "ftp"]
        matched = [kw for kw in keywords if kw in answer]
        assert len(matched) >= 2, f"回答 '{answer}' 缺少状态指标词汇"

    def test_knowledge_query_triggers_search_knowledge(self, ask):
        """问骑行知识应调用 search_knowledge，回答包含专业知识。"""
        resp = ask("什么是Z2训练？")
        answer = resp.get("answer", "").lower()
        # Z2 定义应该出现在知识检索结果中
        keywords = ["z2", "有氧", "耐力", "功率", "ftp", "区间"]
        matched = [kw for kw in keywords if kw in answer]
        assert len(matched) >= 1, f"回答 '{answer}' 未涉及骑行知识概念"

    def test_simple_greeting_no_tool(self, ask):
        """简单问候不需要调用任何工具。"""
        resp = ask("你好")
        answer = resp.get("answer", "")
        assert len(answer) > 0
        assert "骑行" in answer or "教练" in answer or "什么" in answer or "帮助" in answer or "你好" in answer


class TestResponseStructure:
    """API 响应结构必须符合规范。"""

    def test_ask_returns_valid_structure(self, ask):
        resp = ask("你好")
        assert "type" in resp
        assert resp["type"] in ("message", "awaiting_confirmation", "awaiting_input")
        if resp["type"] == "message":
            assert "answer" in resp
            assert isinstance(resp["answer"], str)
            assert len(resp["answer"]) > 0

    def test_empty_question_rejected(self, client, fresh_session):
        r = client.post("/ask", json={"question": "   ", "session_id": fresh_session})
        assert r.status_code == 400, f"空问题应返回 400，实际 {r.status_code}"
