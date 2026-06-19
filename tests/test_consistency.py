"""T6 多轮对话一致性：同一 session 内 Agent 应保持上下文连贯。

依据：Google Cloud RAG checklist — "Conversational / Multi-Turn Behavior"
FutureAGI 6 维度 — "Plan Coherence"（无矛盾、记住前文）
"""


class TestContextPersistence:
    """同一 session 内，Agent 应记住之前的对话。"""

    def test_followup_question_remembers_topic(self, client, fresh_session):
        """先问训练状态，追问时不用重申完整问题。"""
        sid = fresh_session

        r1 = client.post("/ask", json={
            "question": "什么是Z2训练？",
            "session_id": sid,
        })
        assert r1.status_code == 200
        a1 = r1.json().get("answer", "").lower()

        # 追问：应该记住上一轮在说 Z2
        r2 = client.post("/ask", json={
            "question": "我应该在这个区间训练多久？",
            "session_id": sid,
        })
        assert r2.status_code == 200
        a2 = r2.json().get("answer", "").lower()

        # 第二轮回答应包含 Z2 或功率或时长相关内容
        related = ["z2", "功率", "有氧", "耐力", "小时", "分钟", "区间", "强度"]
        matched = [kw for kw in related if kw in a2]
        assert len(matched) >= 1, \
            f"追问 Z2 时长，回答应关联 Z2 相关概念: '{a2[:200]}'"

    def test_personal_context_persists(self, client, fresh_session):
        """先问状态，再问'那我应该怎么练'，应基于前面返回的状态。"""
        sid = fresh_session

        r1 = client.post("/ask", json={
            "question": "我今天状态怎么样？",
            "session_id": sid,
        })
        assert r1.status_code == 200
        a1 = r1.json().get("answer", "")

        r2 = client.post("/ask", json={
            "question": "那我适合练什么？",
            "session_id": sid,
        })
        assert r2.status_code == 200
        a2 = r2.json().get("answer", "")

        # 第二轮应该基于上下文给出建议
        training_terms = ["训练", "恢复", "强度", "区间", "休息", "有氧", "阈值", "间歇"]
        matched = [kw for kw in training_terms if kw in a2]
        assert len(matched) >= 1, \
            f"基于状态的追问应给出训练建议: '{a2[:200]}'"

    def test_plan_modification_remembers_plan(self, client, fresh_session):
        """生成计划后修改，不应要求重新描述。"""
        sid = fresh_session

        r1 = client.post("/ask", json={
            "question": "帮我制定本周训练计划",
            "session_id": sid,
        })
        assert r1.status_code == 200
        type1 = r1.json().get("type")

        # 如果计划已生成（可能在等确认），修改请求应被理解
        if type1 == "awaiting_confirmation":
            # 先确认写入
            client.post("/resume", json={
                "session_id": sid,
                "value": False,  # 不写入但保留计划
            })

        r2 = client.post("/ask", json={
            "question": "把周二的训练改到周三",
            "session_id": sid,
        })
        assert r2.status_code == 200
        a2 = r2.json().get("answer", "")
        # 至少不应说"没有计划"
        assert "没有" not in a2 or "计划" not in a2, \
            f"修改请求不应该被完全忽略: '{a2[:200]}'"
