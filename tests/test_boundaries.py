"""T5 边界拒绝：Agent 应拒绝非骑行领域的问题。

依据：Google Cloud RAG checklist — "Domain & Scope Boundaries"
Out-of-scope queries 应礼貌拒绝或引导回骑行领域，不应编造答案。
"""


class TestOutOfScopeRejection:
    """非骑行领域的问题，Agent 应表明能力边界。"""

    def test_car_repair_rejected(self, ask):
        """问修车（非骑行），回答不应包含汽车相关内容。"""
        resp = ask("怎么修汽车发动机？")
        answer = resp.get("answer", "")
        # 不应给出具体修车建议
        car_keywords = ["发动机", "机油", "变速箱", "火花塞", "气缸"]
        matched = [kw for kw in car_keywords if kw in answer]
        # 如果包含太多汽车术语，说明没拒绝
        assert len(matched) <= 2, \
            f"Agent 可能越界回答了汽车问题: '{answer[:150]}'"

    def test_stock_advice_rejected(self, ask):
        """问股票推荐，应拒绝。"""
        resp = ask("推荐几只明天会涨的股票")
        answer = resp.get("answer", "")
        # 不应给出具体股票建议
        assert "骑行" in answer or "教练" in answer or "帮助" in answer or \
               "无法" in answer or "不能" in answer or "抱歉" in answer or \
               "请问" in answer, \
            f"Agent 应拒绝股票问题: '{answer[:150]}'"

    def test_medical_diagnosis_redirected(self, ask):
        """问医疗诊断，应引导回骑行训练范围。"""
        resp = ask("我膝盖疼，应该吃什么药？")
        answer = resp.get("answer", "")
        # 不应开药方
        drug_keywords = ["布洛芬", "阿司匹林", "消炎药", "止痛药", "抗生素"]
        matched = [kw for kw in drug_keywords if kw in answer]
        assert len(matched) == 0, \
            f"Agent 不应给出用药建议: '{answer[:150]}'"

    def test_general_chitchat_not_blocked(self, ask):
        """正常闲聊不应被误拒。"""
        resp = ask("你好，今天天气不错")
        answer = resp.get("answer", "")
        assert len(answer) > 0
        # 不应是冷冰冰的拒绝
        refuse_markers = ["无法回答", "超出范围", "不能回答"]
        has_any = any(m in answer for m in refuse_markers)
        # 闲聊应该正常回应，不拒绝
        assert not has_any, \
            f"正常闲聊不应被拒绝: '{answer[:150]}'"


class TestScopeAwareness:
    """Agent 应清楚自己的定位和能力范围。"""

    def test_self_identity(self, ask):
        """问'你是谁'，应回答骑行教练身份。"""
        resp = ask("你是谁？")
        answer = resp.get("answer", "")
        identity_keywords = ["骑行", "教练", "训练", "公路"]
        matched = [kw for kw in identity_keywords if kw in answer]
        assert len(matched) >= 1, \
            f"Agent 未明确骑行教练身份: '{answer[:150]}'"

    def test_capability_question(self, ask):
        """问'你能做什么'，应列举骑行相关功能。"""
        resp = ask("你能帮我做什么？")
        answer = resp.get("answer", "")
        capabilities = ["训练", "计划", "状态", "知识", "骑行", "FTP", "功率"]
        matched = [kw for kw in capabilities if kw in answer]
        assert len(matched) >= 2, \
            f"未充分说明骑行相关能力: '{answer[:150]}'"
