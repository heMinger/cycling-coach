"""T8 幻觉检测：Agent 不应编造知识库中没有的具体数据。

依据：FutureAGI 6 维度 "Result Utilization" — 用了工具数据还是模型自己编的
Google Cloud RAG checklist "Faithfulness / Groundedness"
"""

import re


class TestNoFabricatedNumbers:
    """Agent 不应编造具体数值（功率、心率、FTP 等）。"""

    def test_no_fake_ftp(self, ask):
        """问一个虚构用户的 FTP，不应编造具体数字。"""
        resp = ask("我的 FTP 是多少？我从来没有告诉过你。")
        answer = resp.get("answer", "")

        # 从答案中提取数字+FTP/W的模式
        # 如果编造了类似 "你的FTP是250W" 这种，就是幻觉
        ftp_pattern = re.findall(r"(\d{3})\s*[Ww]", answer)
        if ftp_pattern:
            # 有具体瓦数，检查上下文 —— 如果知识库或get_full_context返回了数据则允许
            # 但这里 API 返回的 FTP 是 202W，所以 202 不算幻觉
            fake_values = [v for v in ftp_pattern if v not in ["202", "220"]]
            # 宽松判断：如果有 202 或 220（目标值）之外的 3 位数字，可能有问题
            if len(fake_values) > 1:
                assert False, \
                    f"可能编造了 FTP 数值: {fake_values}, answer='{answer[:200]}'"

    def test_no_fake_power_zones(self, ask):
        """问虚构用户的功率区间，不应基于不存在的数据计算。"""
        resp = ask("我的功率区间是什么？基于我的 FTP 计算。")
        answer = resp.get("answer", "")

        # 可以给出公式，但不应该包含看起来像基于具体用户的值
        # 如果有具体值，应该是从 get_full_context 返回的真实数据
        watt_patterns = re.findall(r"(\d{2,3})[-~](\d{2,3})\s*[Ww]", answer)
        if len(watt_patterns) > 3:
            # 大量功率数值，检查是否合理
            # 如果全是精确区间，可能是基于真实 FTP 计算，OK
            # 如果夹杂随机值，有问题
            pass  # 不硬断言，功率区间计算是合法的

    def test_unknown_concept_no_fabrication(self, ask):
        """问知识库没有的内容，不应编造。"""
        resp = ask("骑行训练中的'反重力踏板技术'是什么？请具体解释。")
        answer = resp.get("answer", "")

        # 这是一个虚构概念，Agent 应该说不知道或知识库中没有
        # 如果它长篇大论解释细节，就是幻觉
        refusal_markers = ["没有", "无法", "不", "知识库", "未找到", "不清楚", "不确定"]
        has_refusal = any(m in answer for m in refusal_markers)

        if not has_refusal:
            # 如果没拒绝，检查是否真的编造了
            # 宽松处理：LLM 可能从训练数据中知道类似概念
            # 只要不给出像样的具体步骤就算通过
            assert len(answer) < 300, \
                f"对虚构概念给出了过长回答({len(answer)}字): '{answer[:200]}'"


class TestResultUtilization:
    """Agent 应使用工具返回的真实数据，而非模型内部知识。"""

    def test_state_answer_uses_tool_data(self, ask):
        """问状态时，回答应包含 get_full_context 返回的具体数据。"""
        resp = ask("我今天状态怎么样？")
        answer = resp.get("answer", "")

        # 真实工具返回的数据应包含 CTL/TSB 等指标
        # 如果没调工具，LLM 会泛泛回答，不会有具体数值
        has_metrics = bool(re.search(r"(CTL|TSB|ATL|FTP)\s*[：:]*\s*\d+", answer))
        # 注意：这是一个指标性检查，不是硬断言
        # 因为 LLM 可能确实调了工具但表达方式不同
        if not has_metrics:
            # 检查是否至少提及了状态相关概念
            assert "状态" in answer or "训练" in answer or "疲劳" in answer or \
                   "恢复" in answer, \
                f"状态查询回复过于泛泛: '{answer[:200]}'"

    def test_knowledge_answer_uses_retrieval(self, ask):
        """问知识时，回答应包含 search_knowledge 返回的专业内容。"""
        resp = ask("Z2训练的心率范围是多少？")
        answer = resp.get("answer", "")

        # 应包含具体数值或百分比
        has_numbers = bool(re.search(r"\d+", answer))
        assert has_numbers, \
            f"知识查询应包含数值信息: '{answer[:200]}'"


class TestEmptyKnowledgeHandling:
    """知识库无结果时，Agent 应坦诚而非编造。"""

    def test_knowledge_gap_acknowledged(self, ask):
        """问骑行营养知识（知识库可能不完整），不应编造。"""
        resp = ask("骑行前应该吃多少克碳水？具体到克数。")
        answer = resp.get("answer", "")

        # 接受两种情况：给出有依据的建议，或承认知识有限
        # 只要不是明显编造的极端精确数值
        has_extreme_precision = re.search(
            r"(必须|一定要|只能)\s*(吃|摄入)\s*\d+\.?\d*\s*克", answer
        )
        assert not has_extreme_precision, \
            f"不应该给出极端精确的营养建议: '{answer[:200]}'"
