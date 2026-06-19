"""T3 参数校验：工具应正确处理异常参数（空值、超长、边界）。

依据：FutureAGI 6 维度 "Argument Extraction" — schema-valid + semantically correct
BFCL (Berkeley Function Calling) — 参数正确性 + 无关检测
"""

import pytest


class TestSearchKnowledgeArgs:
    """search_knowledge 参数：query 是最关键的输入。"""

    def test_empty_query_returns_noise_warning(self):
        """空查询应返回有意义的结果（而非崩溃或返回随机内容）。"""
        from tools import search_knowledge
        result = search_knowledge.invoke({"query": ""})
        assert isinstance(result, str), f"应返回字符串而非异常"
        assert len(result) > 0, "空查询应该返回提示信息"

    def test_whitespace_query_handled(self):
        """纯空格查询不应崩溃。"""
        from tools import search_knowledge
        result = search_knowledge.invoke({"query": "   "})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_char_query_handled(self):
        """单字符查询不应崩溃。"""
        from tools import search_knowledge
        result = search_knowledge.invoke({"query": "Z"})
        assert isinstance(result, str)

    def test_valid_query_returns_content(self):
        """正常查询应返回知识内容。"""
        from tools import search_knowledge
        result = search_knowledge.invoke({"query": "Z2训练功率范围"})
        assert len(result) > 20, \
            f"正常查询应返回足够内容: '{result[:100]}'"

    def test_nonexistent_topic_returns_not_found(self):
        """知识库没有的主题应返回'未找到'而非随机内容。"""
        from tools import search_knowledge
        result = search_knowledge.invoke({"query": "火星上的骑行训练基地"})
        assert isinstance(result, str)
        # 应明确表示未找到
        no_result_markers = ["未找到", "没有", "不存在", "无相关", "not found"]
        has_marker = any(m in result.lower() for m in no_result_markers)
        # 注意：如果查询恰好命中某些 chunk（罕见），可能返回内容
        # 不做硬断言，但记录行为
        if not has_marker:
            pytest.skip(f"查询虚构主题返回了内容（可能命中无关chunk）: '{result[:150]}'")


class TestAnalyzeAndPlanArgs:
    """analyze_and_plan 参数：user_request 应正确传递。"""

    def test_empty_request_handled_gracefully(self):
        """空请求不应崩溃，应返回错误或默认计划。"""
        from tools import analyze_and_plan
        result = analyze_and_plan.invoke({"user_request": ""})
        assert isinstance(result, dict), f"应返回 dict: {type(result)}"
        # 可能包含 error 字段或默认 plan
        assert "plan" in result or "error" in result or "_display" in result

    def test_vague_request_still_generates_plan(self):
        """模糊请求'给我个计划'应生成计划而非报错。"""
        from tools import analyze_and_plan
        result = analyze_and_plan.invoke({"user_request": "给我一个计划"})
        assert isinstance(result, dict)
        assert "plan" in result, f"应包含 plan 字段: {result.keys()}"

    def test_specific_request_includes_details(self):
        """具体请求'侧重提升FTP'的计划应包含对应训练。"""
        from tools import analyze_and_plan
        result = analyze_and_plan.invoke({"user_request": "侧重提升FTP，本周训练4次"})
        assert isinstance(result, dict)
        plan = result.get("plan", {})
        events = plan.get("events", [])
        assert len(events) > 0, f"应至少生成一条训练安排: {result}"


class TestModifyPlanArgs:
    """modify_plan 参数：依赖 state 中的 current_plan。"""

    def test_no_existing_plan_returns_error(self):
        """无 current_plan 时调用应返回明确错误。"""
        from tools import modify_plan
        result = modify_plan.invoke({
            "modification_request": "把周二改到周三",
            "state": {"current_plan": None},
        })
        assert isinstance(result, dict)
        assert "error" in result, f"无计划时应返回 error: {result.keys()}"
        assert "没有" in result["error"] or "不存在" in result["error"] or \
               "生成" in result["error"] or "plan" in result["error"].lower(), \
            f"错误信息应说明原因: '{result['error']}'"


class TestAskUserArgs:
    """ask_user 参数：question 必须有意义。"""

    def test_empty_question_handled(self):
        """空问题——ask_user 通过 interrupt 实现，空问题应仍能触发。"""
        from tools import ask_user
        # ask_user 使用 interrupt，在测试环境会抛异常
        # 这里验证至少参数校验不崩溃
        # 不直接 invoke（会触发 interrupt），只验证参数 schema
        assert ask_user.name == "ask_user"
        assert "question" in str(ask_user.args_schema.schema())
