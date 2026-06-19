"""T4 错误恢复：工具挂了应重试/降级/提示，而非崩溃或静默返回过期数据。

测试策略：
- 单元测试：直接测 _fetch_context 的降级链（实时→缓存→静态档案）
- Agent 集成测试：验证 produce 端不会因 API 异常而 500
- 降级链完整性：验证缓存文件和静态档案存在且内容有效

依据：FutureAGI 6 维度 "Error Recovery"
Google Cloud RAG checklist "Graceful degradation"
"""

import os
import re
from unittest import mock


class TestFetchContextDegradation:
    """直接测试 _fetch_context 的三层降级链。"""

    @mock.patch("strava_client.StravaClient.get_activities")
    @mock.patch("intervals_client.IntervalsClient.get_wellness")
    @mock.patch("intervals_client.IntervalsClient.get_athlete_info")
    def test_real_time_failure_falls_back_to_cache(
        self, mock_athlete, mock_wellness, mock_activities
    ):
        """模拟 Strava + Intervals 全部失败，验证降级到缓存。"""
        from tools import _fetch_context

        mock_activities.side_effect = Exception("Strava down")
        mock_wellness.side_effect = Exception("Intervals down")
        mock_athlete.side_effect = Exception("Intervals down")

        result = _fetch_context()
        assert len(result) > 100, "降级后应返回缓存内容"
        # 缓存数据可能包含旧的训练记录
        assert "训练" in result or "FTP" in result or "CTL" in result or \
               "ctl" in result.lower() or "骑行" in result, \
            f"降级内容不应为空: '{result[:200]}'"

    @mock.patch("strava_client.StravaClient.get_activities")
    @mock.patch("intervals_client.IntervalsClient.get_wellness")
    @mock.patch("intervals_client.IntervalsClient.get_athlete_info")
    @mock.patch("os.path.exists")
    def test_cache_missing_falls_back_to_static(
        self, mock_exists, mock_athlete, mock_wellness, mock_activities
    ):
        """模拟 API 失败 + 缓存不存在，验证降级到静态档案。"""
        from tools import _fetch_context

        mock_activities.side_effect = Exception("Strava down")
        mock_wellness.side_effect = Exception("Intervals down")
        mock_athlete.side_effect = Exception("Intervals down")

        # 缓存路径返回 False，强制跳过缓存直接到静态档案
        def path_exists(path):
            if "cache" in str(path):
                return False
            return True
        mock_exists.side_effect = path_exists

        result = _fetch_context()
        assert len(result) > 50, "应降级到静态档案"
        # 静态档案包含基本信息
        assert "FTP" in result or "体重" in result or "训练" in result, \
            f"静态档案应有用户基本信息: '{result[:200]}'"

    def test_degradation_chain_files_exist(self):
        """三层降级的文件均应存在。"""
        assert os.path.exists("cache/user_profile_cache.md"), \
            "缓存文件缺失"
        assert os.path.exists("data/user_data/user_profile.md"), \
            "静态档案缺失"


class TestAgentSurvivesToolFailure:
    """Agent 层面的错误恢复：某个工具失败不应导致整个请求 500。"""

    def test_agent_responds_normally_after_knowledge_empty(self, ask):
        """知识库返回空结果时，Agent 应正常回复而非 500。"""
        # 用极偏的查询触发 search_knowledge 返回空
        resp = ask("量子力学与骑行训练的交叉研究有哪些？")
        assert resp.get("type") == "message"
        assert len(resp.get("answer", "")) > 0

    def test_api_failure_triggers_degradation_in_real_call(self, ask):
        """真实调用应返回有效响应（不mock，验证实际降级可用）。"""
        resp = ask("我今天状态怎么样？")
        assert resp.get("type") == "message"
        answer = resp.get("answer", "")
        assert len(answer) > 0
        # 回应中应包含状态相关内容
        has_state_words = any(
            w in answer for w in
            ["CTL", "TSB", "ATL", "训练", "状态", "疲劳", "恢复", "FTP"]
        )
        assert has_state_words, \
            f"状态查询应包含训练相关词汇: '{answer[:200]}'"


class TestDataFreshnessIndicators:
    """验证数据来源透明性。"""

    def test_cache_file_has_timestamp(self):
        """缓存文件应可读且内容不应过于陈旧（超过 90 天）."""
        import time
        cache_path = "cache/user_profile_cache.md"
        mtime = os.path.getmtime(cache_path)
        age_days = (time.time() - mtime) / 86400
        # 缓存文件修改时间不应超过 90 天
        assert age_days < 90, \
            f"缓存文件 {age_days:.0f} 天未更新，可能降级链已断裂"

    def test_context_contains_recent_date(self, ask):
        """状态查询的回复应包含近期日期（而非几个月前的）。"""
        from datetime import datetime
        resp = ask("我最近一次训练是什么时候？")
        answer = resp.get("answer", "")

        # 提取所有 20xx 年日期
        dates = re.findall(r"20(\d{2})[年/-](\d{1,2})", answer)
        if dates:
            current_month = datetime.now().month
            for yy, mm in dates:
                month = int(mm)
                # 相差不应超过 2 个月
                diff = abs(current_month - month)
                if diff > 2 and diff < 10:
                    # 可能数据太旧，做软断言
                    # 这可能是降级情况，需要提示
                    assert False, \
                        f"回复包含过时日期 20{yy}年{mm}月，当前{current_month}月: '{answer[:200]}'"
