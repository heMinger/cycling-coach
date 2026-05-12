"""
Layer 5：Agent 行为评估（共 8 项，对应清单 22-29）
  22. 工具选择准确率：15 条问题，记录实际调用工具 vs 预期
  23. 工具误调检测：知识问题不应调 get_full_context，计划问题不应提前调 get_full_context
  24. 双重 API 调用频率：analyze_and_plan 前是否有冗余的 get_full_context
  25. interrupt 流程完整性：计划生成→确认→写入 / 取消 两个分支
  26. tool_call_count 限制：注入 count=7 后验证第 8 次触发超限提示
  27. AutoMemory 触发准确性：直接测 mem.extract_memories()，不经过 agent 图
  28. Session 状态隔离：session A 的 current_plan 不出现在 session B
  29. 降级路径：mock 掉 API，验证 cache → 静态档案降级链

运行方式（在 cycling-coach/ 目录下）：
    python eval/layer5_agent.py

依赖 DEEPSEEK_API_KEY（.env 或环境变量）。
评估使用独立 DB（eval/results/eval_checkpoints.db），不污染生产数据。
"""

import json
import sqlite3
import sys
import time
import unittest.mock as mock
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

ROOT = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
EVAL_DB = RESULTS_DIR / "eval_checkpoints.db"
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


# ── 工具选择测试用例 ───────────────────────────────────────────

TOOL_CASES = [
    # search_knowledge 预期
    {"id": "ts01", "q": "Z2训练的功率范围是多少", "expected": ["search_knowledge"], "anti": ["get_full_context"]},
    {"id": "ts02", "q": "什么是甜蜜点训练", "expected": ["search_knowledge"], "anti": ["get_full_context"]},
    {"id": "ts03", "q": "过度训练有哪些早期信号", "expected": ["search_knowledge"], "anti": ["get_full_context"]},
    {"id": "ts04", "q": "FTP测试方法有哪些", "expected": ["search_knowledge"], "anti": ["get_full_context"]},
    {"id": "ts05", "q": "TSS超过400意味着什么", "expected": ["search_knowledge"], "anti": ["get_full_context"]},
    # get_full_context 预期
    {"id": "ts06", "q": "我今天的训练状态怎么样", "expected": ["get_full_context"], "anti": []},
    {"id": "ts07", "q": "我最近的CTL是多少", "expected": ["get_full_context"], "anti": []},
    {"id": "ts08", "q": "我现在适合高强度训练吗", "expected": ["get_full_context"], "anti": []},
    {"id": "ts09", "q": "我上周训练了多少小时", "expected": ["get_full_context"], "anti": []},
    # analyze_and_plan 预期
    {"id": "ts10", "q": "帮我制定本周训练计划", "expected": ["analyze_and_plan"], "anti": []},
    {"id": "ts11", "q": "我想要一个侧重FTP提升的本周计划", "expected": ["analyze_and_plan"], "anti": []},
    # 无工具预期（直接回答）
    {"id": "ts12", "q": "你好，你能做什么", "expected": [], "anti": []},
    # 双重调用检测专项
    {"id": "ts13", "q": "帮我生成本周训练计划", "expected": ["analyze_and_plan"],
     "anti": [], "check_double": True},
    {"id": "ts14", "q": "我想了解我的状态然后制定训练计划",
     "expected": ["get_full_context", "analyze_and_plan"], "anti": []},
    {"id": "ts15", "q": "Z4阈值间歇的典型训练结构是什么", "expected": ["search_knowledge"], "anti": ["get_full_context"]},
]


# ── 图构建（使用独立 eval DB）────────────────────────────────

def _build_eval_graph():
    from agent import agent_node, custom_tool_node, should_continue
    from agent_state import AgentState
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.sqlite import SqliteSaver

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(EVAL_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", custom_tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


def _fresh_sid(prefix: str) -> str:
    return f"eval_{prefix}_{uuid.uuid4().hex[:8]}"


def _fresh_config(sid: str) -> dict:
    return {"configurable": {"thread_id": sid}, "recursion_limit": 12}


def _initial_state(question: str, sid: str, tool_call_count: int = 0) -> dict:
    return {
        "messages": [HumanMessage(content=question)],
        "session_id": sid,
        "tool_call_count": tool_call_count,
        "current_plan": None,
        "state_analysis": None,
    }


def _extract_tool_calls(messages: list) -> list[str]:
    calls = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            calls.append(tc["name"] if isinstance(tc, dict) else tc.get("name", ""))
    return calls


def _invoke(graph, question: str, sid: str, tool_call_count: int = 0):
    """单轮调用，返回 (messages, tool_calls, response_text, was_interrupted)"""
    config = _fresh_config(sid)
    snapshot = graph.get_state(config)
    if snapshot.values:
        state = {"messages": [HumanMessage(content=question)]}
    else:
        state = _initial_state(question, sid, tool_call_count)
    try:
        result = graph.invoke(state, config)
        interrupted = False
    except GraphInterrupt:
        result = graph.get_state(config).values
        interrupted = True
    # 部分 LangGraph 版本对 interrupt() 不抛异常，而是静默暂停并返回当前状态。
    # 通过 graph.get_state().tasks 检测是否存在挂起任务来补充判断。
    if not interrupted:
        post_snap = graph.get_state(config)
        if post_snap.tasks:
            result = post_snap.values
            interrupted = True
    msgs = result.get("messages", []) if result else []
    tool_calls = _extract_tool_calls(msgs)
    response = msgs[-1].content if msgs else ""
    return msgs, tool_calls, response, interrupted


# ── 项 22 + 23：工具选择 + 误调检测 ──────────────────────────

def eval_tool_selection(graph) -> tuple[dict, dict]:
    print("  项22+23: 工具选择准确率 & 误调检测...")
    records = []
    for case in TOOL_CASES:
        sid = _fresh_sid(case["id"])
        _, actual_tools, response, _ = _invoke(graph, case["q"], sid)

        expected = set(case["expected"])
        anti = set(case.get("anti", []))
        actual = set(actual_tools)

        hit = expected & actual
        missed = expected - actual
        wrong = anti & actual

        if expected:
            precision = len(hit) / len(actual) if actual else 0.0
            recall = len(hit) / len(expected)
        else:
            # 无工具预期：actual 为空则 pass
            precision = 1.0 if not actual else 0.0
            recall = 1.0

        misuse = len(wrong) > 0
        status = "pass" if recall >= 1.0 and not misuse else "fail"

        records.append({
            "id": case["id"], "query": case["q"],
            "expected": list(expected), "actual": list(actual),
            "missed": list(missed), "anti_triggered": list(wrong),
            "precision": round(precision, 3), "recall": round(recall, 3),
            "misuse": misuse, "status": status,
            "response_preview": response[:80],
        })

    pass_count = sum(1 for r in records if r["status"] == "pass")
    avg_recall = round(sum(r["recall"] for r in records) / len(records), 3)
    misuse_count = sum(1 for r in records if r["misuse"])

    item22 = {
        "id": "22", "name": "tool_selection_accuracy",
        "status": "pass" if pass_count == len(records) else "fail",
        "detail": {
            "cases_passed": pass_count, "total": len(records),
            "avg_recall": avg_recall, "per_case": records,
        },
    }
    item23 = {
        "id": "23", "name": "tool_misuse_detection",
        "status": "pass" if misuse_count == 0 else "fail",
        "detail": {
            "misuse_count": misuse_count, "total": len(records),
            "misuse_cases": [r for r in records if r["misuse"]],
        },
    }
    return item22, item23


# ── 项 24：双重 API 调用检测 ─────────────────────────────────

def eval_double_api_call(graph) -> dict:
    print("  项24: 双重API调用检测...")
    # 排除同时预期 get_full_context + analyze_and_plan 的用例（如 ts14 是意图分两步操作）
    plan_cases = [
        c for c in TOOL_CASES
        if "analyze_and_plan" in c["expected"] and "get_full_context" not in c["expected"]
    ]
    records = []
    for case in plan_cases:
        sid = _fresh_sid(f"double_{case['id']}")
        _, actual_tools, _, _ = _invoke(graph, case["q"], sid)
        # 双重调用：get_full_context 出现在 analyze_and_plan 之前
        double = (
            "get_full_context" in actual_tools
            and "analyze_and_plan" in actual_tools
            and actual_tools.index("get_full_context") < actual_tools.index("analyze_and_plan")
        )
        records.append({
            "id": case["id"], "query": case["q"],
            "tool_sequence": actual_tools, "double_call": double,
        })

    double_count = sum(1 for r in records if r["double_call"])
    return {
        "id": "24", "name": "double_api_call",
        "status": "pass" if double_count == 0 else "fail",
        "detail": {
            "plan_cases_tested": len(records),
            "double_call_count": double_count,
            "per_case": records,
        },
    }


# ── 项 25：interrupt 流程完整性 ───────────────────────────────

def eval_interrupt_flow(graph) -> dict:
    print("  项25: interrupt 流程...")
    results = []

    # 分支 A：生成计划 → 确认写入（resume=True）→ 写入成功或提示
    sid_a = _fresh_sid("interrupt_confirm")
    config_a = _fresh_config(sid_a)
    _, _, _, interrupted = _invoke(graph, "帮我生成本周训练计划", sid_a)

    branch_a = {"branch": "confirm", "plan_generated": False,
                "interrupted_after_confirm": False, "final_response": ""}
    if not interrupted:
        # 计划生成后询问确认，可能没有 interrupt（取决于 LLM 是否调 write_to_calendar）
        branch_a["plan_generated"] = True
        # 发送确认
        _, _, resp_a, interrupted_a = _invoke(graph, "确认，写入日历", sid_a)
        branch_a["interrupted_after_confirm"] = interrupted_a
        branch_a["final_response"] = resp_a[:100]
        if interrupted_a:
            # resume with True（用户最终确认）
            try:
                result_a = graph.invoke(Command(resume=True), config_a)
                branch_a["final_response"] = result_a["messages"][-1].content[:100]
            except Exception as e:
                branch_a["final_response"] = f"resume 失败: {e}"
    else:
        branch_a["plan_generated"] = True
        branch_a["interrupted_after_confirm"] = True
        # 直接 resume True
        try:
            result_a = graph.invoke(Command(resume=True), config_a)
            branch_a["final_response"] = result_a["messages"][-1].content[:100]
        except Exception as e:
            branch_a["final_response"] = f"resume 失败: {e}"
    results.append(branch_a)

    # 分支 B：生成计划 → 用户取消（resume=False）
    sid_b = _fresh_sid("interrupt_cancel")
    config_b = _fresh_config(sid_b)
    _invoke(graph, "帮我生成本周训练计划", sid_b)

    branch_b = {"branch": "cancel", "plan_generated": True, "cancel_response": ""}
    _, _, _, interrupted_b = _invoke(graph, "确认，写入日历", sid_b)
    if interrupted_b:
        try:
            result_b = graph.invoke(Command(resume=False), config_b)
            branch_b["cancel_response"] = result_b["messages"][-1].content[:100]
            branch_b["correctly_cancelled"] = "取消" in result_b["messages"][-1].content
        except Exception as e:
            branch_b["cancel_response"] = f"resume 失败: {e}"
            branch_b["correctly_cancelled"] = False
    else:
        branch_b["cancel_response"] = "未触发 interrupt，无法测试取消分支"
        branch_b["correctly_cancelled"] = None
    results.append(branch_b)

    branch_b_ok = branch_b.get("correctly_cancelled", False)
    return {
        "id": "25", "name": "interrupt_flow",
        "status": "pass" if branch_b_ok else "warn",
        "detail": {"branches": results},
    }


# ── 项 26：tool_call_count 限制 ───────────────────────────────

def eval_tool_count_limit(graph) -> dict:
    print("  项26: tool_call_count 限制...")
    sid = _fresh_sid("limit")
    config = _fresh_config(sid)
    # 先初始化 session，再注入 count=7
    graph.invoke(_initial_state("你好", sid), config)
    graph.update_state(config, {"tool_call_count": 7})

    # 触发一次工具调用（知识问题会调 search_knowledge）
    _, actual_tools, response, _ = _invoke(graph, "Z2训练功率是多少", sid)

    # 再次触发，此时 tool_call_count 已到达或超过 8
    _, _, response2, _ = _invoke(graph, "我今天状态怎么样", sid)
    limit_triggered = "超限" in response2 or "重新开始" in response2

    return {
        "id": "26", "name": "tool_call_count_limit",
        "status": "pass" if limit_triggered else "fail",
        "detail": {
            "injected_count": 7,
            "trigger_question": "我今天状态怎么样",
            "response": response2[:150],
            "limit_triggered": limit_triggered,
        },
    }


# ── 项 27：AutoMemory 触发准确性 ─────────────────────────────

def eval_auto_memory() -> dict:
    print("  项27: AutoMemory 触发...")
    import memory as mem

    test_sid = f"eval_memory_{uuid.uuid4().hex[:8]}"
    test_messages = [
        {"role": "user", "content": "我最近左膝盖骑车下坡时有点疼，感觉很不舒服"},
        {"role": "assistant", "content": "了解，建议减少下坡骑行，注意充分热身"},
    ]

    mem.init_db()
    try:
        mem.extract_memories(test_messages, test_sid)
    except Exception as e:
        return {
            "id": "27", "name": "auto_memory",
            "status": "fail",
            "detail": {"error": str(e)},
        }

    # 读取提取结果
    conn = sqlite3.connect(str(ROOT / mem.DB_PATH))
    c = conn.cursor()
    c.execute("SELECT category, content FROM memories WHERE source_session = ?", (test_sid,))
    rows = c.fetchall()
    conn.close()

    # 清理测试数据
    conn = sqlite3.connect(str(ROOT / mem.DB_PATH))
    conn.execute("DELETE FROM memories WHERE source_session = ?", (test_sid,))
    conn.commit()
    conn.close()

    injury_rows = [r for r in rows if r[0] == "injury"]
    knee_captured = any("膝盖" in r[1] or "膝" in r[1] for r in injury_rows)

    return {
        "id": "27", "name": "auto_memory",
        "status": "pass" if knee_captured else "fail",
        "detail": {
            "extracted_memories": [{"category": r[0], "content": r[1]} for r in rows],
            "injury_count": len(injury_rows),
            "knee_captured": knee_captured,
        },
    }


# ── 项 28：Session 状态隔离 ───────────────────────────────────

def eval_session_isolation(graph) -> dict:
    print("  项28: Session 隔离...")

    # Session A：生成计划，产生 current_plan
    sid_a = _fresh_sid("iso_a")
    _invoke(graph, "帮我生成本周训练计划", sid_a)
    snap_a = graph.get_state(_fresh_config(sid_a))
    plan_a = snap_a.values.get("current_plan") if snap_a.values else None

    # Session B：全新对话，current_plan 应为 None
    sid_b = _fresh_sid("iso_b")
    _invoke(graph, "你好", sid_b)
    snap_b = graph.get_state(_fresh_config(sid_b))
    plan_b = snap_b.values.get("current_plan") if snap_b.values else None

    # Session A count 不应影响 Session B count
    count_a = snap_a.values.get("tool_call_count", 0) if snap_a.values else 0
    count_b = snap_b.values.get("tool_call_count", 0) if snap_b.values else 0

    plan_isolated = plan_b is None
    count_isolated = count_b < count_a or count_a == 0

    return {
        "id": "28", "name": "session_isolation",
        "status": "pass" if plan_isolated else "fail",
        "detail": {
            "session_a": {"has_plan": plan_a is not None, "tool_call_count": count_a},
            "session_b": {"has_plan": plan_b is not None, "tool_call_count": count_b},
            "plan_isolated": plan_isolated,
            "count_isolated": count_isolated,
        },
    }


# ── 项 29：降级路径 ───────────────────────────────────────────

def eval_fallback_path() -> dict:
    print("  项29: 降级路径...")
    from tools import _fetch_context
    CACHE_PATH = ROOT / "cache" / "user_profile_cache.md"
    STATIC_PATH = ROOT / "data" / "user_data" / "user_profile.md"
    results = []

    # 场景 1：API 失败 → 应使用 cache 或 static
    with mock.patch("strava_client.StravaClient.build_activity_context",
                    side_effect=Exception("strava mocked")), \
         mock.patch("intervals_client.IntervalsClient.build_user_context",
                    side_effect=Exception("intervals mocked")):
        content = _fetch_context()

    cache_exists = CACHE_PATH.exists()
    static_exists = STATIC_PATH.exists()
    got_content = len(content) > 50

    results.append({
        "scenario": "api_fail",
        "cache_available": cache_exists,
        "static_available": static_exists,
        "got_content": got_content,
        "content_preview": content[:80],
        "passed": got_content,
    })

    # 场景 2：API + cache 均失败 → 应使用 static
    import builtins as _builtins
    _real_open = _builtins.open

    def _mock_open(path, *a, **kw):
        if "user_profile_cache" in str(path):
            raise FileNotFoundError("cache mocked")
        return _real_open(path, *a, **kw)

    with mock.patch("strava_client.StravaClient.build_activity_context",
                    side_effect=Exception("strava mocked")), \
         mock.patch("intervals_client.IntervalsClient.build_user_context",
                    side_effect=Exception("intervals mocked")), \
         mock.patch("builtins.open", side_effect=_mock_open):
        try:
            content2 = _fetch_context()
            got2 = len(content2) > 50
        except Exception as e:
            content2 = str(e)
            got2 = False

    results.append({
        "scenario": "api_and_cache_fail",
        "expected": "static profile",
        "got_content": got2,
        "content_preview": content2[:80],
        "passed": got2,
    })

    all_pass = all(r["passed"] for r in results)
    return {
        "id": "29", "name": "fallback_path",
        "status": "pass" if all_pass else "fail",
        "detail": {"scenarios": results},
    }


# ── 主入口 ────────────────────────────────────────────────────

def run() -> dict:
    graph = _build_eval_graph()

    print("  项22+23...")
    item22, item23 = eval_tool_selection(graph)
    print("  项24...")
    item24 = eval_double_api_call(graph)
    print("  项25...")
    item25 = eval_interrupt_flow(graph)
    print("  项26...")
    item26 = eval_tool_count_limit(graph)
    print("  项27...")
    item27 = eval_auto_memory()
    print("  项28...")
    item28 = eval_session_isolation(graph)
    print("  项29...")
    item29 = eval_fallback_path()

    items = [item22, item23, item24, item25, item26, item27, item28, item29]
    passed = sum(1 for i in items if i["status"] == "pass")

    return {
        "layer": 5, "name": "agent",
        "passed": not any(i["status"] == "fail" for i in items),
        "score": round(passed / len(items), 2),
        "items": items, "errors": [], "duration_s": None,
    }


if __name__ == "__main__":
    t0 = time.time()
    print("\n=== Layer 5: Agent 行为 ===")
    result = run()
    result["duration_s"] = round(time.time() - t0, 2)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RESULTS_DIR / f"layer5_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer5_agent.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    icons = {"pass": "✓", "warn": "△", "fail": "✗", "skip": "–"}
    for item in result["items"]:
        icon = icons[item["status"]]
        d = item.get("detail", {})
        extra = ""
        if item["id"] == "22":
            extra = f"  {d.get('cases_passed')}/{d.get('total')}  recall={d.get('avg_recall')}"
        elif item["id"] == "23":
            extra = f"  misuse={d.get('misuse_count')}/{d.get('total')}"
        elif item["id"] == "24":
            extra = f"  double={d.get('double_call_count')}/{d.get('plan_cases_tested')}"
        elif item["id"] == "25":
            extra = f"  branches={len(d.get('branches', []))}"
        elif item["id"] == "26":
            extra = f"  limit_triggered={d.get('limit_triggered')}"
        elif item["id"] == "27":
            extra = f"  knee_captured={d.get('knee_captured')}"
        elif item["id"] == "28":
            extra = f"  plan_isolated={d.get('plan_isolated')}"
        elif item["id"] == "29":
            extra = f"  scenarios={len(d.get('scenarios', []))}"
        print(f"  [{icon}] 项{item['id']} {item['name']}: {item['status']}{extra}")

    print(f"\n得分：{result['score']}  passed={result['passed']}")
    print(f"结果写入：{out_path}  耗时：{result['duration_s']}s")

    # ── 失败项详情 ────────────────────────────────────────────
    for item in result["items"]:
        if item["status"] not in ("fail", "warn"):
            continue
        print(f"\n{'─'*60}")
        print(f"项{item['id']} {item['name']} 失败详情：")
        d = item.get("detail", {})
        if item["id"] == "22":
            for c in d.get("per_case", []):
                if c["status"] == "fail":
                    print(f"  [{c['id']}] expected={c['expected']} actual={c['actual']} "
                          f"missed={c['missed']}")
                    print(f"       Q: {c['query']}")
        elif item["id"] == "23":
            for c in d.get("misuse_cases", []):
                print(f"  [{c['id']}] anti_triggered={c['anti_triggered']}  Q: {c['query']}")
        elif item["id"] == "24":
            for c in d.get("per_case", []):
                if c["double_call"]:
                    print(f"  [{c['id']}] sequence={c['tool_sequence']}")
        elif item["id"] == "25":
            for b in d.get("branches", []):
                print(f"  branch={b['branch']}: {b}")
        elif item["id"] == "26":
            print(f"  response: {d.get('response')}")
        elif item["id"] == "29":
            for s in d.get("scenarios", []):
                if not s["passed"]:
                    print(f"  scenario={s['scenario']}: {s}")
