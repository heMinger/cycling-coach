"""
Layer 4：生成质量评估（共 6 项，对应清单 16-21）
  16. Faithfulness：回答中的事实是否来自召回 chunk，有无凭空捏造
  17. Answer Relevancy：回答是否切题，有没有答非所问
  18. Domain Correctness：骑行领域计算/判断结果是否正确（确定性检查）
  19. 幻觉专项：问不在档案里的个人数据，是否会编造具体数字
  20. 计划合理性：TSS 总量范围、单次上限、有无休息日
  21. 消融实验：有无知识库检索，回答质量差异

运行方式（在 cycling-coach/ 目录下）：
    python eval/layer4_generation.py

依赖 DEEPSEEK_API_KEY；未设置时所有 LLM 相关项标记为 skip。
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
sys.path.insert(0, str(ROOT))

# ── 测试用例定义 ──────────────────────────────────────────────

# 项 16 + 17 共用：知识问答题
GEN_QUESTIONS = [
    {"id": "q06", "query": "FTP有哪些测试方法？"},
    {"id": "q07", "query": "TSS超过300大概需要恢复几天？"},
    {"id": "q11", "query": "过度训练的早期信号有哪些？"},
]

# 项 18：领域正确性（确定性计算）
DOMAIN_CASES = [
    {
        "id": "dc01",
        "question": "FTP是202W，Z2训练的功率上下限各是多少瓦？只给数字。",
        "check": "range",
        "lo": 110, "hi": 155,    # 202×0.56≈113，202×0.75≈152；允许±3W
        # 注意：60-70%（121-141W）是错误定义，56-75%（113-152W）才是 Z2
        # 检查时同时验证数字是否落在正确区间内（113±3 和 152±3）
        "correct_lo": 113, "correct_hi": 152,
        "note": "56-75% FTP = 113-152W；如果回答 60-70%=121-141W 则为领域错误",
    },
    {
        "id": "dc02",
        "question": "NP是190W，FTP是202W，IF是多少？只给小数。",
        "check": "value",
        "expected": 0.9406,      # 190/202
        "tolerance": 0.02,
        "note": "IF = NP/FTP = 0.94",
    },
    {
        "id": "dc03",
        "question": "TSB是-25，处于哪个状态区间？用一个词或短语回答。",
        "check": "keywords",
        "expected_keywords": ["疲劳", "过渡", "高负荷", "高风险", "积累"],
        "note": "TSB -25 表示高疲劳/过渡区间",
    },
]

# 项 19：幻觉专项——注入残缺档案，问不存在的数据
HALLUCINATION_PROFILE = """用户档案（摘要）：
- FTP：202W，最大心率：194bpm，体重：63kg
- 近期活动：系统离线，暂无逐次训练记录
"""

HALLUCINATION_CASES = [
    {
        "id": "h01",
        "question": "我上周三骑行的NP具体是多少瓦？",
        "danger_patterns": [
            # 只匹配"NP是/为 数字"，避免误触 FTP/功率区间等合法数字
            r"NP\s*[是为：:]\s*\d+",       # "NP是175"
            r"上周[一二三四五六日天]\S*NP\S*\d+",  # "上周三NP约175W"
            r"标准化功率\s*[是为：:]\s*\d+",
        ],
        "note": "档案里没有逐次记录，不应编造具体 NP 瓦数（FTP=202W 是合法引用，不算幻觉）",
    },
    {
        "id": "h02",
        "question": "我3月10日骑行的平均心率是多少？",
        "danger_patterns": [
            r"\b1[3-9]\d\s*bpm\b",     # 130-199bpm
            r"心率\s*[是为：:]\s*1[3-9]\d",
        ],
        "note": "档案里没有该日期数据，不应编造心率数字",
    },
]

# 项 21：消融实验问题
ABLATION_QUESTIONS = [
    {"id": "q03", "query": "Z5训练每周最多做几次，每次多长时间？"},
    {"id": "q11", "query": "过度训练的早期信号有哪些？"},
]


# ── 工具函数 ──────────────────────────────────────────────────

def _has_api_key() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def _get_llm(temperature: float = 0.3):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=temperature,
    )


def _get_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(persist_directory=str(ROOT / "chroma_db"), embedding_function=emb)


def _retrieve(vs, query: str, threshold: float = 0.85, k: int = 5) -> str:
    results = vs.similarity_search_with_score(query, k=k)
    docs = [doc for doc, score in results if score < threshold]
    if not docs:
        return "（知识库未找到相关内容）"
    return "\n\n---\n\n".join(d.page_content for d in docs)


def _generate(llm, question: str, context: str) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    sys_msg = SystemMessage(content=(
        "你是一个专业的公路骑行教练，风格简练直接。"
        "直接给结论，引用具体数据，150字以内，不使用加粗符号。"
    ))
    human = HumanMessage(content=f"参考知识：\n{context}\n\n问题：{question}")
    return llm.invoke([sys_msg, human]).content.strip()


def _judge(llm, question: str, context: str, answer: str, criterion: str) -> dict:
    """LLM-as-judge，返回 {"score": float, "reason": str}"""
    from langchain_core.messages import HumanMessage
    prompt = (
        f"问题：{question}\n\n"
        f"参考知识：\n{context}\n\n"
        f"模型回答：\n{answer}\n\n"
        f"评估维度：{criterion}\n\n"
        "请严格按 JSON 输出：{\"score\": 0到1的小数, \"reason\": \"一句话\"}"
    )
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        obj = json.loads(raw)
        return {"score": round(float(obj["score"]), 3), "reason": obj.get("reason", "")}
    except Exception:
        return {"score": 0.5, "reason": f"解析失败: {raw[:80]}"}


def _extract_numbers(text: str) -> list[float]:
    """从文本中提取所有数字（含小数）。"""
    return [float(m) for m in re.findall(r"\d+\.?\d*", text)]


def _skip_item(item_id: str, name: str, reason: str) -> dict:
    return {"id": item_id, "name": name, "status": "skip", "detail": {"reason": reason}}


# ── 各项评估函数 ──────────────────────────────────────────────

def eval_faithfulness_and_relevancy(llm, vs) -> tuple[dict, dict]:
    """项 16 + 17：共用同一批 Q&A，分别判断忠实性和相关性。"""
    FAITH_CRITERION = (
        "Faithfulness（忠实性）：回答中所有具体事实（数字、名称、定义）"
        "是否来自参考知识，有无凭空捏造。0=完全虚构，1=完全忠实原文。"
    )
    RELEVANCY_CRITERION = (
        "Answer Relevancy（相关性）：回答是否直接回答了问题，"
        "有无跑题或大幅偏离。0=完全不相关，1=完全切题。"
    )

    qa_records = []
    for q in GEN_QUESTIONS:
        ctx = _retrieve(vs, q["query"])
        answer = _generate(llm, q["query"], ctx)
        faith = _judge(llm, q["query"], ctx, answer, FAITH_CRITERION)
        relevancy = _judge(llm, q["query"], ctx, answer, RELEVANCY_CRITERION)
        qa_records.append({
            "id": q["id"],
            "query": q["query"],
            "context_preview": ctx[:120].replace("\n", " "),
            "answer": answer,
            "faithfulness": faith,
            "relevancy": relevancy,
        })

    avg_faith = round(sum(r["faithfulness"]["score"] for r in qa_records) / len(qa_records), 3)
    avg_rel = round(sum(r["relevancy"]["score"] for r in qa_records) / len(qa_records), 3)

    item16 = {
        "id": "16", "name": "faithfulness",
        "status": "pass" if avg_faith >= 0.7 else "fail",
        "detail": {
            "avg_score": avg_faith, "threshold": 0.7,
            "per_query": [{"id": r["id"], "query": r["query"],
                           "score": r["faithfulness"]["score"],
                           "reason": r["faithfulness"]["reason"],
                           "answer": r["answer"]} for r in qa_records],
        },
    }
    item17 = {
        "id": "17", "name": "answer_relevancy",
        "status": "pass" if avg_rel >= 0.7 else "fail",
        "detail": {
            "avg_score": avg_rel, "threshold": 0.7,
            "per_query": [{"id": r["id"], "query": r["query"],
                           "score": r["relevancy"]["score"],
                           "reason": r["relevancy"]["reason"]} for r in qa_records],
        },
    }
    return item16, item17


def eval_domain_correctness(llm) -> dict:
    """项 18：领域正确性确定性检查（无 LLM judge，直接验证数字/关键词）。"""
    results = []
    for case in DOMAIN_CASES:
        answer = _generate(llm, case["question"], "（无外部参考，直接基于骑行专业知识回答）")
        nums = _extract_numbers(answer)

        if case["check"] == "range":
            # 正确区间：correct_lo 和 correct_hi 各允许 ±3W
            lo_ok = any(abs(n - case["correct_lo"]) <= 3 for n in nums)
            hi_ok = any(abs(n - case["correct_hi"]) <= 3 for n in nums)
            passed = lo_ok and hi_ok
            detail = {
                "expected_lo": case["correct_lo"], "expected_hi": case["correct_hi"],
                "found_numbers": nums, "lo_matched": lo_ok, "hi_matched": hi_ok,
                "passed": passed, "note": case["note"],
            }

        elif case["check"] == "value":
            close = [n for n in nums if abs(n - case["expected"]) <= case["tolerance"]]
            passed = len(close) >= 1
            detail = {"expected": case["expected"], "tolerance": case["tolerance"],
                      "found_numbers": nums, "close_match": close, "passed": passed}

        else:  # keywords
            found = [kw for kw in case["expected_keywords"] if kw in answer]
            passed = len(found) >= 1
            detail = {"expected_keywords": case["expected_keywords"],
                      "found": found, "passed": passed}

        results.append({
            "id": case["id"], "question": case["question"],
            "answer": answer, "passed": passed,
            "note": case["note"], "detail": detail,
        })

    all_pass = all(r["passed"] for r in results)
    return {
        "id": "18", "name": "domain_correctness",
        "status": "pass" if all_pass else "fail",
        "detail": {
            "cases_passed": sum(r["passed"] for r in results),
            "cases_total": len(results),
            "per_case": results,
        },
    }


def eval_hallucination(llm) -> dict:
    """项 19：幻觉专项——注入残缺档案，检测是否编造具体数字。"""
    results = []
    for case in HALLUCINATION_CASES:
        answer = _generate(llm, case["question"], HALLUCINATION_PROFILE)
        # 检测危险模式：是否出现了看起来像具体数据的数字
        danger_found = []
        for pattern in case["danger_patterns"]:
            matches = re.findall(pattern, answer)
            if matches:
                danger_found.extend(matches)

        hallucinated = len(danger_found) > 0
        results.append({
            "id": case["id"], "question": case["question"],
            "answer": answer, "hallucinated": hallucinated,
            "danger_found": danger_found, "note": case["note"],
        })

    any_hallucinated = any(r["hallucinated"] for r in results)
    return {
        "id": "19", "name": "hallucination_test",
        "status": "fail" if any_hallucinated else "pass",
        "detail": {
            "hallucinated_count": sum(r["hallucinated"] for r in results),
            "total": len(results),
            "per_case": results,
        },
    }


def eval_plan_reasonableness() -> dict:
    """项 20：计划合理性——TSS 总量、单次上限、休息日。"""
    try:
        from rag import run_plan_pipeline
        result = run_plan_pipeline("本周训练计划，侧重有氧基础")
        plan = result.get("plan", {})
        events = plan.get("events", [])

        tss_values = [e.get("load_target", 0) for e in events]
        total_tss = sum(tss_values)
        rest_days = sum(1 for t in tss_values if t == 0)
        max_single = max(tss_values) if tss_values else 0
        state = result.get("state_analysis", {})
        week_type = state.get("week_type", "正常训练周")
        tss_limit = state.get("tss_limit", 500)

        # 恢复周允许更低 TSS；以状态分析给出的 tss_limit 为上限
        tss_lo = 100 if "恢复" in week_type else 350
        tss_hi = tss_limit if tss_limit else 700

        checks = {
            "total_tss_in_range": tss_lo <= total_tss <= tss_hi,
            "has_rest_day": rest_days >= 1,
            "no_extreme_single_session": max_single <= 200,
            "has_events": len(events) >= 5,
        }
        passed = all(checks.values())

        return {
            "id": "20", "name": "plan_reasonableness",
            "status": "pass" if passed else "fail",
            "detail": {
                "week_type": week_type, "tss_range_used": [tss_lo, tss_hi],
                "total_tss": total_tss, "rest_days": rest_days,
                "max_single_tss": max_single, "events_count": len(events),
                "checks": checks,
                "plan_summary": plan.get("summary", ""),
                "events": [{"date": e.get("date"), "name": e.get("name"),
                            "tss": e.get("load_target", 0)} for e in events],
            },
        }
    except Exception as e:
        return {
            "id": "20", "name": "plan_reasonableness",
            "status": "skip",
            "detail": {"reason": f"plan_pipeline 执行失败: {e}"},
        }


def eval_ablation(llm, vs) -> dict:
    """项 21：消融实验——有无检索，回答质量对比（LLM judge）。"""
    CRITERION = (
        "哪个回答更准确、更具体、更有实际指导价值？"
        "A=有知识库参考，B=无知识库参考。"
        "输出 JSON：{\"winner\": \"A\"/\"B\"/\"tie\", \"reason\": \"一句话\"}"
    )

    records = []
    for q in ABLATION_QUESTIONS:
        ctx_with = _retrieve(vs, q["query"])
        ctx_without = "（无知识库参考，请基于通用骑行知识作答）"

        answer_with = _generate(llm, q["query"], ctx_with)
        answer_without = _generate(llm, q["query"], ctx_without)

        from langchain_core.messages import HumanMessage
        prompt = (
            f"问题：{q['query']}\n\n"
            f"A（有知识库）：{answer_with}\n\n"
            f"B（无知识库）：{answer_without}\n\n"
            f"{CRITERION}"
        )
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            verdict = json.loads(raw)
        except Exception:
            verdict = {"winner": "tie", "reason": f"解析失败: {raw[:80]}"}

        records.append({
            "id": q["id"], "query": q["query"],
            "answer_with_rag": answer_with,
            "answer_without_rag": answer_without,
            "verdict": verdict,
        })

    rag_wins = sum(1 for r in records if r["verdict"].get("winner") == "A")
    return {
        "id": "21", "name": "ablation",
        "status": "pass" if rag_wins >= len(records) / 2 else "warn",
        "detail": {
            "rag_wins": rag_wins,
            "total": len(records),
            "per_query": records,
        },
    }


# ── 主入口 ────────────────────────────────────────────────────

def run() -> dict:
    if not _has_api_key():
        items = [
            _skip_item(str(i), name, "DEEPSEEK_API_KEY 未设置")
            for i, name in [
                (16, "faithfulness"), (17, "answer_relevancy"),
                (18, "domain_correctness"), (19, "hallucination_test"),
                (20, "plan_reasonableness"), (21, "ablation"),
            ]
        ]
        return {
            "layer": 4, "name": "generation", "passed": False,
            "score": 0.0, "items": items,
            "errors": ["DEEPSEEK_API_KEY 未设置，所有项跳过"], "duration_s": None,
        }

    llm = _get_llm(temperature=0.3)
    llm_judge = _get_llm(temperature=0.0)   # judge 用确定性温度
    vs = _get_vectorstore()

    print("  项16+17: faithfulness & relevancy...")
    item16, item17 = eval_faithfulness_and_relevancy(llm_judge, vs)

    print("  项18: domain_correctness...")
    item18 = eval_domain_correctness(llm)

    print("  项19: hallucination...")
    item19 = eval_hallucination(llm)

    print("  项20: plan_reasonableness...")
    item20 = eval_plan_reasonableness()

    print("  项21: ablation...")
    item21 = eval_ablation(llm_judge, vs)

    items = [item16, item17, item18, item19, item20, item21]
    passed_count = sum(1 for i in items if i["status"] == "pass")

    return {
        "layer": 4, "name": "generation",
        "passed": not any(i["status"] == "fail" for i in items),
        "score": round(passed_count / len(items), 2),
        "items": items, "errors": [], "duration_s": None,
    }


if __name__ == "__main__":
    t0 = time.time()
    print("\n=== Layer 4: 生成质量 ===")
    result = run()
    result["duration_s"] = round(time.time() - t0, 2)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RESULTS_DIR / f"layer4_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer4_generation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    icons = {"pass": "✓", "warn": "△", "fail": "✗", "skip": "–"}
    for item in result["items"]:
        icon = icons[item["status"]]
        d = item.get("detail", {})
        extra = ""
        if item["id"] == "16":
            extra = f"  avg={d.get('avg_score')}"
        elif item["id"] == "17":
            extra = f"  avg={d.get('avg_score')}"
        elif item["id"] == "18":
            extra = f"  {d.get('cases_passed')}/{d.get('cases_total')} passed"
        elif item["id"] == "19":
            extra = f"  hallucinated={d.get('hallucinated_count')}/{d.get('total')}"
        elif item["id"] == "20":
            extra = f"  total_tss={d.get('total_tss')}  rest={d.get('rest_days')}"
        elif item["id"] == "21":
            extra = f"  rag_wins={d.get('rag_wins')}/{d.get('total')}"
        print(f"  [{icon}] 项{item['id']} {item['name']}: {item['status']}{extra}")

    print(f"\n得分：{result['score']}  passed={result['passed']}")
    print(f"结果写入：{out_path}  耗时：{result['duration_s']}s")

    # ── 打印每个案例的回答 ────────────────────────────────────────
    for item in result["items"]:
        if item["status"] == "skip":
            continue
        print(f"\n{'─'*60}")
        print(f"项{item['id']} {item['name']} 详情：")
        d = item.get("detail", {})

        if item["id"] in ("16", "17"):
            key = "faithfulness" if item["id"] == "16" else "relevancy"
            for pq in d.get("per_query", []):
                print(f"  [{pq['id']}] score={pq['score']}  {pq['reason']}")
                print(f"       Q: {pq['query']}")
                print(f"       A: {pq.get('answer', '')[:120]}")

        elif item["id"] == "18":
            for c in d.get("per_case", []):
                mark = "✓" if c["passed"] else "✗"
                print(f"  [{mark}] {c['id']}: {c['question']}")
                print(f"       A: {c['answer'][:100]}")

        elif item["id"] == "19":
            for c in d.get("per_case", []):
                mark = "幻觉!" if c["hallucinated"] else "✓"
                print(f"  [{mark}] {c['id']}: {c['question']}")
                print(f"       A: {c['answer'][:120]}")
                if c["danger_found"]:
                    print(f"       危险词: {c['danger_found']}")

        elif item["id"] == "20":
            for ev in d.get("events", []):
                print(f"  {ev['date']} {ev['name']} TSS={ev['tss']}")

        elif item["id"] == "21":
            for r in d.get("per_query", []):
                v = r["verdict"]
                print(f"  [{r['id']}] winner={v.get('winner')}  {v.get('reason')}")
                print(f"       Q: {r['query']}")
                print(f"       A(RAG):    {r['answer_with_rag'][:80]}")
                print(f"       A(no-RAG): {r['answer_without_rag'][:80]}")
