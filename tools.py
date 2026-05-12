import json
import ast
import os
import concurrent.futures
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.types import interrupt
from langgraph.prebuilt import InjectedState

load_dotenv()


# ── 内部辅助函数 ──────────────────────────────────────────────

def _fetch_context() -> str:
    """并行拉取 Strava + Intervals，三层降级：实时 → cache → 静态档案。"""
    CACHE_PATH = "cache/user_profile_cache.md"
    try:
        from strava_client import StravaClient
        from intervals_client import IntervalsClient
        strava = StravaClient()
        intervals = IntervalsClient()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_strava = executor.submit(strava.build_activity_context, 14)
            f_intervals = executor.submit(intervals.build_user_context, 0)
            activity_context = f_strava.result()
            wellness_context = f_intervals.result()

        data = f"""## 基本信息
- 姓名：Minghe，女，23岁，体重63kg
- FTP：202W，最大心率：194bpm

## 训练目标
- 截止：2026年6月15日
- 目标FTP：220W（差18W），目标体重：60kg（差3kg）

{wellness_context}
{activity_context}
"""
        os.makedirs("cache", exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(data)
        return data

    except Exception as e:
        print(f"实时 API 失败：{e}")
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                print("降级：使用缓存")
                return f.read()
        except FileNotFoundError:
            print("降级：使用静态档案")
            with open("data/user_data/user_profile.md", "r", encoding="utf-8") as f:
                return f.read()


def _get_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.7,
    )


def _parse_json(raw: str) -> dict:
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    return json.loads(raw.strip())


def _format_plan(state_analysis: dict, plan: dict) -> str:
    """把结构化计划转成可直接展示给用户的纯文本。"""
    lines = []
    if state_analysis:
        lines.append(f"状态：{state_analysis.get('tsb_interpretation', '')}")
        lines.append(f"本周类型：{state_analysis.get('week_type', '')}，TSS上限 {state_analysis.get('tss_limit', '-')}")
        lines.append("")
    lines.append(f"计划：{plan.get('summary', '')}")
    lines.append("")
    for ev in plan.get("events", []):
        tss = ev.get("load_target", 0)
        tss_str = f"（目标TSS {tss}）" if tss > 0 else "（休息日）"
        lines.append(f"{ev['date']} {ev.get('name', '')}{tss_str}")
        if ev.get("description"):
            lines.append(f"  {ev['description']}")
    lines.append("")
    lines.append("确认写入 Intervals.icu 日历吗？")
    return "\n".join(lines)


# ── Tool 1: get_full_context ──────────────────────────────────

@tool
def get_full_context() -> str:
    """获取用户实时训练上下文：CTL/ATL/TSB 状态、近14天活动记录、长期记忆。

    这是获取用户个人训练数据的唯一途径。回答任何涉及用户个人状态、
    疲劳程度、近期训练表现的问题前，必须先调用此工具。
    内部并行拉取 Strava 和 Intervals.icu，三层降级保证可用性。

    不适用：用户询问通用骑行知识（区间定义/指标含义），那类问题用 search_knowledge。
    """
    import memory as mem
    context = _fetch_context()
    long_term = mem.get_all_memories()
    memory_section = f"\n## 关于你的长期记忆\n{long_term}" if long_term else "\n## 关于你的长期记忆\n（暂无）"
    return context + memory_section


# ── 模块级 Embedding + Vectorstore 单例（避免每次调用重新加载模型）──

def _get_vectorstore():
    """懒加载并缓存 embedding 模型和向量库，整个进程只初始化一次。"""
    if _get_vectorstore._instance is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _get_vectorstore._instance = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
        )
    return _get_vectorstore._instance

_get_vectorstore._instance = None

# L2 距离阈值：超过此值的 chunk 视为无关，不纳入上下文
# 基于 Layer 3 阈值扫描（k=5）：0.85 是 precision/recall 最优平衡点
#   0.80 → precision=0.72 但 recall 掉到 0.90（有效 chunk 被过滤）
#   0.85 → precision=0.65，recall=1.0，avg_hits≈3.1
#   0.90 → precision=0.56，recall=1.0，avg_hits≈3.9（噪声重新进来）
_RETRIEVAL_SCORE_THRESHOLD = 0.85


# ── Tool 2: search_knowledge ──────────────────────────────────

@tool
def search_knowledge(query: str) -> str:
    """在骑行训练知识库中检索专业知识。

    这是回答通用骑行训练知识问题的唯一途径，覆盖训练区间（Z1-Z6）、
    FTP/TSS/IF/NP/CTL/ATL/TSB 等指标定义、恢复理论、过度训练信号等内容。

    不适用：查询用户个人训练数据或状态（用 get_full_context）。

    参数 query：简洁检索词，如「Z2训练功率范围」「TSB负值意味着什么」。
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=5)
    docs = [doc for doc, score in results if score < _RETRIEVAL_SCORE_THRESHOLD]
    if not docs:
        # 全部低于阈值说明 query 超出知识库范围，明确告知 LLM 而非塞噪声
        return "知识库中未找到与此问题相关的内容。若为通用骑行知识请基于专业判断作答，若涉及用户个人数据请改用 get_full_context。"
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── Tool 3: analyze_and_plan ──────────────────────────────────

@tool
def analyze_and_plan(user_request: str) -> dict:
    """分析训练状态并生成个性化训练计划，是制定新计划的唯一正确途径。

    内部自动获取实时数据，无需提前调用 get_full_context。
    两步执行：Step1 分析CTL/ATL/TSB确定本周类型和TSS上限；
    Step2 基于状态约束生成逐天训练安排（JSON）。
    工具返回已格式化的计划文本，收到后直接展示给用户，询问是否确认写入日历。

    不适用：用户只想了解当前状态（用 get_full_context）；
            用户想修改已有计划（用 modify_plan）。

    参数 user_request：用户的计划需求，如「本周训练计划」「侧重提升FTP的计划」。
    返回值包含 _display 字段（可直接展示的格式化文本）供透传给用户。
    """
    import memory as mem

    user_profile = _fetch_context()
    long_term_memory = mem.get_all_memories()
    llm = _get_llm()

    # Step 1：状态分析
    analysis_prompt = f"""你是专业骑行教练。根据以下用户训练数据，分析当前状态。

{user_profile}

请严格按 JSON 输出，不要有任何其他内容：
{{
  "tsb_value": 数字,
  "tsb_interpretation": "TSB是多少，处于什么区间",
  "week_type": "恢复周/积累周/激活周",
  "reasoning": "判断理由，引用具体CTL/ATL/TSB数值",
  "tss_limit": 数字,
  "forbidden_zones": ["不适合的训练强度"],
  "recommended_intensity": "适合的训练强度建议"
}}"""
    try:
        state_analysis = _parse_json(llm.invoke(analysis_prompt).content)
    except Exception:
        state_analysis = {
            "tsb_value": 0, "tsb_interpretation": "状态解析失败",
            "week_type": "正常训练周", "reasoning": "解析失败",
            "tss_limit": 400, "forbidden_zones": [],
            "recommended_intensity": "按正常计划训练",
        }

    # Step 2：计划生成
    constraints = f"""当前状态：{state_analysis.get('tsb_interpretation')}
本周类型：{state_analysis.get('week_type')}
判断依据：{state_analysis.get('reasoning')}
本周TSS上限：{state_analysis.get('tss_limit', 400)}
不适合的强度：{', '.join(state_analysis.get('forbidden_zones', [])) or '无限制'}
强度建议：{state_analysis.get('recommended_intensity')}"""

    memory_section = (
        f"\n关于用户的长期记忆（制定计划时考虑）：\n{long_term_memory}"
        if long_term_memory else ""
    )

    plan_prompt = f"""你是专业骑行教练。已完成状态分析：

{constraints}

用户档案：
{user_profile}
{memory_section}

用户需求：{user_request}

请严格按 JSON 输出：
{{
  "summary": "一句话说明本周计划思路",
  "events": [
    {{
      "date": "2026-05-05",
      "name": "训练名称",
      "description": "具体内容，包含功率区间、时长",
      "load_target": 目标TSS数字
    }}
  ]
}}"""
    plan = _parse_json(llm.invoke(plan_prompt).content)

    return {
        "state_analysis": state_analysis,
        "plan": plan,
        "_display": _format_plan(state_analysis, plan),
    }


# ── Tool 4: modify_plan ───────────────────────────────────────

@tool
def modify_plan(
    modification_request: str,
    state: Annotated[dict, InjectedState],
) -> dict:
    """对当前训练计划进行精确修改，是修改已有计划的唯一正确途径。

    只做用户要求的精确改动，保留其他所有内容。自动读取当前计划，
    工具返回已格式化的修改后计划文本，收到后直接展示，询问是否确认写入日历。

    不适用：用户想全新制定计划（用 analyze_and_plan）。
    前置条件：必须已有 current_plan（用户已生成过计划）。

    参数 modification_request：具体修改描述，如「把周二换到周三」「降低周四强度」。
    """
    current_plan = state.get("current_plan")
    if not current_plan:
        return {"error": "没有已生成的计划，请先使用 analyze_and_plan 生成计划"}

    llm = _get_llm()
    prompt = f"""你是专业骑行教练。用户想修改已有训练计划。

当前计划：
{json.dumps(current_plan, ensure_ascii=False, indent=2)}

修改请求：{modification_request}

要求：只做用户要求的精确修改，保留所有其他内容不变。

请严格按 JSON 输出修改后的完整计划：
{{
  "summary": "更新后的计划说明",
  "events": [...]
}}"""
    updated_plan = _parse_json(llm.invoke(prompt).content)
    return {
        **updated_plan,
        "_display": _format_plan({}, updated_plan),
    }


# ── Tool 5: write_to_calendar ─────────────────────────────────

@tool
def write_to_calendar(state: Annotated[dict, InjectedState]) -> str:
    """将当前训练计划写入 Intervals.icu 日历。不可逆操作。

    当用户明确说「确认」「写入」「好的」等表示同意时调用此工具，无需传入任何参数，
    工具自动读取已生成的计划。调用后系统会再次暂停等待用户在界面上最终确认。

    前置条件：必须已调用过 analyze_and_plan 或 modify_plan 生成计划。
    不适用：用户还未确认计划内容，或尚未生成计划。
    """
    plan = state.get("current_plan")
    if not plan:
        return "没有可写入的计划，请先使用 analyze_and_plan 生成计划"

    confirmed = interrupt({
        "type": "confirm",
        "plan": plan,
        "message": "以下计划将写入 Intervals.icu 日历，确认吗？",
    })

    if not confirmed:
        return "用户已取消，计划未写入"

    from intervals_client import IntervalsClient
    client = IntervalsClient()
    written, errors = 0, []

    for event in plan.get("events", []):
        if event.get("load_target", 0) <= 0:
            continue
        try:
            client.create_event(
                date=event["date"],
                name=event["name"],
                description=event["description"],
                load_target=event.get("load_target"),
            )
            written += 1
        except Exception as e:
            errors.append(f"{event.get('date', '?')}: {e}")

    result = f"成功写入 {written} 天训练计划到 Intervals.icu 日历"
    if errors:
        result += f"\n失败 {len(errors)} 条：{'; '.join(errors)}"
    return result


# ── Tool 6: ask_user ──────────────────────────────────────────

@tool
def ask_user(question: str) -> str:
    """向用户主动提问以获取关键信息，用于 onboarding 或澄清模糊请求。

    适用：长期记忆为空（新用户），需收集 FTP、训练目标、伤病史、固定休息日；
          用户请求模糊需要澄清；长期记忆中有伤病记录，需确认最新状态。
    不适用：已有足够信息时不要过度询问。

    参数 question：向用户提出的具体问题。
    返回：用户的回答，直接作为信息使用。
    """
    answer = interrupt({"type": "input", "question": question})
    return str(answer)


ALL_TOOLS = [
    get_full_context,
    search_knowledge,
    analyze_and_plan,
    modify_plan,
    write_to_calendar,
    ask_user,
]
