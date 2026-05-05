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


# ── 内部辅助函数（不暴露为工具）─────────────────────────────────

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
    """解析 LLM 返回的 JSON，兼容 markdown 代码块包裹格式。"""
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    return json.loads(raw.strip())


# ── Tool 1: get_full_context ──────────────────────────────────

@tool
def get_full_context() -> str:
    """获取用户完整训练上下文，包括实时训练状态、近14天活动和长期记忆。

    适用：对话开始时，回答任何个人训练相关问题前调用一次。
    不适用：纯知识类问题（如「Z2是什么」）不需要调用。

    返回：包含 CTL/ATL/TSB、近期活动摘要、长期记忆的格式化文本。
    """
    import memory as mem

    context = _fetch_context()
    long_term = mem.get_all_memories()
    memory_section = f"\n## 关于你的长期记忆\n{long_term}" if long_term else "\n## 关于你的长期记忆\n（暂无）"
    return context + memory_section


# ── Tool 2: search_knowledge ──────────────────────────────────

@tool
def search_knowledge(query: str) -> str:
    """在骑行训练专业知识库中检索相关知识。

    适用：用户询问训练区间（Z1-Z6）、FTP/TSS/IF/NP 等指标含义、
          恢复理论、过度训练信号等通用骑行知识。
    不适用：查询用户个人训练数据（用 get_full_context）。

    参数 query：简洁检索词，如「Z2训练功率范围」。
    返回：最相关的3段知识文本。
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── Tool 3: analyze_and_plan ──────────────────────────────────

@tool
def analyze_and_plan(user_request: str) -> dict:
    """分析当前训练状态并生成个性化训练计划。

    内部分两步：
    Step1 根据 CTL/ATL/TSB 判断本周类型（恢复/积累/激活周），确定 TSS 上限和禁止强度区间。
    Step2 基于状态约束和用户需求生成具体训练计划（JSON）。

    适用：用户想制定新训练计划时。
    不适用：只想了解当前状态（用 get_full_context）；想修改已有计划（用 modify_plan）。

    参数 user_request：用户的计划需求描述。
    返回：{"state_analysis": {...}, "plan": {"summary": "...", "events": [...]}}
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
  "tsb_interpretation": "TSB是多少，处于什么区间（精力充沛/最优训练/过渡/高风险）",
  "week_type": "本周类型（恢复周/积累周/激活周）",
  "reasoning": "判断理由，引用具体CTL/ATL/TSB数值",
  "tss_limit": 数字,
  "forbidden_zones": ["不适合的训练强度，如Z5、Z6"],
  "recommended_intensity": "适合的训练强度建议"
}}"""

    try:
        state_analysis = _parse_json(llm.invoke(analysis_prompt).content)
    except (json.JSONDecodeError, IndexError):
        state_analysis = {
            "tsb_value": 0,
            "tsb_interpretation": "状态解析失败，按正常周处理",
            "week_type": "正常训练周",
            "reasoning": "解析失败",
            "tss_limit": 400,
            "forbidden_zones": [],
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

请严格按 JSON 输出，不要有任何其他内容：
{{
  "summary": "一句话说明本周计划思路",
  "events": [
    {{
      "date": "2026-05-05",
      "name": "训练名称",
      "description": "具体内容，包含功率区间、时长、组数等",
      "load_target": 目标TSS数字
    }}
  ]
}}"""

    plan = _parse_json(llm.invoke(plan_prompt).content)
    return {"state_analysis": state_analysis, "plan": plan}


# ── Tool 4: modify_plan ───────────────────────────────────────

@tool
def modify_plan(
    modification_request: str,
    state: Annotated[dict, InjectedState],
) -> dict:
    """对已生成的训练计划进行精确修改，不重新生成整个计划。

    适用：用户已有计划，提出具体修改，如「把周二换到周三」「降低周四强度」「增加休息日」。
    不适用：用户想全新制定计划（用 analyze_and_plan）。
    前置条件：必须已存在 current_plan，否则提示用户先生成计划。

    参数 modification_request：用户的具体修改描述。
    返回：修改后的完整计划 JSON，格式与 analyze_and_plan 的 plan 字段相同。
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

    return _parse_json(llm.invoke(prompt).content)


# ── Tool 5: write_to_calendar ─────────────────────────────────

@tool
def write_to_calendar(plan: dict) -> str:
    """将训练计划写入 Intervals.icu 日历。这是不可逆操作。

    调用此工具后系统会暂停，等待用户在界面上确认后才执行写入。
    用户取消则不写入。不要在其他地方处理确认逻辑。

    参数 plan：包含 events 列表的计划 dict，每个 event 需有 date/name/description/load_target。
    返回：写入结果摘要。
    """
    confirmed = interrupt({
        "type": "confirm",
        "plan": plan,
        "message": "以下计划将写入 Intervals.icu 日历，确认吗？",
    })

    if not confirmed:
        return "用户已取消，计划未写入"

    from intervals_client import IntervalsClient
    client = IntervalsClient()

    written = 0
    errors = []
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
        result += f"\n写入失败 {len(errors)} 条：{'; '.join(errors)}"
    return result


# ── Tool 6: ask_user ──────────────────────────────────────────

@tool
def ask_user(question: str) -> str:
    """向用户主动提问，获取信息或确认状态。

    适用场景：
    - 长期记忆中有伤病记录，需确认最新状态
    - 用户请求模糊，需要澄清
    - Onboarding：长期记忆为空时收集基本信息（FTP/目标/伤病/固定休息日）
    - 目标赛事临近，确认参赛状态

    不适用：信息已足够时不要过度询问。

    参数 question：向用户提出的具体问题。
    返回：用户的回答文本。
    """
    answer = interrupt({"type": "input", "question": question})
    return str(answer)


# 导出工具列表
ALL_TOOLS = [
    get_full_context,
    search_knowledge,
    analyze_and_plan,
    modify_plan,
    write_to_calendar,
    ask_user,
]
