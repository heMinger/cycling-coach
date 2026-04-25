from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os
import json

load_dotenv()

# ── 第一步：加载文档 ──────────────────────────────────────────
def load_documents(knowledge_dir="data/knowledge", user_dir="data/user_data"):
    loader_knowledge = DirectoryLoader(
        knowledge_dir, glob="**/*.md", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    loader_user = DirectoryLoader(
        user_dir, glob="**/*.md", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader_knowledge.load() + loader_user.load()
    print(f"加载文档数：{len(docs)}")
    return docs

# ── 第二步：切片 ──────────────────────────────────────────────
def split_documents(docs):
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"切片数量：{len(chunks)}")
    return chunks

# ── 第三步：向量化 + 存库 ─────────────────────────────────────
def build_vectorstore(chunks):
    print("正在加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("向量库构建完成")
    return vectorstore

# ── 公共：获取用户数据 ────────────────────────────────────────
def get_user_profile(_):
    """
    线程池并行拉取 Strava + Intervals，三层降级：
    实时数据 → 缓存 → 静态档案
    """
    import concurrent.futures, time
    CACHE_PATH = "cache/user_profile_cache.md"
    try:
        from strava_client import StravaClient
        from intervals_client import IntervalsClient
        strava = StravaClient()
        intervals = IntervalsClient()

        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_strava = executor.submit(strava.build_activity_context, 14)
            f_intervals = executor.submit(intervals.build_user_context, 0)
            activity_context = f_strava.result()
            wellness_context = f_intervals.result()
        print(f"并行拉取耗时：{time.time()-start:.2f}s")

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
        print(f"API 获取失败：{e}")
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                print("使用缓存数据")
                return f.read()
        except FileNotFoundError:
            print("缓存不存在，使用静态档案")
            with open("data/user_data/user_profile.md", "r", encoding="utf-8") as f:
                return f.read()

# ── 公共：LLM 实例 ────────────────────────────────────────────
def get_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.7
    )

# ── 意图路由 ──────────────────────────────────────────────────
def route_request(question: str) -> str:
    llm = get_llm()
    prompt = f"""判断用户请求的类型，只返回一个英文单词：
- plan：用户想制定、生成或调整训练计划
- analysis：用户想了解当前训练状态、疲劳程度、是否适合训练
- knowledge：用户在问骑行训练知识（训练区间、FTP含义等）
- general：其他

用户请求：{question}
"""
    result = llm.invoke(prompt).content.strip().lower()
    if result not in ["plan", "analysis", "knowledge", "general"]:
        return "general"
    return result

# ── 问答链 ────────────────────────────────────────────────────
def build_chain(vectorstore):
    """
    接受字典输入：
    {
        "question": "用户问题",
        "short_term_history": "最近N轮对话文本"（可选，默认空）
    }
    """
    import memory as mem

    llm = get_llm()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的公路骑行教练，风格简练直接。

## 回答要求
- 直接给结论，不要重复用户说的内容
- 引用具体数据支撑判断（如 IF、TSS、FTP占比）
- 控制在200字以内
- 不需要总结段
- 不要使用 ** 加粗符号，直接输出纯文本

## 主动询问规则
如果长期记忆里有受伤/身体不适或目标赛事信息，在回答前主动确认最新状态。
"""),
        ("human",
"""## 我的训练档案
{user_profile}

## 关于我的长期记忆
{long_term_memory}

## 相关知识参考
{context}

{short_term_history}

## 我的问题
{question}""")
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    def get_long_term_memory(_):
        memories = mem.get_all_memories()
        return memories if memories else "（暂无长期记忆）"

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "user_profile": RunnableLambda(get_user_profile),
            "long_term_memory": RunnableLambda(get_long_term_memory),
            "short_term_history": lambda x: x.get("short_term_history", ""),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── 计划 Pipeline ─────────────────────────────────────────────
def run_state_analysis(user_profile: str) -> dict:
    llm = get_llm()
    prompt = f"""你是专业骑行教练。根据以下用户训练数据，分析当前状态。

{user_profile}

请严格按照以下 JSON 格式输出，不要有任何其他内容：
{{
  "tsb_value": 数字,
  "tsb_interpretation": "TSB是多少，处于什么区间（精力充沛/最优训练/过渡/高风险）",
  "week_type": "本周类型（恢复周/积累周/激活周）",
  "reasoning": "判断理由，引用具体CTL/ATL/TSB数值",
  "tss_limit": 数字,
  "forbidden_zones": ["不适合的训练强度，如Z5、Z6"],
  "recommended_intensity": "适合的训练强度建议"
}}
"""
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {
            "tsb_value": 0, "tsb_interpretation": "状态分析暂时无法解析",
            "week_type": "正常训练周", "reasoning": raw,
            "tss_limit": 400, "forbidden_zones": [],
            "recommended_intensity": "按正常计划训练"
        }

def run_plan_generation(
    user_request: str, state_analysis: dict,
    user_profile: str, long_term_memory: str = ""
) -> dict:
    llm = get_llm()
    constraints = f"""当前状态：{state_analysis.get('tsb_interpretation', '')}
本周类型：{state_analysis.get('week_type', '')}
判断依据：{state_analysis.get('reasoning', '')}
本周TSS上限：{state_analysis.get('tss_limit', 400)}
不适合的强度：{', '.join(state_analysis.get('forbidden_zones', [])) or '无限制'}
强度建议：{state_analysis.get('recommended_intensity', '')}"""

    memory_section = (
        f"\n关于用户的长期记忆（制定计划时需考虑）：\n{long_term_memory}"
        if long_term_memory else ""
    )

    prompt = f"""你是专业骑行教练。已完成状态分析如下：

{constraints}

用户档案：
{user_profile}
{memory_section}

用户需求：{user_request}

请严格按照以下 JSON 格式输出，不要有任何其他内容：
{{
  "summary": "一句话说明本周计划思路",
  "events": [
    {{
      "date": "2026-04-28",
      "name": "训练名称",
      "description": "具体内容，包含功率区间、时长、组数等",
      "load_target": 目标TSS数字
    }}
  ]
}}
"""
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        raise ValueError(f"计划生成解析失败：{raw}")

def run_plan_pipeline(user_request: str) -> dict:
    import memory as mem
    user_profile = get_user_profile(None)
    long_term_memory = mem.get_all_memories()

    print("计划Pipeline - Step 1: 状态分析...")
    state_analysis = run_state_analysis(user_profile)

    print("计划Pipeline - Step 2: 生成训练计划...")
    plan = run_plan_generation(user_request, state_analysis, user_profile, long_term_memory)

    return {"state_analysis": state_analysis, "plan": plan}


if __name__ == "__main__":
    import memory as mem
    mem.init_db()

    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    chain = build_chain(vectorstore)

    print("\n=== 意图路由测试 ===")
    for q in ["我今天状态怎么样？", "帮我制定下周训练计划", "Z2训练是什么？"]:
        print(f"  {q} → {route_request(q)}")

    print("\n=== 问答链测试 ===")
    answer = chain.invoke({"question": "我今天状态怎么样？", "short_term_history": ""})
    print(answer)
