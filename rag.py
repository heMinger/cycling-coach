from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# ── 第一步：加载文档（不变）────────────────────────────────────
def load_documents(knowledge_dir="data/knowledge", user_dir="data/user_data"):
    loader_knowledge = DirectoryLoader(
        knowledge_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    loader_user = DirectoryLoader(
        user_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader_knowledge.load() + loader_user.load()
    print(f"加载文档数：{len(docs)}")
    return docs

# ── 第二步：切片（不变）──────────────────────────────────────
def split_documents(docs):
    splitter = MarkdownTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"切片数量：{len(chunks)}")
    return chunks

# ── 第三步：向量化 + 存库（不变）─────────────────────────────
def build_vectorstore(chunks):
    print("正在加载 Embedding 模型...")
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="BAAI/bge-small-zh-v1.5",
    #     model_kwargs={"device": "cpu"},
    #     encode_kwargs={"normalize_embeddings": True}
    # )
    embeddings = HuggingFaceEmbeddings(
        model_name="/home/lmh/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620",
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

# ── 第四步：构建 LCEL 链（核心改动）──────────────────────────
#
# 改造前（手写）：
#   retrieve() → 手拼字符串 → openai.client.chat.completions.create()
#
# 改造后（LCEL）：
#   retriever | ChatPromptTemplate | ChatOpenAI | StrOutputParser
#
def build_chain(vectorstore):
    # 1. LLM：用 LangChain 的 ChatOpenAI 封装 DeepSeek
    #    之前：openai.OpenAI(api_key=..., base_url=...)
    #    现在：ChatOpenAI(...)，接口统一，换模型只改这里
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.7
    )

    # 2. Retriever：把 vectorstore 包装成标准检索接口
    #    之前：vectorstore.similarity_search(query, k=3)
    #    现在：retriever.invoke(query)，接口标准化，可以换检索策略
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 3. 读取用户档案（固定注入，不靠检索）
    with open("data/user_data/user_profile.md", "r", encoding="utf-8") as f:
        user_profile = f.read()

    # 4. Prompt Template：参数化 prompt，不再手拼字符串
    #    之前：system_prompt = "...{context}...".format(context=context)
    #    现在：ChatPromptTemplate，有结构（system/human），可复用
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的公路骑行教练，风格简练直接。

## 用户档案（每次必读）
{user_profile}

## 相关知识参考
{context}

## 回答要求
- 直接给结论，不要重复用户说的内容
- 引用具体数据支撑判断（如 IF、TSS、FTP占比）
- 控制在200字以内
- 不需要总结段
"""),
        ("human", "{question}")
    ])

    # 5. 把检索到的 chunk 列表拼成字符串
    def format_docs(docs):
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    # 6. 构建 LCEL 链
    #
    #    数据流：
    #    用户输入(question)
    #      → 同时触发三路：retriever检索 / user_profile固定注入 / question透传
    #      → 填入 prompt 模板
    #      → 传给 llm 生成回答
    #      → StrOutputParser 解析成字符串
    #
    chain = (
        {
            # 动态检索：用问题去向量库找相关 chunk
            "context": retriever | format_docs,
            # 固定注入：用户档案每次完整传入
            "user_profile": lambda _: user_profile,
            # 透传：问题原样传给 prompt 的 {question}
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── 主流程 ────────────────────────────────────────────────────
if __name__ == "__main__":
    # 构建向量库
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)

    # 构建链
    chain = build_chain(vectorstore)

    # 用三个问题测试，覆盖不同场景
    questions = [
        "我今天骑了2小时，功率180W，感觉很累，状态怎么样？",
        "我的FTP是多少？最近训练负荷合理吗？",
        "Z2训练应该控制在什么功率范围？"
    ]

    for q in questions:
        print(f"\n问题：{q}")
        print("=" * 50)
        answer = chain.invoke(q)
        print(answer)
