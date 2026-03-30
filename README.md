# 🚴 公路骑行 AI 教练

基于 RAG（检索增强生成）的个性化骑行训练建议系统。结合用户真实训练数据与专业骑行知识库，通过 LangChain + DeepSeek 生成有据可查的训练分析和建议。

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | DeepSeek V3 API |
| RAG 框架 | LangChain LCEL |
| Embedding | BAAI/bge-small-zh-v1.5 |
| 向量数据库 | Chroma |
| API 服务 | FastAPI + Uvicorn |

## 项目结构

```
cycling-coach/
├── rag.py              # RAG 核心：文档加载、向量化、LCEL 链构建
├── api.py              # FastAPI 服务：HTTP 接口封装
├── main.py             # 基础 API 调用测试
├── data/
│   ├── knowledge/      # 专业骑行知识库
│   │   ├── training_zones.md       # 功率训练区间
│   │   ├── ftp_and_metrics.md      # FTP、TSS、IF 等核心指标
│   │   └── recovery_and_adaptation.md  # 恢复与训练适应
│   └── user_data/
│       └── user_profile.md         # 用户训练档案（基于 Intervals.icu 真实数据）
└── .env                # API Key（不提交）
```

## 核心设计

### 双知识库策略
- **专业知识库**：骑行训练理论（功率区间、FTP、恢复原理），通过向量检索动态召回
- **用户档案**：个人 FTP、近期训练记录，固定注入 System Prompt，避免召回不足

### RAG 流程
```
用户问题
  → Embedding（bge-small-zh）
  → Chroma 向量检索（Top-3 相关 chunk）
  → 注入用户档案 + 检索结果到 Prompt
  → DeepSeek V3 生成回答
```

### LCEL 链结构
```python
chain = (
    {
        "context": retriever | format_docs,   # 动态检索
        "user_profile": lambda _: profile,    # 固定注入
        "question": RunnablePassthrough()     # 透传
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

## 快速开始

### 1. 安装依赖
```bash
conda create -n cycling-coach python=3.11 -y
conda activate cycling-coach
pip install langchain langchain-community langchain-openai langchain-huggingface langchain-chroma chromadb sentence-transformers fastapi uvicorn python-dotenv openai
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 填入 DeepSeek API Key
```

### 3. 启动服务
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 4. 调用接口
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "我今天骑了2小时，功率180W，感觉很累，状态怎么样？"}'
```

### 5. API 文档
启动后访问 `http://localhost:8000/docs` 查看交互式 API 文档。

## 示例输出

**问题**：我今天骑了2小时，功率180W，感觉很累，状态怎么样？

**回答**：
> 状态处于疲劳期。今天 IF 约 0.89，属高强度阈值训练，TSS 预估约 160。结合近四周 1714 的总负荷和 -21 的负状态值，疲劳感是近期累积（特别是第11周 691 TSS）的正常反应。建议明天安排主动恢复或休息。

**问题**：Z2 训练应该控制在什么功率范围？

**回答**：
> 根据你的 FTP 202W，Z2 功率范围应为 56-75% FTP，即 **114-152W**。从第10周数据看，你 Z2 训练功率（160-165W）已接近 Z2 上限，建议适当下调至 140-150W 区间，确保真正的有氧刺激。
