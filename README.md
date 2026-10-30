# 骑行 AI 教练

## 计划质量改进分支

当前计划生成已改为 `planning.py`：读取完整训练依据卡片，结合历史对话和明确约束，生成分段课程并校验。详见 [诊断与测试报告](docs/plan-quality-review.md)。

关键行为与上方旧版说明不同：
- 计划使用全部已审核的小型知识卡片，不走问答的 Top-3 检索。
- 用户档案不再与训练知识混合索引；不再写死 FTP 或旧目标。
- 当前数据获取失败时显示需补充信息，不生成默认 400 TSS 计划。
- 从 `/ui` 访问时前端连接同源后端，不再连接仓库中写死的服务器地址。
- 表单可设置开始日期、7天总时间、单日时间、不可训练日期及确认的 FTP。
- 课程参数调整通过修改需求后重新生成；写入前由步骤重建描述，避免自由文本与计算值不一致。

新的 `/plan/generate` 请求可包含 `session_id` 和 `constraints`：

```json
{"question":"根据近期训练安排耐力训练","session_id":"my-session","constraints":{"days":7,"max_minutes":180,"max_daily_minutes":60,"ftp":200,"unavailable_dates":[]}}
```

`start_date` 省略时以亚洲/上海当天为起点，需生成下周时请在界面选择日期。
响应为 `{state_analysis, plan, constraints, validation, evidence}`；`plan.status` 为 `draft` 或 `needs_info`。
`/plan/create` 必须接收 `{plan, constraints}`，不再接受旧版 `{events}`。前后端需同时更新。

测试（Python 3.11+）：

```sh
pip install -r requirements-test.txt
python -m unittest discover -s tests -v
node tests/frontend.test.cjs
python evaluations/run.py
```

最后一条仅导出提示词预览；配置 `DEEPSEEK_API_KEY` 后加 `--live` 才会调用真实模型。测试不会访问训练平台或写入日历。完整应用依赖仍见前文安装步骤。


## 初始版本说明（接口以当前改进说明为准）

# 🚴 骑行 AI 教练

基于 RAG 的垂直领域智能训练对话系统。对接 Strava 和 Intervals.icu 双平台 API，实时读取训练数据，通过 LangChain + DeepSeek V3 生成个性化训练分析和建议，并支持将 AI 生成的训练计划直接写回 Intervals 日历。

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | DeepSeek V3 API |
| RAG 框架 | LangChain LCEL |
| Embedding | BAAI/bge-small-zh-v1.5 |
| 向量数据库 | Chroma |
| API 服务 | FastAPI + Uvicorn |
| 训练数据 | Strava API（OAuth 2.0）|
| 训练状态 | Intervals.icu API |

## 系统架构

```
用户（浏览器 index.html）
    ↓ HTTP
FastAPI (api.py)
    ├── /ask          → LangChain LCEL Chain
    │                      ├── 固定注入：Strava 活动 + Intervals CTL/ATL/TSB
    │                      ├── 动态检索：Chroma 向量库（专业骑行知识）
    │                      └── DeepSeek V3 生成回答
    ├── /plan/generate → 同上，约束 JSON 结构化输出
    └── /plan/create   → Intervals.icu API 写入训练日历
```

## 核心设计

### 双知识库策略
- **专业知识库**：骑行训练理论（功率区间、FTP、恢复原理），通过向量检索动态召回
- **用户实时数据**：从 Strava/Intervals 拉取的训练记录和状态值，固定注入 System Prompt，避免召回不足

### RAG 流程
```
用户问题
  → bge-small-zh 向量化
  → Chroma 检索 Top-3 专业知识 chunk
  → 注入实时用户数据（Strava 近14天 + Intervals CTL/ATL/TSB）
  → DeepSeek V3 生成个性化回答
```

### LCEL 链结构
```python
chain = (
    {
        "context": retriever | format_docs,    # 动态检索专业知识
        "user_profile": get_user_profile,      # 实时拉取用户数据
        "question": RunnablePassthrough()      # 透传问题
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

### 训练计划闭环
```
AI 生成计划（JSON）→ 用户编辑卡片 → 写入 Intervals 日历
```

## 项目结构

```
cycling-coach/
├── rag.py                  # RAG 核心：LCEL 链构建、实时数据注入
├── api.py                  # FastAPI 服务：问答/计划生成/写入接口
├── strava_client.py        # Strava API：OAuth 授权、训练数据读取
├── intervals_client.py     # Intervals API：状态读取、计划写入
├── index.html              # 交互式前端：对话 + 训练计划编辑器
├── main.py                 # 基础测试
├── data/
│   ├── knowledge/          # 专业骑行知识库
│   │   ├── training_zones.md
│   │   ├── ftp_and_metrics.md
│   │   └── recovery_and_adaptation.md
│   └── user_data/
│       └── user_profile.md # 降级备用档案
└── .env                    # API Keys（不提交）
```

## 快速开始

### 1. 安装依赖
```bash
conda create -n cycling-coach python=3.11 -y
conda activate cycling-coach
pip install langchain langchain-community langchain-openai langchain-huggingface \
    langchain-chroma chromadb sentence-transformers fastapi uvicorn \
    python-dotenv openai requests
```

### 2. 配置环境变量
```bash
cp .env.example .env
```

编辑 `.env`：
```
# DeepSeek
DEEPSEEK_API_KEY=your_key

# Intervals.icu
INTERVALS_ATHLETE_ID=your_athlete_id
INTERVALS_API_KEY=your_key

# Strava（需要先完成 OAuth 授权，见下方）
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
STRAVA_ACCESS_TOKEN=your_access_token
STRAVA_REFRESH_TOKEN=your_refresh_token
STRAVA_EXPIRES_AT=your_expires_at
```

### 3. Strava OAuth 授权

```bash
# 第一步：浏览器打开授权链接（替换 YOUR_CLIENT_ID）
# https://www.strava.com/oauth/authorize?client_id=YOUR_CLIENT_ID&response_type=code&redirect_uri=http://localhost/callback&scope=activity:read_all

# 第二步：复制回调 URL 里的 code，换取 token
curl -X POST https://www.strava.com/oauth/token \
  -d client_id=YOUR_CLIENT_ID \
  -d client_secret=YOUR_CLIENT_SECRET \
  -d code=YOUR_CODE \
  -d grant_type=authorization_code

# 将返回的 access_token / refresh_token / expires_at 填入 .env
```

### 4. 启动服务
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 5. 使用前端
浏览器打开 `index.html`，或访问 `http://localhost:8000/docs` 查看交互式 API 文档。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
| POST | `/ask` | 训练问答 |
| POST | `/plan/generate` | 生成结构化训练计划（JSON）|
| POST | `/plan/create` | 写入训练计划到 Intervals 日历 |

## 示例

**问答：**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "我今天状态怎么样，适合高强度训练吗？"}'
```

回答示例：
> CTL 61，ATL 52，TSB +9，处于精力充沛区。近14天总训练时长 16.7h，3月28日单次 NP 194W 接近 FTP，强度偏高。今天状态较好，可以安排一次阈值间歇，建议 4x8min @ 95-105% FTP。

**生成并写入训练计划：**
```bash
# 生成计划
curl -X POST http://localhost:8000/plan/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "帮我制定本周训练计划，目标提升FTP"}'

# 写入 Intervals 日历
curl -X POST http://localhost:8000/plan/create \
  -H "Content-Type: application/json" \
  -d '{"events": [{"date": "2026-04-06", "name": "Z2有氧耐力", "description": "130-150W骑行90分钟", "load_target": 80}]}'
```
