# CLAUDE.md — cycling-coach 项目开发规则

## 项目

公路骑行 AI 教练。LangGraph Agent + RAG + DeepSeek，对接 Strava / Intervals.icu 双平台 API，通过 Web UI 对话制定个性化训练计划。

### 技术栈
- LLM: DeepSeek V3 (`deepseek-chat` via `api.deepseek.com`)
- Agent: LangGraph（reducer + interrupt + checkpoint）
- RAG: Chroma + BAAI/bge-small-zh-v1.5
- API: FastAPI + Uvicorn
- 前端: 单页 HTML（`index.html`）
- 数据: Strava API（OAuth 2.0）+ Intervals.icu API（Basic Auth）

## 常用命令

```bash
# 启动服务（8000 端口）
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cycling-coach
uvicorn api:app --host 0.0.0.0 --port 8000 &

# 运行测试
python -m pytest tests/ -v

# 安装依赖（PyPI 直连，腾讯云镜像缺少部分包）
pip install <package> -i https://pypi.org/simple/
```

## 架构决策（不可修改，出自 specs/agent-v2-plan.md）

1. **框架：LangGraph** — interrupt() 原生支持 human-in-the-loop
2. **路由：无显式路由节点** — LLM 通过 tool description 自主选择工具
3. **AgentState 极简** — 只放 messages 解决不了的字段
4. **工具数量：6 个**（get_full_context, search_knowledge, analyze_and_plan, modify_plan, write_to_calendar, ask_user）
5. **安全：三层** — recursion_limit(10)、tool_call_count(≤8)、write 前 interrupt()

## 编码规则

- **五步开发流程（强制执行）**：① 搜索行业典范实现 → ② 留档参考方案 → ③ 对比方法制定计划，存 `specs/` → ④ 用户确认 → ⑤ 实施并更新 `specs/dev-log.md`
- **注释只写为什么**，不写是什么。如果移除注释不会让未来读者困惑，就不要写
- **不引入不必要的抽象**。三个相似行好过一个过早的工厂类
- **每次改动必须记录**到 `specs/dev-log.md`，包含：日期、改动内容、依据/参考、踩坑和修复
- **修改代码后运行测试套件**：`python -m pytest tests/ -v`
- **System prompt 必须注入当前日期**（通过 `_today_str()` 读用户时区）
- **Intervals.icu 请求必须绕过代理直连**（`proxies={"http": None, "https": None}`），服务器 7890 端口代理会打断 SSL 握手
- **普通回答控制在 150 字以内，不使用 markdown 格式**（前端 textContent 渲染）

## 已知踩坑

| # | 问题 | 原因 | 修复 | 日期 |
|---|------|------|------|------|
| 1 | Agent 回答日期错误 | System prompt 未注入日期，DeepSeek 不自动提供 | `agent.py:110` 注入 `今天是 {datetime.now()...}` | 2026-06-19 |
| 2 | 训练数据总是 5月5日 | Intervals API 走代理 SSL 失败 → 静默降级到旧缓存 | `intervals_client.py` 所有请求加 `proxies={"http": None, "https": None}` | 2026-06-19 |
| 3 | 旧 session checkpoint 存了过时 system prompt | LangGraph 持久化历史消息，新规则不生效 | 新建 session 或用 `localStorage.clear()` | 2026-06-19 |

## 项目结构

```
cycling-coach/
├── agent.py              # Agent 图 + system prompt + agent_node
├── agent_state.py        # AgentState TypedDict
├── api.py                # FastAPI: /ask, /resume, /plan/current, /memory/*
├── tools.py              # 6 个 tool + _fetch_context + 格式化
├── memory.py             # 长期记忆（SQLite）
├── rag.py                # 旧 RAG 链（被 agent 替代，tools.py 复用逻辑）
├── strava_client.py      # Strava API（OAuth 自动刷新 token）
├── intervals_client.py   # Intervals API（Basic Auth，绕过代理）
├── index.html            # 前端：聊天 + 计划编辑器 + 确认卡片
├── main.py               # 基础测试入口
│
├── tests/                # pytest 自动化测试
│   ├── conftest.py
│   ├── test_api.py       # 端点状态码、响应结构
│   ├── test_basics.py    # 日期注入、工具路由
│   └── test_regression.py # 已修复 bug 防护
│
├── eval/                 # 质量评估（4 层：切片 → 检索 → 生成 → Agent）
│
├── specs/                # 设计文档
│   ├── agent-v2-plan.md  # Agent v2 架构方案
│   ├── phase2-plan.md    # 二期功能规划
│   └── dev-log.md        # 开发日志
│
├── data/knowledge/       # 骑行知识库 markdown
├── data/user_data/       # 用户档案（降级备用）
├── cache/                # API 数据缓存
└── chroma_db/            # 向量数据库
```
