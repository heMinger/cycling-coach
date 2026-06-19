# 开发日志

记录每次改动、方法选择依据、踩坑和修复。

---

## 2026-06-19（续）— 测试扩展：边界拒绝 + 多轮一致性

### 新增测试文件
- `tests/test_boundaries.py`（6 条）：越界拒绝（修车/股票/用药）、身份感知
- `tests/test_consistency.py`（3 条）：追问记住上下文、状态到计划连续、计划修改识别
- **依据**：Google Cloud RAG checklist "Domain Boundaries" + FutureAGI "Plan Coherence" 维度
- **结果**：27/27 全部通过（118s）

---

## 2026-06-19 — 二期开发启动：目录整理、基础功能修复、测试框架

### 目录结构重组
- 新建 `specs/`：`agent-v2-plan.md` 从根目录迁入，新增 `phase2-plan.md`
- 新建 `tests/`：conftest + test_api + test_basics + test_regression（18/18 pass）
- 新建 `CLAUDE.md`：项目上下文、编码规则、已知踩坑
- 新建 `specs/dev-log.md`：本文件
- `test_kvcache.py` 从根目录迁入 `eval/`
- `INTERVIEW_PREP.md` 从根目录迁入 `specs/interview-prep.md`

### Bug 修复：Agent 不知道当前日期
- **问题**：问"今天是哪一天"返回 2025年5月18日，导致状态分析日期错位
- **根因**：DeepSeek API 是裸模型，不自动注入日期；system prompt 未提供
- **依据**：Claude Code、ChatGPT 网页版、DeepSeek 网页版之所以知道日期，是因为**产品层注入了，不是模型自己知道**。所有 LLM API（OpenAI、Anthropic、DeepSeek）都是裸模型
- **修复**：`agent.py:110` system prompt 加 `今天是 {datetime.now().strftime("%Y年%m月%d日")}`
- **关联**：`specs/phase2-plan.md` 1.1

### Bug 修复：Intervals API SSL 错误导致训练数据降级到 45 天前缓存
- **问题**：问"我今天状态怎么样"返回 5月5日数据，实际最新是 6月18日
- **根因**：服务器有 `https_proxy=http://127.0.0.1:7890`，代理到 intervals.icu 的 SSL 握手失败（`SSL: UNEXPECTED_EOF_WHILE_READING`），`_fetch_context()` 抛异常 → 静默降级到 `cache/user_profile_cache.md`（5月5日写入）
- **诊断方法**：直连 vs 代理对比测试 — `curl https://intervals.icu` 直连成功，走代理失败
- **修复**：`intervals_client.py` 四个方法（`_get`/`_post`/`_put`/`_delete`）加 `proxies={"http": None, "https": None}` 绕过代理直连
- **关联**：`specs/phase2-plan.md` 2.4, 2.5

### 功能规划：二期需求清单
- **来源**：行业调研（FutureAGI 6 维度评估、Google Cloud RAG checklist、Thoughtworks 三层框架）
- **收录**：基础聊天功能 7 项、训练分析准确性 5 项、外部内容源接入 2 项、知识库扩展、检索增强
- **文件**：`specs/phase2-plan.md`

### 测试框架搭建
- **依据**：行业最佳实践 — 轨迹评估优先于输出评估、确定性优先于 LLM-judge、回归测试自动化
- **实现**：`tests/conftest.py`（fixtures + .env 加载）、`tests/test_api.py`（5 条）、`tests/test_basics.py`（6 条）、`tests/test_regression.py`（3 条）
- **结果**：18/18 通过（75s）

---

## 2026-05-09 — 四层评估体系建立

- 建立 Layer 1-5 评估框架（切片 → 检索 → 生成 → Agent 行为）
- Layer 3 检索 precision=0.578，不达标（阈值 0.6），根因是 chunk_03 hub 效应
- 通过阈值扫描确定 L2 距离 0.85 为最优过滤点（precision 0.65, recall 1.0）
- **文件**：`eval/EVAL_REPORT.md`

---

## 2026-05-05 — Agent v2 升级

- 从 RAG + 固定路由 workflow 升级为 LangGraph Agent
- 设计决策：无显式路由节点（LLM 自主选工具）、自定义 ToolNode 拦截返回值、三层安全
- 原始 9 个设计问题 + 3 个新发现 bug，全部修复
- **文件**：`specs/agent-v2-plan.md`

---

## 格式约定

每条记录包含：
- **日期** — 改动发生日
- **改动内容** — 做了什么
- **依据/参考** — 为什么这样做（行业标准、文档引用、诊断过程）
- **踩坑** — 遇到的问题和修复方法
