# 开发日志

记录每次改动、方法选择依据、踩坑和修复。

---

## 2026-06-19（续5）— 修复：时区（UTC → Asia/Shanghai）

### 修复
- **agent.py**：system prompt 日期从 `datetime.now()`（服务器 UTC）改为 `_today_str()` 读用户时区
- **tools.py**：新增 `_get_user_timezone()` 缓存函数，一次 API 调用后内存缓存，避免 agent_node 每次调 API
- **intervals_client.py**：新增 `get_timezone()` 方法 + context 加入时区行
- **数据来源**：Intervals API 返回 `timezone: "Asia/Shanghai"`，直接读取，无需配置
- **结果**：用户时区 Asia/Shanghai，50/50 通过

---

## 2026-06-19（续4）— 修复：API 数据字段路径 + 去硬编码 + 计划日期

### 修复清单
- **Intervals API 读取 FTP/体重/LTHR**：原代码读 `athlete["ftp"]`（顶层永远 null），改为读 `sportSettings[0].ftp`（200W）、`icu_weight`（63kg）、`sportSettings[0].lthr`（182bpm）
- **移除 `_fetch_context()` 硬编码用户信息**：API 数据正确后不再需要写死的姓名/FTP/体重/目标
- **计划日期写死**：`tools.py:242` 示例 `"2026-05-05"` 改为 `datetime.now()` 动态日期
- **结果**：FTP 从硬编码 202W 改为 API 实际值 200W，50/50 测试通过

---

## 2026-06-19（续3）— 测试扩展：参数校验

### 新增测试文件
- `tests/test_tools.py`（10 条）：search_knowledge 空/空白/单字符/正常/不存在；analyze_and_plan 空/模糊/具体；modify_plan 无计划报错；ask_user 参数 schema
- **依据**：FutureAGI "Argument Extraction" + BFCL 参数正确性
- **结果**：50/50 全部通过（217s）

---

## 2026-06-19（续2）— 测试扩展：错误恢复 + 幻觉检测

### 新增测试文件
- `tests/test_error_recovery.py`（6 条）：三层降级链单元测试（实时→缓存→静态档案）、Agent 不因工具失败而 500、数据新鲜度检查
- `tests/test_hallucination.py`（6 条）：不编造 FTP/功率数值、虚构概念拒绝、使用工具数据而非模型知识
- **依据**：FutureAGI "Error Recovery" + "Result Utilization" 维度
- **踩坑**：LangGraph 工具在独立线程执行，`unittest.mock.patch` 模块级函数不起作用；改为直接调用 `_fetch_context()` 做单元测试
- **结果**：40/40 全部通过（188s）

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
