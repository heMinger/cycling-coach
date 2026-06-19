# 二期开发计划（2026-06-18−）

## 目标

补全网页聊天应用的基础功能，扩展知识库来源，提升训练分析的准确性和用户体验。

---

## 需求清单

### 1. 基础聊天功能（对标 ChatGPT 等网页版）

| # | 功能 | 优先级 | 说明 |
|---|------|--------|------|
| 1.1 | 日期感知 | ✅ 已修复 | System prompt 注入当前日期 |
| 1.2 | 刷新保留对话 | 高 | 刷新后消息列表全空，需加载历史；后端加 `GET /messages?session_id=xxx`，前端 onload 拉取 |
| 1.3 | 流式输出（SSE） | 中 | 当前一次性返回，长回答等待体验差 |
| 1.4 | Markdown 渲染 | 中 | 当前 textContent 纯文本，被迫在 prompt 里禁止 markdown |
| 1.5 | 多会话管理 | 中 | 新建对话 / 切换 / 回看历史会话列表 |
| 1.6 | 停止生成 | 低 | 无法中断进行中的回答 |
| 1.7 | 复制消息 | 低 | 单条消息一键复制 |

### 2. 训练分析准确性

| # | 问题 | 优先级 | 修复方向 |
|---|------|--------|----------|
| 2.1 | 时区 | 高 | 服务器 UTC，用户 UTC+8；"今天"定义差 8 小时，影响 CTL/TSB 判断。方案：前端传时区或配置用户时区，注入 prompt |
| 2.2 | 计划日期错误 | 高 | `tools.py:242` 示例日期写死 `2026-05-05`，LLM 可能照抄。改为当天日期 |
| 2.3 | 用户档案硬编码 | 高 | `tools.py:33-37` 写死姓名/FTP/体重/目标日期，换人无法用。改为可编辑配置 |
| 2.4 | Intervals API 不稳定 | 高 | 海外服务器，国内直连 HTTPS 不稳定（SSL EOF 错误），导致静默降级到 45 天前缓存。方案：① 代理（`HTTPS_PROXY` 或 `requests` proxies）；② 重试机制；③ 部分降级（Strava 成功则保留 Strava 数据） |
| 2.5 | 过期数据无提示 | 中 | API 失败降级到缓存时，用户毫不知情，看到 45 天前数据以为是最新的。方案：标注数据来源和更新时间，询问用户是否继续使用旧数据 |

### 3. 外部内容源接入

#### 3.1 学术期刊检索
- 数据源：《Science》《Journal of Applied Physiology》《Medicine & Science in Sports & Exercise》等运动科学期刊
- 内容形式：最新论文标题 + 摘要，按主题（功率训练、恢复生理、营养补给等）分类入库
- 更新策略：定期抓取，追踪特定关键词的新发表
- 技术方案待定：PubMed API、Semantic Scholar API、或 RSS feed

#### 3.2 博主/UP主视频内容
- 数据源：YouTube、B站 等平台的骑行教练/运动科学频道
- 内容形式：视频字幕转录 → 分段摘要 → 向量化入 Chroma
- 更新策略：订阅频道，新发布时自动转录入库
- 技术方案待定：youtube-dl/yt-dlp 字幕下载、Whisper 转录、LLM 摘要

### 4. 知识库扩展

- 当前 `data/knowledge/` 仅 3 个 markdown 文件
- 需扩展：营养补给、力量训练、心理准备、比赛策略、伤病预防等主题
- 同时支持"静态编写 + 动态抓取"双轨内容

### 5. 检索增强

- 当前 `search_knowledge` 仅查 Chroma 向量库
- 需扩展为：本地知识库 + 外部实时检索（如 PubMed、Semantic Scholar）
- 考虑增加 `search_external` 工具，与 `search_knowledge` 并列

---

## 待决策

- [ ] 学术论文检索选型（PubMed vs Semantic Scholar vs 其他）
- [ ] 视频平台优先级（YouTube vs B站 vs 两者都做）
- [ ] 外源内容入库频率（实时 vs 每日 vs 每周）
- [ ] 是否需要人工审核入库内容
- [ ] 用户档案存储方式（前端设置页 vs 配置文件 vs 数据库）
- [ ] 时区方案（前端自动检测 vs 用户手动配置）

---

## 相关文件

- `data/knowledge/` — 现有知识库
- `tools.py` → `search_knowledge`、`_fetch_context`、`analyze_and_plan` — 需调整
- `agent.py` → `agent_node` — system prompt 需加时区
- `api.py` — 需加历史消息接口、会话列表接口
- `index.html` — 前端需大幅增强
- `chroma_db/` — 向量数据库
