"""
memory.py - 记忆系统

负责：
1. SQLite 建表和读写
2. 短期会话历史（内存，最近5轮）
3. AutoMemory 提取和更新（长期记忆）
4. 关键词触发 + 每5轮批量更新
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "memory.db"
MAX_SHORT_TERM = 5       # 短期记忆保留轮数
BATCH_TRIGGER = 5        # 每N轮触发一次批量AutoMemory更新
CONVERSATION_EXPIRE_DAYS = 30  # 原文保留天数

# 触发立刻更新的关键词
TRIGGER_KEYWORDS = [
    "膝盖", "受伤", "痛", "不舒服", "受凉", "拉伤", "扭伤",
    "比赛", "赛事", "目标",
    "喜欢", "不喜欢", "讨厌", "偏好", "习惯"
]

# ── 建表 ──────────────────────────────────────────────────────
def init_db():
    """初始化数据库，创建表（幂等，可重复调用）"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 完整对话原文（30天）
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        intent TEXT DEFAULT 'general',
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        extracted INTEGER DEFAULT 0
    )
    """)

    # AutoMemory 长期记忆（永久）
    c.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        source_session TEXT
    )
    """)

    # 会话元数据（记录轮数，用于批量触发）
    c.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        turn_count INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_active DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
    print("数据库初始化完成")

# ── 会话管理 ──────────────────────────────────────────────────
def get_or_create_session(session_id: str):
    """获取或创建会话记录"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT turn_count FROM sessions WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    if not row:
        c.execute("INSERT INTO sessions (session_id) VALUES (?)", (session_id,))
        conn.commit()
        turn_count = 0
    else:
        turn_count = row[0]
    conn.close()
    return turn_count

def increment_turn(session_id: str) -> int:
    """增加会话轮数，返回新轮数"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE sessions 
        SET turn_count = turn_count + 1, last_active = CURRENT_TIMESTAMP
        WHERE session_id = ?
    """, (session_id,))
    c.execute("SELECT turn_count FROM sessions WHERE session_id = ?", (session_id,))
    turn_count = c.fetchone()[0]
    conn.commit()
    conn.close()
    return turn_count

# ── 对话原文存取 ──────────────────────────────────────────────
def save_message(session_id: str, role: str, content: str, intent: str = "general"):
    """保存一条对话消息"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO conversations (session_id, role, content, intent)
        VALUES (?, ?, ?, ?)
    """, (session_id, role, content, intent))
    conn.commit()
    conn.close()

def get_recent_messages(session_id: str, n_turns: int = MAX_SHORT_TERM) -> list:
    """
    获取最近N轮对话，用于注入短期记忆。
    返回格式：[{"role": "user", "content": "..."}, ...]
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 取最近 n_turns*2 条（每轮2条：user+assistant）
    c.execute("""
        SELECT role, content FROM conversations
        WHERE session_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (session_id, n_turns * 2))
    rows = c.fetchall()
    conn.close()
    # 反转顺序（从最早到最新）
    rows.reverse()
    return [{"role": row[0], "content": row[1]} for row in rows]

def get_unextracted_messages(session_id: str) -> list:
    """获取未被AutoMemory提取过的消息"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT role, content FROM conversations
        WHERE session_id = ? AND extracted = 0
        ORDER BY timestamp ASC
    """, (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def mark_messages_extracted(session_id: str):
    """标记该会话的消息已被提取"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE conversations SET extracted = 1
        WHERE session_id = ? AND extracted = 0
    """, (session_id,))
    conn.commit()
    conn.close()

def clean_old_conversations():
    """删除30天前的对话原文"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    expire_date = (datetime.now() - timedelta(days=CONVERSATION_EXPIRE_DAYS)).isoformat()
    c.execute("DELETE FROM conversations WHERE timestamp < ?", (expire_date,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        print(f"清理过期对话：{deleted} 条")

# ── AutoMemory 长期记忆 ───────────────────────────────────────
def get_all_memories() -> str:
    """
    获取所有长期记忆，格式化为可注入 prompt 的字符串。
    按类别分组展示。
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT category, content, updated_at FROM memories
        ORDER BY category, updated_at DESC
    """)
    rows = c.fetchall()
    conn.close()

    if not rows:
        return ""

    # 按类别分组
    categories = {}
    for category, content, updated_at in rows:
        if category not in categories:
            categories[category] = []
        categories[category].append(content)

    # 格式化
    category_names = {
        "injury": "伤病/身体状况",
        "preference": "训练偏好",
        "goal": "目标赛事",
        "schedule": "固定日程"
    }

    lines = []
    for cat, contents in categories.items():
        name = category_names.get(cat, cat)
        lines.append(f"【{name}】")
        for content in contents:
            lines.append(f"  - {content}")

    return "\n".join(lines)

def save_memory(category: str, content: str, source_session: str = None):
    """保存或更新一条长期记忆"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 检查是否已有相同类别的相似记忆（简单去重）
    c.execute("""
        INSERT INTO memories (category, content, source_session)
        VALUES (?, ?, ?)
    """, (category, content, source_session))
    conn.commit()
    conn.close()

def update_memory(memory_id: int, content: str):
    """更新一条已有的记忆"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE memories SET content = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (content, memory_id))
    conn.commit()
    conn.close()

# ── AutoMemory 提取 ───────────────────────────────────────────
def extract_memories(messages: list, session_id: str):
    """
    用 LLM 从对话里提取值得长期记忆的信息。
    触发条件：关键词检测 或 每5轮批量
    """
    if not messages:
        return

    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    # 把消息格式化成对话文本
    conversation_text = "\n".join([
        f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}"
        for m in messages
    ])

    # 获取现有记忆，用于去重和更新
    existing_memories = get_all_memories()

    prompt = f"""分析下面的对话，提取值得长期记忆的用户信息。

现有记忆：
{existing_memories if existing_memories else "（暂无）"}

对话内容：
{conversation_text}

提取规则：
- 只提取真正重要的、持久的信息
- 如果和现有记忆重复或矛盾，说明需要更新
- 如果没有值得提取的信息，返回空数组

类别说明：
- injury：受伤、疼痛、身体不适（例：左膝盖下坡时疼痛）
- preference：训练偏好（例：喜欢爬坡路线，不喜欢室内训练）
- goal：目标赛事或数据目标（例：6月15日参加绕圈赛）
- schedule：固定不能训练的时间（例：每周三不能训练）

请严格按照以下 JSON 格式输出，不要有任何其他内容：
{{
  "extracted": [
    {{
      "category": "injury/preference/goal/schedule",
      "content": "简洁描述，一句话",
      "action": "add/update/ignore",
      "reason": "为什么提取这条"
    }}
  ]
}}

如果没有值得提取的信息：{{"extracted": []}}
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        raw = response.choices[0].message.content.strip()

        # 清理 markdown
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        extracted = result.get("extracted", [])

        for item in extracted:
            if item.get("action") in ["add", "update"]:
                save_memory(
                    category=item["category"],
                    content=item["content"],
                    source_session=session_id
                )
                print(f"AutoMemory 新增：[{item['category']}] {item['content']}")

        # 标记消息已提取
        mark_messages_extracted(session_id)

    except Exception as e:
        print(f"AutoMemory 提取失败：{e}")

# ── 触发逻辑 ──────────────────────────────────────────────────
def should_trigger_immediate(user_message: str) -> bool:
    """检测是否包含关键词，需要立刻触发AutoMemory更新"""
    return any(kw in user_message for kw in TRIGGER_KEYWORDS)

def process_memory_update(session_id: str, user_message: str, turn_count: int):
    """
    决定是否触发AutoMemory更新。
    触发条件：关键词命中 或 每5轮
    """
    should_update = False

    if should_trigger_immediate(user_message):
        print(f"关键词触发AutoMemory更新")
        should_update = True
    elif turn_count % BATCH_TRIGGER == 0:
        print(f"第{turn_count}轮，批量触发AutoMemory更新")
        should_update = True

    if should_update:
        messages = get_unextracted_messages(session_id)
        if messages:
            extract_memories(messages, session_id)

# ── 格式化短期历史（注入 prompt 用）─────────────────────────
def format_short_term_history(messages: list) -> str:
    """把最近几轮对话格式化为可注入 prompt 的字符串"""
    if not messages:
        return ""

    lines = ["## 最近对话记录"]
    for msg in messages:
        role = "用户" if msg["role"] == "user" else "AI"
        lines.append(f"{role}：{msg['content']}")

    return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    init_db()

    # 模拟一次对话
    session_id = "test-session-001"
    get_or_create_session(session_id)

    save_message(session_id, "user", "我最近左膝盖有点疼，骑车下坡时明显", "general")
    save_message(session_id, "assistant", "建议减少下坡骑行，注意热身", "general")

    turn = increment_turn(session_id)
    process_memory_update(session_id, "我最近左膝盖有点疼", turn)

    print("\n当前长期记忆：")
    print(get_all_memories())

    print("\n最近对话：")
    recent = get_recent_messages(session_id)
    print(format_short_term_history(recent))
