"""
测试两种 prompt 结构对 KV Cache 命中率的影响

旧版：user_profile 和 context 放在 system 里（每次都变）
新版：system 只有固定内容，user_profile 和 context 放在 human 里
"""
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ── 模拟真实数据 ──────────────────────────────────────────────
USER_PROFILE = """## 基本信息
- 姓名：Minghe，女，23岁，体重63kg
- FTP：202W，最大心率：194bpm

## 训练目标
- 截止：2026年6月15日
- 目标FTP：220W（差18W），目标体重：60kg（差3kg）

## 当前训练状态
- 健康度（CTL）：61
- 疲劳度（ATL）：52
- 状态值（TSB）：+9
- 变化率：-1.2

## 近期训练
| 日期 | 名称 | 时长 | 均功率 | 均心率 |
|------|------|------|--------|--------|
| 4/20 | Evening Ride | 70min | 108W | 138bpm |
| 4/18 | Morning Ride | 85min | 158W | 163bpm |
| 4/15 | Afternoon Ride | 100min | 163W | 161bpm |
| 4/12 | Evening Ride | 89min | 93W | 127bpm |
"""

KNOWLEDGE_CONTEXT = """## 功率训练区间
Z1 主动恢复：<55% FTP（<111W）
Z2 有氧耐力：56-75% FTP（113-152W）
Z3 节奏骑行：76-90% FTP（154-182W）
Z4 乳酸阈值：91-105% FTP（184-212W）
Z5 最大摄氧量：106-120% FTP（214-242W）

---

## TSS 参考值
- <150：恢复快，次日可正常训练
- 150-300：中等疲劳，需1-2天恢复
- >450：极高，需要数天充分恢复

---

## 恢复理论
TSB 正值说明身体处于精力充沛状态，适合高质量训练。
CTL 代表长期训练健康度，ATL 代表短期疲劳度。"""

QUESTIONS = [
    "我今天适合高强度训练吗？",
    "Z2训练应该控制在什么功率？",
    "我距离目标FTP还差多少？",
]

# ── 旧版 prompt（user_profile 在 system 里）────────────────────
def test_old_prompt():
    print("=" * 60)
    print("旧版 prompt：user_profile + context 在 system 里")
    print("=" * 60)

    for i, q in enumerate(QUESTIONS):
        # 旧版：system 每次包含 user_profile 和 context（模拟每次都变）
        system = f"""你是一个专业的公路骑行教练，风格简练直接。

## 用户档案（每次必读）
{USER_PROFILE}

## 相关知识参考
{KNOWLEDGE_CONTEXT}

## 回答要求
- 直接给结论，不要重复用户说的内容
- 引用具体数据支撑判断（如 IF、TSS、FTP占比）
- 控制在200字以内
- 不需要总结段
"""
        start = time.time()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": q}
            ]
        )
        elapsed = time.time() - start
        usage = response.usage

        print(f"\n请求 {i+1}：{q}")
        print(f"  耗时：{elapsed:.2f}s")
        print(f"  prompt_tokens：{usage.prompt_tokens}")
        print(f"  completion_tokens：{usage.completion_tokens}")
        # DeepSeek 返回的缓存命中信息
        if hasattr(usage, 'prompt_cache_hit_tokens'):
            print(f"  prompt_cache_hit_tokens：{usage.prompt_cache_hit_tokens}")
            print(f"  prompt_cache_miss_tokens：{usage.prompt_cache_miss_tokens}")
        print(f"  完整usage：{usage}")

# ── 新版 prompt（system 只有固定内容）──────────────────────────
def test_new_prompt():
    print("\n" + "=" * 60)
    print("新版 prompt：system 只有固定内容，user_profile 在 human 里")
    print("=" * 60)

    # 新版：system 永远不变
    system = """你是一个专业的公路骑行教练，风格简练直接。

## 回答要求
- 直接给结论，不要重复用户说的内容
- 引用具体数据支撑判断（如 IF、TSS、FTP占比）
- 控制在200字以内
- 不需要总结段
- 不要使用 ** 加粗符号，直接输出纯文本
"""

    for i, q in enumerate(QUESTIONS):
        # 新版：变化的内容放在 human 里
        human = f"""## 我的训练档案
{USER_PROFILE}

## 相关知识参考
{KNOWLEDGE_CONTEXT}

## 我的问题
{q}"""

        start = time.time()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": human}
            ]
        )
        elapsed = time.time() - start
        usage = response.usage

        print(f"\n请求 {i+1}：{q}")
        print(f"  耗时：{elapsed:.2f}s")
        print(f"  prompt_tokens：{usage.prompt_tokens}")
        print(f"  completion_tokens：{usage.completion_tokens}")
        if hasattr(usage, 'prompt_cache_hit_tokens'):
            print(f"  prompt_cache_hit_tokens：{usage.prompt_cache_hit_tokens}")
            print(f"  prompt_cache_miss_tokens：{usage.prompt_cache_miss_tokens}")
        print(f"  完整usage：{usage}")


if __name__ == "__main__":
    test_old_prompt()
    test_new_prompt()

    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print("旧版：system 每次包含 user_profile，前缀每次都变")
    print("      → KV Cache 命中率应接近零")
    print("新版：system 固定不变，只有回答要求")
    print("      → 第2、3次请求应有 prompt_cache_hit_tokens > 0")
