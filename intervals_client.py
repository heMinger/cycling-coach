import requests
from base64 import b64encode
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json

load_dotenv()

class IntervalsClient:
    """
    Intervals.icu API 客户端
    文档：https://intervals.icu/api/v1/docs
    
    认证方式：HTTP Basic Auth
    username: API_KEY（固定字符串）
    password: 你的 API Key
    """

    def __init__(self, athlete_id: str = None, api_key: str = None):
        self.base_url = "https://intervals.icu/api/v1"
        self.athlete_id = athlete_id or os.getenv("INTERVALS_ATHLETE_ID")
        api_key = api_key or os.getenv("INTERVALS_API_KEY")

        # Intervals 用 Basic Auth，username 固定是 "API_KEY"
        token = b64encode(f"API_KEY:{api_key}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json"
        }

    def _get(self, endpoint: str, params: dict = None):
        """GET 请求封装"""
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, data: dict):
        """POST 请求封装"""
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

    def _put(self, endpoint: str, data: dict):
        """PUT 请求封装"""
        url = f"{self.base_url}{endpoint}"
        response = requests.put(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

    def _delete(self, endpoint: str):
        """DELETE 请求封装"""
        url = f"{self.base_url}{endpoint}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.status_code

    # ── 读取：训练数据 ────────────────────────────────────────

    def get_activities(self, days: int = 14) -> list:
        """
        拉取最近 n 天的训练活动列表
        返回字段包括：id, name, type, start_date_local,
                      moving_time, distance, average_watts,
                      average_heartrate, training_load(TSS) 等
        """
        oldest = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00")
        newest = datetime.now().strftime("%Y-%m-%dT23:59:59")

        data = self._get(
            f"/athlete/{self.athlete_id}/activities",
            params={
                "oldest": oldest,
                "newest": newest
            }
        )
        return data

    def get_activity_detail(self, activity_id: str) -> dict:
        """
        拉取单个活动的详细数据
        包括：逐秒功率、心率、配速、海拔等
        """
        return self._get(f"/activity/{activity_id}")

    def get_wellness(self, days: int = 14) -> list:
        """
        拉取每日健康数据
        包括：ctl(健康度), atl(疲劳度), tsb(状态值),
              rampRate(变化率), hrv, restingHR 等
        这就是 Intervals 侧边栏显示的那些指标
        """
        oldest = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        newest = datetime.now().strftime("%Y-%m-%d")

        return self._get(
            f"/athlete/{self.athlete_id}/wellness",
            params={
                "oldest": oldest,
                "newest": newest
            }
        )

    def get_athlete_info(self) -> dict:
        """
        拉取运动员基本信息
        包括：ftp, lthr(乳酸阈值心率), weight 等
        """
        return self._get(f"/athlete/{self.athlete_id}")

    # ── 读取：训练计划 ────────────────────────────────────────

    def get_events(self, days_ahead: int = 14, days_back: int = 0) -> list:
        """
        拉取日历上的训练计划（events）
        days_ahead：往后看几天
        days_back：往前看几天
        """
        oldest = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        newest = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        return self._get(
            f"/athlete/{self.athlete_id}/events",
            params={
                "oldest": oldest,
                "newest": newest
            }
        )

    # ── 写入：训练计划 ────────────────────────────────────────

    def create_event(self,
                     date: str,
                     name: str,
                     description: str,
                     load_target: int = None,
                     workout_doc: dict = None) -> dict:
        """
        在日历上创建一个训练计划
        
        参数：
            date: 日期，格式 "2026-04-01"
            name: 计划名称，如 "Z2 有氧耐力 2h"
            description: 训练描述，如 "保持 Z2 区间，FTP 的 65-75%"
            load_target: 目标 TSS
            workout_doc: 结构化训练文档（可选，用于在码表上显示）
        
        返回：创建的 event 对象，包含 id
        """
        event = {
            "category": "WORKOUT",
            "type": "Ride", 
            "start_date_local": f"{date}T08:00:00",  # 默认早上8点
            "end_date_local": f"{date}T00:00:00",
            "name": name,
            "description": description,
        }

        if load_target:
            event["load_target"] = load_target

        if workout_doc:
            event["workout_doc"] = workout_doc

        return self._post(
            f"/athlete/{self.athlete_id}/events",
            data=event
        )

    def update_event(self, event_id: str, updates: dict) -> dict:
        """
        修改已有训练计划
        
        updates 可包含：name, description, load_target 等
        例：update_event("123", {"description": "根据疲劳状态调整为 Z1 恢复"})
        """
        return self._put(
            f"/athlete/{self.athlete_id}/events/{event_id}",
            data=updates
        )

    def delete_event(self, event_id: str) -> int:
        """删除训练计划，返回 HTTP 状态码（200 表示成功）"""
        return self._delete(
            f"/athlete/{self.athlete_id}/events/{event_id}"
        )

    # ── 数据格式化：供 RAG 使用 ───────────────────────────────

    def build_user_context(self, activity_days: int = 14) -> str:
        """
        整合所有数据，构建供 RAG 注入的用户上下文字符串
        替代静态的 user_profile.md
        """
        # 1. 基本信息
        athlete = self.get_athlete_info()
        ftp = athlete.get("ftp", "未知")
        weight = athlete.get("weight", "未知")
        lthr = athlete.get("lthr", "未知")

        # 2. 近期训练
        activities = self.get_activities(days=activity_days)
        activity_lines = []
        for a in activities:
            date = a.get("start_date_local", "")[:10]
            name = a.get("name", "")
            duration = round(a.get("moving_time", 0) / 60)  # 秒转分钟
            power = a.get("average_watts", "-")
            hr = a.get("average_heartrate", "-")
            tss = a.get("training_load", "-")
            activity_lines.append(
                f"| {date} | {name} | {duration}min | {power}W | {hr}bpm | TSS {tss} |"
            )

        # 3. 健康状态
        wellness = self.get_wellness(days=7)
        latest = wellness[-1] if wellness else {}
        ctl = latest.get("ctl", "-")    # 健康度
        atl = latest.get("atl", "-")    # 疲劳度
        # tsb = latest.get("tsb", "-")    # 状态值
        ramp = latest.get("rampRate", "-")

        # # 原来
        # tsb = latest.get("tsb", "-")

        # 改成
        ctl_val = latest.get("ctl", 0)
        atl_val = latest.get("atl", 0)
        tsb = round(ctl_val - atl_val, 1) if ctl_val and atl_val else "-"

        # 同时读取 eFTP
        sport_info = latest.get("sportInfo", [])
        ride_info = next((s for s in sport_info if s.get("type") == "Ride"), {})
        eftp = ride_info.get("eftp", None)
        eftp_str = f"{round(eftp)}W" if eftp else "未知"

        # 4. 本周计划
        events = self.get_events(days_ahead=7, days_back=0)
        event_lines = []
        for e in events:
            date = e.get("start_date_local", "")[:10]
            name = e.get("name", "")
            desc = e.get("description", "")
            load = e.get("load_target", "-")
            event_lines.append(f"| {date} | {name} | 目标TSS {load} | {desc} |")

        # 5. 拼成 Markdown 字符串
        context = f"""## 运动员基本信息
- FTP：{ftp}W
- 体重：{weight}kg
- 乳酸阈值心率（LTHR）：{lthr}bpm

## 当前训练状态（最新）
- 健康度（CTL）：{ctl}
- 疲劳度（ATL）：{atl}
- 状态值（TSB）：{tsb}
- 变化率：{ramp}

## 近{activity_days}天训练记录
| 日期 | 名称 | 时长 | 均功率 | 均心率 | TSS |
|------|------|------|--------|--------|-----|
{chr(10).join(activity_lines) if activity_lines else "| 暂无数据 |"}

## 本周训练计划
| 日期 | 名称 | 目标TSS | 描述 |
|------|------|---------|------|
{chr(10).join(event_lines) if event_lines else "| 暂无计划 |"}
"""
        return context


# ── 测试 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    client = IntervalsClient()

    print("=== 运动员信息 ===")
    athlete = client.get_athlete_info()
    print(f"FTP: {athlete.get('ftp')}W")
    print(f"体重: {athlete.get('weight')}kg")

    print("\n=== 近7天训练 ===")
    activities = client.get_activities(days=7)
    for a in activities:
        print(f"{a.get('start_date_local', '')[:10]} | "
              f"{a.get('name', '')} | "
              f"TSS {a.get('training_load', '-')}")

    print("\n=== 当前状态 ===")
    wellness = client.get_wellness(days=3)
    if wellness:
        latest = wellness[-1]
        print(f"CTL: {latest.get('ctl')} | "
              f"ATL: {latest.get('atl')} | "
              f"TSB: {latest.get('tsb')}")

    print("\n=== 用户上下文（供RAG使用）===")
    context = client.build_user_context()
    print(context)
