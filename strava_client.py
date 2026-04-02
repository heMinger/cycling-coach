import requests
import time
from dotenv import load_dotenv, set_key
from datetime import datetime, timedelta
import os

load_dotenv()

class StravaClient:
    """
    Strava API 客户端，支持 access_token 自动刷新
    
    Token 机制：
    - access_token：有效期 6 小时，用于 API 调用
    - refresh_token：长期有效，用于刷新 access_token
    - expires_at：Unix 时间戳，access_token 过期时间
    """

    BASE_URL = "https://www.strava.com/api/v3"

    def __init__(self):
        self.client_id = os.getenv("STRAVA_CLIENT_ID")
        self.client_secret = os.getenv("STRAVA_CLIENT_SECRET")
        self.access_token = os.getenv("STRAVA_ACCESS_TOKEN")
        self.refresh_token = os.getenv("STRAVA_REFRESH_TOKEN")
        self.expires_at = int(os.getenv("STRAVA_EXPIRES_AT", "0"))

    def _refresh_token_if_needed(self):
        """
        检查 token 是否过期，过期则自动刷新
        提前 5 分钟刷新，避免临界情况
        """
        if time.time() > self.expires_at - 300:
            print("access_token 即将过期，正在刷新...")
            response = requests.post(
                "https://www.strava.com/oauth/token",
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token"
                }
            )
            response.raise_for_status()
            data = response.json()

            # 更新内存中的 token
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.expires_at = data["expires_at"]

            # 写回 .env 文件，下次启动不需要重新授权
            set_key(".env", "STRAVA_ACCESS_TOKEN", self.access_token)
            set_key(".env", "STRAVA_REFRESH_TOKEN", self.refresh_token)
            set_key(".env", "STRAVA_EXPIRES_AT", str(self.expires_at))
            print("token 刷新成功")

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """GET 请求封装，自动带上 token"""
        self._refresh_token_if_needed()
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.get(
            f"{self.BASE_URL}{endpoint}",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

    # ── 读取：运动员信息 ──────────────────────────────────────

    def get_athlete(self) -> dict:
        """获取运动员基本信息"""
        return self._get("/athlete")

    # ── 读取：训练活动 ────────────────────────────────────────

    def get_activities(self, days: int = 14, per_page: int = 30) -> list:
        """
        获取最近 n 天的训练活动列表
        
        返回字段包括：
        - id, name, type
        - start_date_local
        - moving_time（秒）
        - distance（米）
        - average_watts, weighted_average_watts（NP）
        - average_heartrate, max_heartrate
        - suffer_score（Strava 的训练压力分，类似 TSS）
        - total_elevation_gain
        """
        after = int((datetime.now() - timedelta(days=days)).timestamp())
        return self._get("/athlete/activities", params={
            "after": after,
            "per_page": per_page
        })

    def get_activity_detail(self, activity_id: int) -> dict:
        """
        获取单个活动的详细数据
        包括逐圈数据（laps）、区间数据（segments）等
        """
        return self._get(f"/activities/{activity_id}")

    def get_activity_zones(self, activity_id: int) -> dict:
        """
        获取活动的心率/功率区间分布
        需要 Strava Summit（付费）才能获取功率区间
        """
        return self._get(f"/activities/{activity_id}/zones")

    # ── 数据格式化：供 RAG 使用 ───────────────────────────────

    def build_activity_context(self, days: int = 14) -> str:
        """
        整合近期训练数据，构建供 RAG 注入的上下文字符串
        """
        activities = self.get_activities(days=days)

        if not activities:
            return "## 近期训练记录\n暂无数据\n"

        lines = []
        total_tss = 0
        total_time = 0

        for a in activities:
            date = a.get("start_date_local", "")[:10]
            name = a.get("name", "未命名")
            activity_type = a.get("type", "")

            # 只处理骑行类活动
            if activity_type not in ["Ride", "VirtualRide", "GravelRide"]:
                continue

            duration_min = round(a.get("moving_time", 0) / 60)
            distance_km = round(a.get("distance", 0) / 1000, 1)
            avg_power = a.get("average_watts", "-")
            np = a.get("weighted_average_watts", "-")
            avg_hr = a.get("average_heartrate", "-")
            suffer = a.get("suffer_score", "-")
            elevation = a.get("total_elevation_gain", "-")

            if isinstance(suffer, (int, float)):
                total_tss += suffer
            if isinstance(a.get("moving_time"), (int, float)):
                total_time += a.get("moving_time", 0)

            lines.append(
                f"| {date} | {name} | {duration_min}min | "
                f"{distance_km}km | {avg_power}W | NP {np}W | "
                f"{avg_hr}bpm | 压力分 {suffer} | 爬升 {elevation}m |"
            )

        total_hours = round(total_time / 3600, 1)

        context = f"""## 近{days}天骑行训练记录

| 日期 | 名称 | 时长 | 距离 | 均功率 | NP | 均心率 | 压力分 | 爬升 |
|------|------|------|------|--------|-----|--------|--------|------|
{chr(10).join(lines) if lines else "| 暂无骑行数据 |"}

**汇总**：总训练时长 {total_hours}h，累计压力分 {total_tss}
"""
        return context


# ── 测试 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    client = StravaClient()

    print("=== 运动员信息 ===")
    athlete = client.get_athlete()
    print(f"姓名：{athlete.get('firstname')} {athlete.get('lastname')}")
    print(f"城市：{athlete.get('city')}")

    print("\n=== 近14天骑行训练 ===")
    activities = client.get_activities(days=14)
    ride_count = 0
    for a in activities:
        if a.get("type") in ["Ride", "VirtualRide", "GravelRide"]:
            ride_count += 1
            date = a.get("start_date_local", "")[:10]
            name = a.get("name", "未命名")
            duration = round(a.get("moving_time", 0) / 60)
            power = a.get("average_watts", "-")
            hr = a.get("average_heartrate", "-")
            print(f"{date} | {name} | {duration}min | {power}W | {hr}bpm")

    print(f"\n共 {ride_count} 次骑行")

    print("\n=== RAG 上下文预览 ===")
    context = client.build_activity_context(days=14)
    print(context)
