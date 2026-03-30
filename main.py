from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个专业的公路骑行教练。"},
        {"role": "user", "content": "我今天骑了2小时，平均功率180W，心率155bpm，感觉很累。请分析我的训练状态。"}
    ]
)

print(response.choices[0].message.content)
