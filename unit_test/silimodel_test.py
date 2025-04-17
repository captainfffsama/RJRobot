import requests
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(top_level_dir)

from rjrobot.constants import SUBTASK_SYS_PROMPT 
from rjrobot.utils import  img_to_base64
url = "https://api.siliconflow.cn/v1/chat/completions"

b64=img_to_base64(r"D:\tmp\temp\123.jpg",contain_header=True)
# print(b64)

payload = {
    "model": "deepseek-ai/deepseek-vl2",
    "stream": False,
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "stop": [],
    "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SUBTASK_SYS_PROMPT 
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "image_url": {
                        "detail": "auto",
                        "url": b64
                    },
                    "type": "image_url",
                },
                {
                    "text": "挂上绝缘子",
                    "type": "text"
                }
            ]
        }
    ]
}
headers = {
    "Authorization": f"Bearer {os.environ["SILICONFLOW_API"]}",
    "Content-Type": "application/json"
}

def parser_response_json(response: str):
    import re
    import json
    
    # 使用正则表达式查找被```json ```或```包裹的JSON内容
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    
    match = re.search(pattern, response)
    
    if match:
        try:
            # 提取JSON字符串并解析
            json_str = match.group(1).strip()
            data = json.loads(json_str)
            
            # 从JSON中提取所有值组成列表
            values = []
            for _, value in data.items():
                values.extend(value)
            
            return values
        except json.JSONDecodeError:
            return []
    else:
        return []

response = requests.request("POST", url, json=payload, headers=headers)

print(parser_response_json(response.json().get("choices")[0].get("message",{}).get("content","")))