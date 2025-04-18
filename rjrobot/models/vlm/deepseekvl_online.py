from typing import Any, Dict, List, Optional, Union
import requests
import os
import logging
import re
import json

from .base_model import VLMBaseModel
from rjrobot.utils import img_to_base64


class OnlineVLM(VLMBaseModel):
    def __init__(
        self,
        url: str,
        model: str,
        api: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 50,
        frequency_penalty: float = 0.5,
    ):
        self.url = url
        self.model = model
        # self.api = api
        # DEBUG: 测试用
        self.api = api if api else os.environ["SILICONFLOW_API"]

        self.headers = {
            "Authorization": f"Bearer {self.api}",
            "Content-Type": "application/json",
        }
        self.system_prompt = {}
        self.history_messages = []
        self.payload = {
            "model": self.model,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "n": 1,
            "stop": [],
            "messages": self.history_messages,
        }
        self.pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"

    def set_system_prompt(self, system_prompt):
        self.system_prompt = {"role": "system", "content": system_prompt}

    def set_param(
        self,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        frequency_penalty: float = 0.5,
    ):
        param_dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
        }
        self.payload.update(param_dict)

    def clear_history(self):
        self.history_messages.clear()

    def predict(self, observation_data: Any, prompt_text: str):
        self.history_messages.append(self._input2prompt(observation_data, prompt_text))
        if self.system_prompt:
            self.payload["messages"] = [self.system_prompt] + self.history_messages
        logging.info("vlm inference start")
        response = requests.request(
            "POST", self.url, json=self.payload, headers=self.headers
        )
        if response.status_code != 200:
            logging.warning(response.text)
            return []
        r_message = response.json().get("choices")[0].get("message", None)
        if r_message is None:
            logging.warning("warning: vlm response None:", response.json())
            return []
        self.history_messages.append(r_message)
        return self._parser_response_json(r_message.get("content", None))

    def _input2prompt(self, input: Dict[str, Dict[str, Any]], prompt_text: str) -> dict:
        user_prompt = []
        for img_name, img in input["image"].items():
            img_b = img_to_base64(img[0], contain_header=True)
            user_prompt.append(
                {
                    "image_url": {
                        "detail": "low",
                        "url": img_b,
                    },
                    "type": "image_url",
                }
            )
        user_prompt.append(
            {
                "text": f"图片为不同视角摄像机的采集照片，需要机械臂执行的任务为：{prompt_text}",
                "type": "text",
            }
        )
        return {"role": "user", "content": user_prompt}

    def _parser_response_json(self, response: str):
        match = re.search(self.pattern, response)
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
            # except json.JSONDecodeError:
            except Exception as e:
                logging.warning(
                    f"无法解析响应中的JSON内容,原始响应内容:{response}"
                )
                print(
                    f"无法解析响应中的JSON内容,原始响应内容:{response}"
                )
                return []
        else:
            logging.warning(f"响应中未找到JSON内容,原始响应内容:{response}")
            print(f"响应中未找到JSON内容,原始响应内容:{response}")
            return []
