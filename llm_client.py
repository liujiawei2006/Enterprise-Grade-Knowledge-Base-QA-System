"""大模型 API 客户端：OpenAI 兼容协议，支持流式输出。"""
import json
from typing import Dict, Generator, List

import requests


class LLMClient:
    """基于 requests 的流式聊天客户端，兼容任意 OpenAI 风格接口。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float, timeout: int):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """发送对话请求，以生成器形式逐块 yield 文本增量（delta content）。"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }

        resp = self._session.post(
            url, json=payload, stream=True, timeout=self.timeout
        )

        if resp.status_code != 200:
            detail = resp.text[:500]
            resp.close()
            raise RuntimeError(f"API 请求失败 [HTTP {resp.status_code}]: {detail}")

        # 直接读取原始字节并手动按 UTF-8 解码，彻底规避 requests 对 SSE 响应
        # 编码猜测（默认 ISO-8859-1）导致的中文乱码问题。
        try:
            for raw in resp.iter_lines():  # 返回 bytes
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                # SSE 每行形如: "data: {...}"
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta
        finally:
            resp.close()
