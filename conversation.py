"""多轮对话上下文管理：维护消息历史、裁剪窗口、重置。"""
from typing import Dict, List


class Conversation:
    """保存一份可复位的对话历史（system + 多轮 user/assistant）。"""

    def __init__(self, system_prompt: str, max_history: int = 20):
        self.system_prompt = system_prompt
        self.max_history = max_history  # 最多保留的“历史消息条数”，0 = 不限制
        self._messages: List[Dict[str, str]] = []

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def reset(self) -> None:
        """清空上下文，开始全新对话。"""
        self._messages.clear()

    def build_prompt(self) -> List[Dict[str, str]]:
        """构造发送给模型的完整 messages（system + 裁剪后的历史）。"""
        history = self._messages
        if self.max_history > 0 and len(history) > self.max_history:
            history = history[-self.max_history:]
        return [{"role": "system", "content": self.system_prompt}, *history]

    @property
    def turn_count(self) -> int:
        """已进行的对话轮数（一问一答算一轮）。"""
        return sum(1 for m in self._messages if m["role"] == "user")
