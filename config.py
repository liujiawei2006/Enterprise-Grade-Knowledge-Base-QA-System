"""配置加载模块：从环境变量 / .env 文件读取运行参数。"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class LLMConfig:
    """大模型服务的连接配置。"""
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_history: int
    timeout: int
    system_prompt: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key or api_key.startswith("sk-你的"):
            raise SystemExit(
                "❌ 未检测到有效的 LLM_API_KEY。\n"
                "   请先复制 .env.example 为 .env 并填入你的 API 密钥。\n"
                "   免费申请：https://cloud.siliconflow.cn （注册即送额度）"
            )
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"),
            api_key=api_key,
            model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            temperature=_get_float("LLM_TEMPERATURE", 0.7),
            max_history=_get_int("LLM_MAX_HISTORY", 20),
            timeout=_get_int("LLM_TIMEOUT", 60),
            system_prompt=os.getenv("LLM_SYSTEM_PROMPT", "你是一个乐于助人的中文 AI 助手。"),
        )
