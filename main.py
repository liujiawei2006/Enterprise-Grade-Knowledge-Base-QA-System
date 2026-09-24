"""命令行对话机器人入口：多轮上下文 + 流式输出 + 对话重置。"""
import sys

import requests

from config import LLMConfig
from conversation import Conversation
from llm_client import LLMClient

BANNER = r"""
====================================================
        🤖  命令行智能对话机器人  🤖
   多轮上下文 · 流式输出 · 支持重置 (OpenAI 兼容)
====================================================
"""

HELP_TEXT = """
可用命令：
  /help        显示本帮助
  /reset       清空上下文，开始新对话
  /history     查看当前上下文轮数
  /exit /quit  退出程序
其余任意输入都会作为消息发送给模型。
"""


def print_stream(client: LLMClient, conv: Conversation) -> None:
    """发送当前上下文并以流式方式打印回复，同时把完整回复写入历史。"""
    full_reply = []
    print("\n🤖 助手: ", end="", flush=True)
    try:
        for delta in client.chat_stream(conv.build_prompt()):
            print(delta, end="", flush=True)   # 逐字打字机效果
            full_reply.append(delta)
        print()  # 换行
    except requests.RequestException as e:
        print(f"\n\n⚠️  网络/请求错误: {e}")
        return
    except RuntimeError as e:
        print(f"\n\n⚠️  {e}")
        return
    except KeyboardInterrupt:
        print("\n\n⏹️  已中断本次生成。")
        return

    text = "".join(full_reply).strip()
    if text:
        conv.add_assistant(text)   # 只有成功才写入历史，避免污染上下文


def main() -> None:
    print(BANNER)
    try:
        cfg = LLMConfig.from_env()
    except SystemExit as e:
        print(e)
        sys.exit(1)

    print(f"✅ 已连接服务: {cfg.base_url}")
    print(f"✅ 当前模型  : {cfg.model}")
    print(HELP_TEXT)

    client = LLMClient(cfg.base_url, cfg.api_key, cfg.model,
                       cfg.temperature, cfg.timeout)
    conv = Conversation(cfg.system_prompt, cfg.max_history)

    while True:
        try:
            user_input = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit"):
            print("\n👋 再见！")
            break
        if cmd == "/help":
            print(HELP_TEXT)
            continue
        if cmd == "/reset":
            conv.reset()
            print("🔄 上下文已重置，开始全新对话。")
            continue
        if cmd == "/history":
            print(f"📚 当前上下文轮数: {conv.turn_count}")
            continue

        conv.add_user(user_input)
        print_stream(client, conv)


if __name__ == "__main__":
    main()
