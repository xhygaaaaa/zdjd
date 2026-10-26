"""
大模型客户端封装
====================
对应 PRD 第十一章「技术方案」。

设计目标：
1. 默认使用 mock 模式——不填 API Key 也能完整运行、演示（面试友好）。
2. 填入 API Key 后自动切换到真实大模型（当前接入 MiniMax，兼容 OpenAI 格式，
   后续换 DeepSeek / 通义千问只需改 BASE_URL 和 MODEL）。

环境变量（见 .env.example）：
- LLM_PROVIDER : "mock" | "minimax" | "deepseek"
- LLM_API_KEY  : 各平台申请的密钥
- LLM_BASE_URL : API 地址
- LLM_MODEL    : 模型名
"""

import os
import json


class LLMClient:
    """统一的大模型调用入口。"""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "mock").lower()
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.minimax.chat/v1")
        self.model = os.getenv("LLM_MODEL", "abab6.5s-chat")

        # 没有 Key 时强制回落到 mock，保证项目开箱即跑
        if self.provider != "mock" and not self.api_key:
            print("[LLMClient] 未检测到 API Key，自动切换为 mock 模式")
            self.provider = "mock"

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """
        发送一次对话请求，返回模型的文本回复。

        Args:
            system_prompt: 系统提示词（定义 Agent 的角色和职责）
            user_prompt:   用户输入 / 上游 Agent 传来的内容
            temperature:   采样温度，越高越发散
        """
        if self.provider == "mock":
            return self._mock_chat(system_prompt, user_prompt)
        return self._real_chat(system_prompt, user_prompt, temperature)

    # ------------------------------------------------------------------
    # 真实调用（MiniMax / DeepSeek 均兼容 OpenAI 的 chat/completions 格式）
    # ------------------------------------------------------------------
    def _real_chat(self, system_prompt, user_prompt, temperature):
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            # 生产环境应记录日志；这里降级返回，保证服务不中断
            print(f"[LLMClient] 真实调用失败，降级 mock：{e}")
            return self._mock_chat(system_prompt, user_prompt)

    # ------------------------------------------------------------------
    # Mock：无 Key 时用于演示。根据 system_prompt 里的角色标记返回结构化占位。
    # ------------------------------------------------------------------
    def _mock_chat(self, system_prompt, user_prompt):
        if "总调度" in system_prompt or "Orchestrator" in system_prompt:
            # 简单启发式：含"吗/是不是/真的假的"判为辟谣，否则冷知识
            debunk_kw = ["吗", "是不是", "真的假的", "有害", "会不会", "谣言"]
            route = "debunk" if any(k in user_prompt for k in debunk_kw) else "knowledge"
            return json.dumps({"route": route}, ensure_ascii=False)
        if "科普达人" in system_prompt:
            return "[mock 科普回答] 基于检索到的资料，这是一段生动有趣的科普解释。"
        if "辟谣专家" in system_prompt:
            return "[mock 辟谣结论] 经核实，这个说法需要结合资料判断真伪。"
        return "[mock] 这是 mock 模式的占位回复，填入 API Key 后将调用真实大模型。"


# 单例，供各 Agent 共享
llm = LLMClient()
