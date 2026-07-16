"""
总调度 Agent（Orchestrator）
==============================
对应 PRD 第五章 5.1 整体架构 与 5.3 冷知识 vs 辟谣的动态调度。

职责：
1. 判断用户问题类型（冷知识 knowledge / 误解辟谣 debunk）。
2. 按 PRD 5.3 分发到不同 Agent 链路：
     冷知识："检索 → 翻译 → 科普 → 溯源 → 延伸 → 审查"
     辟谣  ："检索 → 翻译 → 辟谣 → 误解溯源 → 溯源 → 延伸 → 审查"
3. 汇总各 Agent 结果为统一响应结构，交给前端渲染
   （对应 PRD 第六章辟谣回答结构 与 第九章 AI 交互设计）。
"""

import json

from llm_client import llm
from agents import (
    RetrievalAgent,
    SciencePopAgent,
    DebunkAgent,
    MisconceptionOriginAgent,
    SourceAgent,
    ExtensionAgent,
    ReviewAgent,
)


class Orchestrator:
    # PRD 5.2 Agent 0（总调度）的 Prompt
    ROUTER_PROMPT = (
        "你是总调度 Agent（Orchestrator）。判断用户输入属于哪种类型，"
        "只返回 JSON：{\"route\": \"knowledge\"} 表示纯冷知识提问，"
        "{\"route\": \"debunk\"} 表示在验证某个说法真假 / 辟谣。"
        "含“吗、是不是、真的假的、有害、会不会”等验证语气的判为 debunk。"
    )

    def __init__(self):
        self.retrieval = RetrievalAgent()
        self.science = SciencePopAgent()
        self.debunk = DebunkAgent()
        self.origin = MisconceptionOriginAgent()
        self.source = SourceAgent()
        self.extension = ExtensionAgent()
        self.review = ReviewAgent()

    # ------------------------------------------------------------------
    def _route(self, query: str) -> str:
        """第 1 步：判断问题类型（1 次轻量 LLM 调用）。"""
        raw = llm.chat(self.ROUTER_PROMPT, query, temperature=0)
        try:
            return json.loads(raw).get("route", "knowledge")
        except Exception:
            # 解析失败时用关键词兜底
            kw = ["吗", "是不是", "真的假的", "有害", "会不会", "谣言"]
            return "debunk" if any(k in query for k in kw) else "knowledge"

    # ------------------------------------------------------------------
    def handle(self, query: str) -> dict:
        """
        主流程。返回统一响应结构，字段对齐前端渲染需求：
        {
          route, verdict, answer, truth, origin, why,
          source, confidence, confidence_label, related, materials_hit
        }
        """
        route = self._route(query)

        # 第 2 步：检索 Agent（本地 RAG，不耗 LLM）
        materials = self.retrieval.run(query)

        # 防幻觉：无资料且真实模式下，诚实拒答（PRD 7.2 防线 3）
        review = self.review.run(materials)

        resp = {
            "route": route,
            "verdict": None,
            "answer": "",
            "truth": "",
            "origin": "",
            "why": "",
            "source": self.source.run(materials),
            "confidence": review["level"],
            "confidence_label": review["label"],
            "related": self.extension.run(materials),
            "materials_hit": len(materials),
        }

        # 命中知识库时，直接采用结构化字段（质量最高，PRD 4.2 第①层）
        top = materials[0] if materials else None

        if route == "debunk":
            # 辟谣链路：辟谣 Agent + 误解溯源 Agent
            resp["answer"] = self.debunk.run(query, materials)
            if top:
                resp["verdict"] = top.get("verdict")
                resp["truth"] = top.get("truth", "")
            origin = self.origin.run(materials)
            resp["origin"] = origin["origin"]
            resp["why"] = origin["why"]
        else:
            # 冷知识链路：科普 Agent
            resp["answer"] = self.science.run(query, materials)
            if top:
                resp["verdict"] = top.get("verdict")
                resp["truth"] = top.get("truth", "")

        # 若命中高质量条目，用库内 answer 覆盖（演示时更稳定；真实模式下 LLM 已生成）
        if top and llm.provider == "mock":
            resp["answer"] = top.get("answer", resp["answer"])

        return resp


orchestrator = Orchestrator()
