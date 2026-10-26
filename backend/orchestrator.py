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
    IntentAgent,
    RetrievalAgent,
    QueryRewriteAgent,
    WebAnswerAgent,
    SciencePopAgent,
    DebunkAgent,
    MisconceptionOriginAgent,
    SourceAgent,
    ExtensionAgent,
    ReviewAgent,
    _parse_json,
)
from web_search import web_search


class Orchestrator:
    def __init__(self):
        self.intent = IntentAgent()
        self.retrieval = RetrievalAgent()
        self.rewrite = QueryRewriteAgent()
        self.web_answer = WebAnswerAgent()
        self.science = SciencePopAgent()
        self.debunk = DebunkAgent()
        self.origin = MisconceptionOriginAgent()
        self.source = SourceAgent()
        self.extension = ExtensionAgent()
        self.review = ReviewAgent()

    # ------------------------------------------------------------------
    def _route(self, query: str):
        """
        第 1 步：先判意图，再决定链路。

        verify（求证）→ 辟谣链路，回答顶部给“真的/假的”；
        explain（求知）→ 冷知识链路，回答顶部给“问题讲解”，不下真假结论。
        """
        intent = self.intent.run(query)
        return intent, ("debunk" if intent == "verify" else "knowledge")

    # ------------------------------------------------------------------
    @staticmethod
    def _norm_verdict(verdict):
        """知识库历史数据里 fake/false 混用，统一成前端约定的三种值。"""
        if verdict == "false":
            return "fake"
        if verdict in ("true", "fake", "partial"):
            return verdict
        return None

    @staticmethod
    def _rank_for_route(materials: list, route: str) -> list:
        """
        按路由重排检索结果。

        辟谣问题优先取 debunk 条目；同分时原来按文件顺序取 top-1，导致
        “人的血液是蓝色的吗”命中章鱼条目（知识）而不是人类血液条目（辟谣）。
        排序键依次为：类型是否匹配路由 → 相关度 → 标题覆盖 → 正文覆盖。
        """
        prefer = "debunk" if route == "debunk" else "knowledge"
        return sorted(
            materials,
            key=lambda m: (
                m.get("type") == prefer,
                m.get("_score", 0),
                m.get("_title_overlap", 0),
                m.get("_body_overlap", 0),
            ),
            reverse=True,
        )

    @staticmethod
    def _evidence_for_route(materials: list, route: str) -> list:
        """
        辟谣链路只接受“能直接支撑判断”的证据：
          - debunk 条目本身，或
          - 标题直接命中 ≥2 个实义 bigram 的 knowledge 条目。
        只有间接正文提及的条目（如“太阳从西边升起吗”误命中“金星的一天”）
        不作为辟谣依据，避免用不相关材料硬编结论。
        """
        if route != "debunk":
            return materials
        return [
            m for m in materials
            if m.get("type") == "debunk"
            or m.get("_exact")
            or m.get("_title_overlap", 0) >= 2
        ]

    def _web_answer(self, query: str, route: str, intent: str, results: list) -> dict:
        """
        本地库无依据时的第二来源：联网搜索 + 引用编号。

        置信度只看「搜到的来源里有没有学术/权威站点」，不让模型自评：
          - 有 → 🟢 有权威来源支撑（仍不下真/假结论，因为没人工核校）
          - 无 → 🟡 无法确认来源，仅供参考
        """
        answer = self.web_answer.run(query, results)
        has_authority = any(r.get("authoritative") for r in results)

        # 模型自己都说“搜不到”，就不该挂高置信度
        uncertain = any(
            k in answer for k in ("无法确认", "未提及", "没有找到", "未涉及", "不足以", "未找到")
        )
        if uncertain:
            level, label = "low", "🟡 搜索结果不足以确认 · 仅供参考"
        elif has_authority:
            level, label = "high", "🟢 有学术/权威来源支撑 · 建议查看原始链接"
        else:
            level, label = "medium", "🟡 无法确认来源 · 仅展示搜索结果"

        return {
            "route": route,
            "intent": intent,
            "verdict": None,
            "answer": answer,
            "truth": "",
            "origin": "",
            "why": "",
            "source": f"联网搜索 · {len(results)} 条结果（本地知识库未命中）",
            "confidence": level,
            "confidence_label": label,
            "related": [],
            "materials_hit": 0,
            "materials_candidates": 0,
            "refused": False,
            "evidence_source": "web",
            "web_results": results,
        }

    def _refuse(self, route: str, candidates: int, intent: str = "explain") -> dict:
        """无可靠证据时的诚实拒答（PRD 7.2 防线 3）。"""
        return {
            "route": route,
            "intent": intent,
            "evidence_source": "none",
            "verdict": None,
            "answer": (
                "抱歉，这个问题我没有在本地知识库里找到可靠依据，不能凭空下结论。"
                "你可以换一种问法，或补充具体场景；涉及健康、安全的判断建议咨询专业人士。"
            ),
            "truth": "",
            "origin": "",
            "why": "",
            "source": "暂无可靠来源",
            "confidence": "low",
            "confidence_label": "🔴 存疑 · 无可靠证据，已拒答",
            "related": [],
            "materials_hit": 0,
            "materials_candidates": candidates,
            "refused": True,
        }

    # ------------------------------------------------------------------
    def handle(self, query: str) -> dict:
        """
        主流程。返回统一响应结构，字段对齐前端渲染需求：
        {
          route, intent, verdict, answer, truth, origin, why,
          source, confidence, confidence_label, related, materials_hit, refused
        }
        """
        intent, route = self._route(query)

        # 第 2 步：检索 Agent（本地 RAG，不耗 LLM），按路由重排 + 证据过滤
        candidates = self.retrieval.run(query)
        materials = self._evidence_for_route(
            self._rank_for_route(candidates, route), route
        )

        # 本地没有直接证据 → 先联网搜；搜不到才诚实拒答
        if not materials:
            # 先把问句改写成检索式，再联网；改写失败就用原句
            search_query = self.rewrite.run(query) if llm.provider != "mock" else query
            results = web_search.search(search_query)
            if results:
                return self._web_answer(query, route, intent, results)
            return self._refuse(route, len(candidates), intent)

        review = self.review.run(materials)
        top = materials[0]

        resp = {
            "route": route,
            "intent": intent,
            "evidence_source": "local",
            "verdict": None,
            "answer": "",
            "truth": "",
            "origin": "",
            "why": "",
            "source": self.source.run(materials),
            "confidence": review["level"],
            "confidence_label": review["label"],
            "related": self.extension.run(query, materials),
            "materials_hit": len(materials),
            "materials_candidates": len(candidates),
            "refused": False,
        }

        if route == "debunk":
            # 辟谣链路：辟谣 Agent + 误解溯源 Agent
            debunk_result = self.debunk.run(query, materials)
            resp["answer"] = debunk_result.get("answer", "")
            resp["verdict"] = debunk_result.get("verdict") or top.get("verdict")
            resp["truth"] = debunk_result.get("truth", "") or top.get("truth", "")
            origin = self.origin.run(query, materials)
            resp["origin"] = origin["origin"]
            resp["why"] = origin["why"]
        else:
            # 冷知识链路：科普 Agent
            resp["answer"] = self.science.run(query, materials)
            resp["truth"] = top.get("truth", "")

        # 只有“求证型”问题才给真假结论；问“是什么/为什么”不下真假判断，
        # 否则会出现“屎是什么味道的 → 假的！”这种答非所问的标题。
        if intent != "verify":
            resp["verdict"] = None
        else:
            resp["verdict"] = self._norm_verdict(resp["verdict"])

        # mock 模式下用库内整理好的答案覆盖占位文本，演示更稳定
        if llm.provider == "mock":
            resp["answer"] = top.get("answer", resp["answer"])

        return resp


orchestrator = Orchestrator()
