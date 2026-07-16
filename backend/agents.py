"""
功能 Agent 集合
==================
对应 PRD 第五章「多 Agent 架构设计」5.2 各 Agent 职责与 Prompt。

PRD 定义了 8 个 Agent。为控制真实 API 的调用成本与延迟（见 AskUser 决策），
本实现采用「精简版」：保留每个 Agent 的独立职责与 Prompt 定义（便于讲解架构、
未来拆分），但在执行时合并为 2-3 次大模型调用：
  - 检索 Agent      -> 本地完成（不消耗 LLM）
  - 翻译 Agent      -> 本地库为中文，透传（不消耗 LLM）
  - 科普/辟谣 Agent -> 1 次 LLM 调用（生成主体回答）
  - 溯源/误解溯源   -> 复用检索结果 + 生成结果，本地整合
  - 延伸 Agent      -> 复用知识库 related 字段，本地完成
  - 审查 Agent      -> 1 次 LLM 调用（可选，评估置信度）

每个 Agent 的 SYSTEM_PROMPT 直接抄录自 PRD，方便对照。
"""

from llm_client import llm
from knowledge_base import kb


# ---------------------------------------------------------------------------
# Agent 1：检索 Agent —— PRD 5.2 Agent 1
# ---------------------------------------------------------------------------
class RetrievalAgent:
    """从本地知识库检索相关资料（RAG 第①层）。命中返回资料，未命中返回空。"""

    def run(self, query: str):
        results = kb.search(query, top_k=3)
        return results


# ---------------------------------------------------------------------------
# Agent 3：科普 Agent —— PRD 5.2 Agent 3
# ---------------------------------------------------------------------------
class SciencePopAgent:
    SYSTEM_PROMPT = (
        "你是擅长讲冷知识的科普达人。"
        "用轻松有趣的语气回答：1.先给直接答案 2.再通俗解释为什么 "
        "3.语气活泼像朋友聊天 4.不超300字 5.只用提供的资料信息，不编造。"
    )

    def run(self, query: str, materials: list):
        context = self._format_materials(materials)
        user = f"用户问题：{query}\n\n检索到的资料：\n{context}"
        return llm.chat(self.SYSTEM_PROMPT, user)

    @staticmethod
    def _format_materials(materials):
        if not materials:
            return "（未检索到本地资料）"
        return "\n".join(
            f"- {m['title']}：{m.get('answer', m.get('truth', ''))}" for m in materials
        )


# ---------------------------------------------------------------------------
# Agent 4：辟谣 Agent —— PRD 5.2 Agent 4
# ---------------------------------------------------------------------------
class DebunkAgent:
    SYSTEM_PROMPT = (
        "你是辟谣专家。判断用户说法是：✅真 / ❌假 / 🔶部分正确。"
        "给出：1.明确结论 2.正确的事实是什么 3.基于资料，不臆断。"
        "如果资料不足以判断，说“暂时无法确认真伪”。不超300字。"
    )

    def run(self, query: str, materials: list):
        context = SciencePopAgent._format_materials(materials)
        user = f"用户说法：{query}\n\n检索到的资料：\n{context}"
        return llm.chat(self.SYSTEM_PROMPT, user)


# ---------------------------------------------------------------------------
# Agent 5：误解溯源 Agent（PRD 亮点）—— PRD 5.2 Agent 5
# ---------------------------------------------------------------------------
class MisconceptionOriginAgent:
    """分析误解的起源与传播原因。优先复用知识库已整理的 origin/why 字段。"""

    def run(self, materials: list):
        for m in materials:
            if m.get("origin") or m.get("why"):
                return {"origin": m.get("origin", ""), "why": m.get("why", "")}
        return {"origin": "误解起源待考证", "why": ""}


# ---------------------------------------------------------------------------
# Agent 7：溯源 Agent —— PRD 5.2 Agent 7
# ---------------------------------------------------------------------------
class SourceAgent:
    """标注信息来源。复用知识库 source 字段。"""

    def run(self, materials: list):
        if materials and materials[0].get("source"):
            return materials[0]["source"]
        return "来源待考证"


# ---------------------------------------------------------------------------
# Agent 6：延伸 Agent —— PRD 5.2 Agent 6
# ---------------------------------------------------------------------------
class ExtensionAgent:
    """生成「你可能还想知道」。复用知识库 related 关联字段。"""

    def run(self, materials: list):
        related_titles = []
        seen = set()
        for m in materials:
            for rid in m.get("related", []):
                if rid in seen:
                    continue
                seen.add(rid)
                item = kb.get_by_id(rid)
                if item:
                    related_titles.append(item["title"])
        return related_titles[:3]


# ---------------------------------------------------------------------------
# Agent 8：审查 Agent（防幻觉灵魂）—— PRD 5.2 Agent 8 / 第七章防幻觉
# ---------------------------------------------------------------------------
class ReviewAgent:
    """
    评估置信度，对应 PRD 第七章「三级置信度机制」：
      🟢 确证：有权威来源
      🟡 趣味参考：有来源但非权威
      🔴 存疑：无可靠来源 -> 改为诚实拒答
    """

    def run(self, materials: list):
        if not materials:
            return {"level": "low", "label": "🔴 存疑 · 暂时没找到可靠资料"}
        conf = materials[0].get("confidence", "medium")
        if conf == "high":
            return {"level": "high", "label": "🟢 确证 · 有权威来源"}
        return {"level": "medium", "label": "🟡 趣味参考 · 建议核实"}
