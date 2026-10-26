"""
功能 Agent 集合
==================
对应 PRD 第五章「多 Agent 架构设计」5.2 各 Agent 职责与 Prompt。

设计原则（本次重构核心）：
每个 Agent 都是「知识库优先 + LLM 兜底」的双通道：
  - 命中本地知识库时，直接采用整理好的结构化字段（质量最高、零延迟）。
  - 未命中时，调用大模型现场生成结构化结果，而不是返回"待考证"占位。
这样即使用户问的是库里没有的自由问题，辟谣链路的
verdict / truth / origin / why / related 也都是真实产出，而非空壳。

调用次数（真实模式）：
  - 冷知识：路由(1) + 科普(1) + 延伸兜底(0~1)
  - 辟谣  ：路由(1) + 辟谣(1) + 误解溯源兜底(0~1) + 延伸兜底(0~1)
命中知识库时兜底调用自动省去。
"""

import json
import re

from llm_client import llm
from knowledge_base import kb


# ---------------------------------------------------------------------------
# 工具：从大模型返回文本里稳健地抽取 JSON
# ---------------------------------------------------------------------------
def _parse_json(raw: str, default):
    """
    大模型有时会用 ```json ``` 包裹或前后带说明文字，这里做容错解析。
    解析失败返回 default，保证上层永不崩。
    """
    if not raw:
        return default
    text = raw.strip()
    # 去掉 ```json ... ``` 代码围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    # 直接尝试
    try:
        return json.loads(text)
    except Exception:
        pass
    # 退而求其次：截取第一个 { 或 [ 到最后一个 } 或 ]
    for lb, rb in (("{", "}"), ("[", "]")):
        i, j = text.find(lb), text.rfind(rb)
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                continue
    return default


# ---------------------------------------------------------------------------
# Agent 0.5：意图 Agent —— 区分“问知识”与“验真假”
# ---------------------------------------------------------------------------
class IntentAgent:
    """
    判断用户到底是在「问一个开放问题」还是在「求证一个说法」。

    背景：原来的总调度只知道 knowledge/debunk 两条链路，于是
    “屎是什么味道的”这种纯提问也会走辟谣链路，右侧顶部挂一个
    “真的假的? → 假的!”，看上去像在给用户的问题判真假。

    规则优先（确定性、零延迟、可解释），只有规则判不出来时才问模型：
      verify  = 求证型：「…吗」「是不是」「真的假的」「会不会」「致癌吗」
      explain = 求知型：「为什么」「怎么」「是什么」「多少」「哪里」
    """

    EXPLAIN_PAT = re.compile(
        r"(为什么|为何|怎么|怎样|如何|是什么|什么味道|什么颜色|什么原理|"
        r"多少|多久|多大|多高|多远|多快|哪里|哪儿|哪个|哪些|什么样|原理|区别)"
    )
    VERIFY_PAT = re.compile(
        r"(是不是|是否|真的假的|真的吗|假的吗|会不会|有没有|能不能|"
        r"可不可以|算不算|对不对|谣言|辟谣|有害|有毒|致癌|安全吗|智商税)"
    )

    SYSTEM_PROMPT = (
        "判断用户输入属于哪一类，只返回 JSON："
        "{\"intent\":\"verify\"} 表示用户在求证某个说法真假；"
        "{\"intent\":\"explain\"} 表示用户在询问知识/原理/事实。"
    )

    def run(self, query: str) -> str:
        q = query.strip()
        has_verify = bool(self.VERIFY_PAT.search(q))
        has_explain = bool(self.EXPLAIN_PAT.search(q))

        # 只命中一类，直接给结论
        if has_verify and not has_explain:
            return "verify"
        if has_explain and not has_verify:
            return "explain"

        # 两类都命中（如“为什么…吗”），看句尾语气：以“吗/么”收尾更像求证
        if has_verify and has_explain:
            tail = q.rstrip("?？。!！ ")
            return "verify" if tail.endswith(("吗", "么")) else "explain"

        # 都没命中：陈述句（如“隔夜水致癌”）交给模型判断
        raw = llm.chat(self.SYSTEM_PROMPT, q, temperature=0)
        data = _parse_json(raw, None)
        if isinstance(data, dict) and data.get("intent") in ("verify", "explain"):
            return data["intent"]
        return "explain"


# ---------------------------------------------------------------------------
# Agent 1：检索 Agent —— PRD 5.2 Agent 1
# ---------------------------------------------------------------------------
class RetrievalAgent:
    """从本地知识库检索相关资料（RAG 第①层）。命中返回资料，未命中返回空。"""

    def run(self, query: str):
        return kb.search(query, top_k=3)


def _format_materials(materials):
    """把检索到的资料拼成给大模型看的上下文。"""
    if not materials:
        return "（未检索到本地资料，请基于公认科学常识回答）"
    return "\n".join(
        f"- {m['title']}：{m.get('answer', m.get('truth', ''))}" for m in materials
    )


# ---------------------------------------------------------------------------
# Agent 2.4：检索改写 Agent —— 把问句改写成搜索引擎友好的关键词
# ---------------------------------------------------------------------------
class QueryRewriteAgent:
    """
    搜索引擎对整句提问很迟钝：“南京联成的主营业务是什么”会返回南京旅游攻略，
    而“南京联成 主营业务”能直接命中公司百科。
    所以联网前先用一次低温调用做 query rewriting。
    """

    SYSTEM_PROMPT = (
        "把用户问题改写成适合搜索引擎的关键词，只返回 JSON："
        "{\"query\":\"关键词1 关键词2\"}。要求：保留核心实体，去掉“是什么/为什么/吗”"
        "等疑问词，2-5 个关键词，空格分隔，不要标点和解释。"
    )

    def run(self, query: str) -> str:
        raw = llm.chat(self.SYSTEM_PROMPT, query, temperature=0)
        data = _parse_json(raw, None)
        if isinstance(data, dict):
            q = str(data.get("query", "")).strip()
            if q:
                return q
        return query


# ---------------------------------------------------------------------------
# Agent 2.5：联网搜索兜底 Agent —— 本地库没有时的第二来源
# ---------------------------------------------------------------------------
class WebAnswerAgent:
    """
    只依据联网搜索结果回答，不凭记忆，不下真假结论。

    为什么不让模型自由发挥：搜索结果本身可能是错的（知乎、自媒体），
    所以这里要求逐句标 [1][2]，凡是结果里没有的内容就不许写。
    置信度由调用方按“是否有学术/权威来源”判定，不由模型自评。
    """

    SYSTEM_PROMPT = (
        "你是严谨的科普助手。只依据给定的搜索结果回答用户问题。"
        "规则：1. 每条结论后用 [1][2] 标出依据的搜索结果编号；"
        "2. 搜索结果不足以回答时，直接说“现有搜索结果无法确认”，不要脑补；"
        "3. 不要下“真/假”的结论，只陈述搜到的事实；"
        "4. 不要编造结果中没有的数据、论文名或来源；5. 不超250字。"
    )

    def run(self, query: str, results: list) -> str:
        context = "\n".join(
            f"[{i + 1}] {r['title']}｜{r.get('snippet', '')}"
            for i, r in enumerate(results)
        )
        user = f"用户问题：{query}\n\n搜索结果：\n{context}"
        return llm.chat(self.SYSTEM_PROMPT, user, temperature=0.2)


# ---------------------------------------------------------------------------
# Agent 3：科普 Agent —— PRD 5.2 Agent 3
# ---------------------------------------------------------------------------
class SciencePopAgent:
    SYSTEM_PROMPT = (
        "你是擅长讲冷知识的科普达人。"
        "用轻松有趣的语气回答：1.先给直接答案 2.再通俗解释为什么 "
        "3.语气活泼像朋友聊天 4.不超300字 5.优先用提供的资料，"
        "资料不足时可用公认科学常识，但不要编造具体数据或来源。"
    )

    # 兼容旧调用：orchestrator 里曾用 SciencePopAgent._format_materials
    _format_materials = staticmethod(_format_materials)

    def run(self, query: str, materials: list):
        user = f"用户问题：{query}\n\n检索到的资料：\n{_format_materials(materials)}"
        return llm.chat(self.SYSTEM_PROMPT, user)


# ---------------------------------------------------------------------------
# Agent 4：辟谣 Agent —— PRD 5.2 Agent 4
# ---------------------------------------------------------------------------
class DebunkAgent:
    """
    辟谣专家。返回结构化结果 {verdict, answer, truth}，
    而不是一段纯文本 —— 这样即使本地库没命中，前端的真/假徽章和
    "真相"栏也能拿到真实内容。
    """

    SYSTEM_PROMPT = (
        "你是辟谣专家。基于用户说法和提供的资料判断真伪，只返回 JSON，不要任何多余文字。"
        "格式：{\"verdict\":\"true|fake|partial\",\"answer\":\"面向用户的辟谣解释，不超250字\","
        "\"truth\":\"一句话概括正确的事实\"}。"
        "verdict 含义：true=说法正确，fake=说法错误，partial=部分正确。"
        "优先依据资料；资料不足时依据公认科学常识判断，并在 answer 中说明这是基于常识的分析。"
        "不要编造精确数据或虚构来源。"
    )

    def run(self, query: str, materials: list) -> dict:
        user = f"用户说法：{query}\n\n检索到的资料：\n{_format_materials(materials)}"
        raw = llm.chat(self.SYSTEM_PROMPT, user, temperature=0.3)
        data = _parse_json(raw, None)
        if isinstance(data, dict) and data.get("answer"):
            verdict = data.get("verdict")
            if verdict == "false":          # 兼容模型输出的 false
                verdict = "fake"
            if verdict not in ("true", "fake", "partial"):
                verdict = None
            return {
                "verdict": verdict,
                "answer": data["answer"],
                "truth": data.get("truth", ""),
            }
        # 解析失败（如 mock 模式返回纯文本）：把原文当作答案，其余留空
        return {"verdict": None, "answer": raw, "truth": ""}


# ---------------------------------------------------------------------------
# Agent 5：误解溯源 Agent（PRD 亮点）—— PRD 5.2 Agent 5
# ---------------------------------------------------------------------------
class MisconceptionOriginAgent:
    """
    分析误解的起源与传播原因。
    知识库有整理好的 origin/why 就直接用；否则调用大模型现场分析。
    """

    SYSTEM_PROMPT = (
        "你是误解溯源专家。分析用户提到的这个常见说法/误解的来源与流传原因，"
        "只返回 JSON，不要多余文字：{\"origin\":\"误解的可能起源，一两句\","
        "\"why\":\"为什么会广泛流传，一两句\"}。"
        "如果这个说法其实是正确的、不存在误解，返回 {\"origin\":\"\",\"why\":\"\"}。"
    )

    def run(self, query: str, materials: list) -> dict:
        # ① 知识库命中优先
        for m in materials:
            if m.get("origin") or m.get("why"):
                return {"origin": m.get("origin", ""), "why": m.get("why", "")}
        # ② LLM 兜底
        raw = llm.chat(self.SYSTEM_PROMPT, f"说法：{query}", temperature=0.5)
        data = _parse_json(raw, None)
        if isinstance(data, dict):
            return {"origin": data.get("origin", ""), "why": data.get("why", "")}
        return {"origin": "", "why": ""}


# ---------------------------------------------------------------------------
# Agent 7：溯源 Agent —— PRD 5.2 Agent 7
# ---------------------------------------------------------------------------
class SourceAgent:
    """标注信息来源。库内有权威来源就用；否则诚实标注为 AI 分析，不编造来源。"""

    def run(self, materials: list) -> str:
        if materials and materials[0].get("source"):
            return materials[0]["source"]
        if llm.provider != "mock":
            return "AI 综合分析 · 无本地权威来源，建议自行核实"
        return "来源待考证"


# ---------------------------------------------------------------------------
# Agent 6：延伸 Agent —— PRD 5.2 Agent 6
# ---------------------------------------------------------------------------
class ExtensionAgent:
    """
    生成「你可能还想知道」。
    命中知识库时复用 related 关联；否则让大模型现场推荐延伸问题。
    """

    SYSTEM_PROMPT = (
        "根据用户的问题，推荐3个相关的、能引发好奇心的延伸问题标题。"
        "只返回 JSON 数组，例如 [\"...\",\"...\",\"...\"]，每个标题不超20字，不要多余文字。"
    )

    def run(self, query: str, materials: list) -> list:
        # ① 知识库关联优先
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
        if related_titles:
            return related_titles[:3]

        # ② LLM 兜底（mock 模式跳过，避免返回无意义占位）
        if llm.provider == "mock":
            return []
        raw = llm.chat(self.SYSTEM_PROMPT, f"用户问题：{query}", temperature=0.7)
        data = _parse_json(raw, [])
        if isinstance(data, list):
            return [str(t) for t in data][:3]
        return []


# ---------------------------------------------------------------------------
# Agent 8：审查 Agent（防幻觉灵魂）—— PRD 5.2 Agent 8 / 第七章防幻觉
# ---------------------------------------------------------------------------
class ReviewAgent:
    """
    评估置信度，对应 PRD 第七章「三级置信度机制」：
      🟢 确证：有权威来源 + 问题直接命中标题（标题覆盖 ≥2 个实义 bigram）
      🟡 参考：有相关资料，但只是间接命中 / 多来源结论不一致
      🔴 存疑：没有可靠证据（该分支由 Orchestrator 直接拒答，不再调用模型）

    说明：这是「基于检索证据的规则置信度」，不是让 LLM 审查自己是否幻觉；
    它降低“答得自信但没依据”的概率，不能证明生成答案一定正确。
    """

    @staticmethod
    def _norm(verdict):
        return "fake" if verdict == "false" else verdict

    def run(self, materials: list) -> dict:
        if not materials:
            return {"level": "low", "label": "🔴 存疑 · 没有可靠证据"}

        top = materials[0]
        title_overlap = top.get("_title_overlap", 0)
        exact = bool(top.get("_exact"))
        has_source = bool(top.get("source"))
        declared = top.get("confidence", "medium")

        # 只有“同一问题的多个直接命中来源”结论不一致时才降级。
        # 不能把不同问题（如“太阳从东边升起”与“太阳从西边升起”）误判为冲突。
        verdicts = {
            self._norm(m.get("verdict"))
            for m in materials
            if m.get("type") == "debunk" and m.get("verdict") and m.get("_exact")
        }
        if len(verdicts) >= 2:
            return {
                "level": "medium",
                "label": "🟡 多来源存在分歧 · 建议核实",
            }

        if declared == "high" and has_source and (exact or title_overlap >= 2):
            return {"level": "high", "label": "🟢 确证 · 直接命中且有权威来源"}
        if has_source or title_overlap >= 1:
            return {
                "level": "medium",
                "label": "🟡 参考 · 有相关资料，但非直接命中",
            }
        return {"level": "medium", "label": "🟡 参考 · 仅有间接资料，建议核实"}
