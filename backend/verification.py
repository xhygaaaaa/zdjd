"""
生成后校验（Post-generation Grounding Check）
================================================
解决原实现的盲区：ReviewAgent 只在「生成前」看检索元数据，
生成出来的回答是否真的被证据支撑，从来没有被检查过。

这里做两件事：
1. 逐条主张核对：回答里的每个事实性主张，能否在检索到的证据里找到支撑；
2. 核心/细节分层：support_ratio 只看核心主张，解释性扩展单独统计，
   避免「结论正确但讲得详细」被误判成幻觉；
3. 结论一致性：模型给出的 verdict 是否与证据里的 verdict 冲突。

输出统一结构，由 Orchestrator 决定是否降级置信度/改判拒答。
mock 模式下降级为 bigram 覆盖率启发式，保证离线也能跑。
"""
import re

from llm_client import llm
from agents import _parse_json

_FACT_SPLIT = re.compile(r"[。！？!?；;\n]+")


def _bigrams(text: str) -> set:
    clean = re.sub(r"[^\w一-鿿]", "", (text or "").lower())
    return {clean[i:i + 2] for i in range(len(clean) - 1)}


class GroundingVerifier:
    """生成后的事实核查器。"""

    SYSTEM_PROMPT = (
        "你是事实核查员。给定【问题】【AI回答】【检索证据】，逐条检查回答中的事实性主张。"
        "只返回 JSON，不要多余文字："
        '{"claims":[{"claim":"主张原文","core":true/false,'
        '"supported":true/false,"evidence":"支撑它的证据片段，没有则空字符串"}],'
        '"support_ratio":0到1之间的小数,"consistent":true/false,'
        '"reason":"一句话说明"}。'
        "判定标准："
        "1. core=true 只标对问题的直接回答/核心结论（通常 1-3 条）；"
        "扩展解释、举例、生活建议等细节一律 core=false；"
        "2. 证据没有提到的具体数字、因果、结论，一律 supported=false；"
        "3. 只陈述常识但证据未提及，也算 false；"
        "4. support_ratio 只按 core=true 的主张计算：支持的核心主张数 / 核心主张总数；"
        "5. consistent 表示回答结论是否与证据结论冲突；"
        "6. 不要用你自己的知识补全证据，只以给定证据为准。"
    )

    def run(self, query: str, answer: str, materials: list,
            verdict: str = None, mock_override: bool = False) -> dict:
        """返回 {support_ratio, claims, consistent, reason, mode, unsupported_claims}。"""
        if not answer or not materials:
            return {
                "support_ratio": 0.0 if answer else None,
                "detail_support_ratio": None,
                "core_claims": 0,
                "claims": [],
                "unsupported_claims": [],
                "consistent": True,
                "reason": "无回答或无证据，跳过校验",
                "mode": "skipped",
            }

        if mock_override or llm.provider == "mock":
            return self._heuristic(answer, materials, verdict)

        evidence = "\n".join(
            f"- 《{m.get('title', '')}》 结论：{m.get('truth') or m.get('answer', '')}"
            f"｜来源：{m.get('source', '')}"
            for m in materials
        )
        user = (
            f"问题：{query}\n\nAI回答：{answer}\n\n"
            f"模型给出的结论：{verdict or '未给出'}\n\n检索证据：\n{evidence}"
        )
        data = _parse_json(llm.chat(self.SYSTEM_PROMPT, user, temperature=0), None)

        if not isinstance(data, dict):
            return self._heuristic(answer, materials, verdict,
                                   reason="校验模型输出解析失败，已用启发式")

        claims = data.get("claims") or []
        claims = [c for c in claims if isinstance(c, dict)]
        # 核心主张才是判拒答的依据；解释性扩展单独统计，避免长回答被细节拖垮
        core = [c for c in claims if c.get("core") is True]
        basis = core or claims
        supported = [c for c in basis if c.get("supported") is True]
        ratio = (len(supported) / len(basis)) if basis else None
        detail = [c for c in claims if c.get("core") is not True]
        detail_ratio = None
        if detail:
            detail_ratio = len([c for c in detail if c.get("supported") is True]) / len(detail)

        return {
            "support_ratio": ratio,
            "detail_support_ratio": detail_ratio,
            "core_claims": len(core),
            "claims": claims[:20],
            "unsupported_claims": [c.get("claim", "") for c in basis
                                   if c.get("supported") is not True][:10],
            "consistent": bool(data.get("consistent", True)),
            "reason": data.get("reason", ""),
            "mode": "llm",
        }

    def _heuristic(self, answer: str, materials: list,
                   verdict: str = None, reason: str = "") -> dict:
        """无模型时的兜底：回答 bigram 被证据覆盖的比例。"""
        evidence_text = " ".join(
            str(m.get("truth", "")) + " " + str(m.get("answer", "")) + " "
            + str(m.get("preview", ""))
            for m in materials
        )
        ans = _bigrams(answer)
        ev = _bigrams(evidence_text)
        ratio = (len(ans & ev) / len(ans)) if ans else None

        consistent = True
        if verdict:
            top_verdict = materials[0].get("verdict")
            if top_verdict and verdict in ("true", "fake", "partial"):
                top_verdict = "fake" if top_verdict == "false" else top_verdict
                consistent = (top_verdict == verdict)

        return {
            "support_ratio": ratio,
            "detail_support_ratio": None,
            "core_claims": 0,
            "claims": [],
            "unsupported_claims": [],
            "consistent": consistent,
            "reason": reason or "mock/启发式校验：按回答与证据的 bigram 覆盖率估算",
            "mode": "heuristic",
        }
