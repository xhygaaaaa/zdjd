"""辟谣系统离线评测（可复现）

指标（对应 PRD「有据可查率 ≥ 80%」）：
  - source_rate     来源标注率：回答是否带非空来源 + 证据链；
  - grounding_rate  有据可查率：生成后校验 support_ratio ≥ 0.5 的占比；
  - citation_rate   证据链完整率：每条证据都有可读片段（有 URL 时也必须非空）；
  - verdict_accuracy 辟谣题的 verdict 与知识库标注一致率；
  - refusal_accuracy 超纲题正确拒答率。

用法：
    cd backend
    python evaluate.py --limit 12 --no-web
    python evaluate.py --limit 12 --no-web --out evaluation/results/run_xxx
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime

# Windows 控制台默认 GBK，标题里的 emoji 会让 print 直接崩
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from knowledge_base import kb

# 明显超出本地知识库范围的问题：正确行为是拒答
OUT_OF_SCOPE = [
    "南京联成的主营业务是什么？",
    "明天南京会下雨吗？",
    "怎么挑选一支股票？",
    "人类什么时候能移民火星？",
    "2026 年世界杯冠军是谁？",
    "如何申请护照？",
]

GROUNDING_THRESHOLD = 0.5


def _norm_verdict(v):
    return "fake" if v == "false" else v


def build_cases(limit: int, seed: int = 42):
    rng = random.Random(seed)
    debunk = [it for it in kb.items if it.get("type") == "debunk" and it.get("title")]
    knowledge = [it for it in kb.items
                 if it.get("type") != "debunk" and it.get("title")]

    n_each = max(1, limit // 2)
    picked_debunk = rng.sample(debunk, min(n_each, len(debunk)))
    picked_knowledge = rng.sample(knowledge, min(n_each, len(knowledge)))

    cases = []
    for it in picked_debunk:
        title = it["title"].rstrip("？? ")
        # 标题多为陈述句（“吞下口香糖会在胃里待7年”），
        # 评测时补成求证句式，才能触发 IntentAgent 的 verify 路由。
        question = title if title.endswith("吗") else f"{title}是真的吗？"
        cases.append({
            "kind": "debunk",
            "question": question,
            "expected_verdict": _norm_verdict(it.get("verdict")),
            "expected_type": "debunk",
            "item_id": it.get("id"),
        })
    for it in picked_knowledge:
        title = it["title"].rstrip("？? ")
        # 用「这是什么原理」固定触发 explain 路由，避免陈述句标题被模型判成 verify
        cases.append({
            "kind": "knowledge",
            "question": f"{title}，这是什么原理？",
            "expected_verdict": None,
            "expected_type": "knowledge",
            "item_id": it.get("id"),
        })
    if limit >= 6:
        cases.extend({"kind": "out_of_scope", "question": q,
                      "expected_verdict": None, "expected_type": "none",
                      "item_id": None} for q in OUT_OF_SCOPE[:max(2, limit // 4)])
    return cases


def score_case(case: dict, result: dict) -> dict:
    refused = bool(result.get("refused"))
    chain = result.get("evidence_chain") or []
    verification = result.get("verification") or {}
    support = verification.get("support_ratio")

    if case["kind"] == "out_of_scope":
        return {
            "passed": refused,
            "refusal_correct": refused,
            "source_ok": None,
            "grounded": None,
            "citation_ok": None,
            "verdict_ok": None,
        }

    source_ok = bool(result.get("source")) and result.get("source") != "暂无可靠来源"
    grounded = (support is not None and support >= GROUNDING_THRESHOLD)
    citation_ok = bool(chain) and all(e.get("snippet") for e in chain)
    verdict_ok = None
    if case.get("expected_verdict"):
        verdict_ok = (result.get("verdict") == case["expected_verdict"])

    # 被生成后校验改判拒答的，视为「不通过有据可查」
    if refused:
        grounded = False

    return {
        "passed": bool(source_ok and grounded),
        "refusal_correct": None,
        "source_ok": source_ok,
        "grounded": grounded,
        "citation_ok": citation_ok,
        "verdict_ok": verdict_ok,
        "support_ratio": support,
    }


def summarize(rows: list) -> dict:
    def rate(key, subset=None):
        vals = [r["score"][key] for r in rows
                if (subset is None or r["case"]["kind"] == subset)
                and r["score"].get(key) is not None]
        return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None

    in_scope = [r for r in rows if r["case"]["kind"] != "out_of_scope"]
    return {
        "n": len(rows),
        "in_scope_n": len(in_scope),
        "out_of_scope_n": len(rows) - len(in_scope),
        "source_rate": rate("source_ok"),
        "grounding_rate": rate("grounded"),
        "verifiable_rate": rate("passed"),
        "citation_rate": rate("citation_ok"),
        "verdict_accuracy": rate("verdict_ok", "debunk"),
        "refusal_accuracy": rate("refusal_correct", "out_of_scope"),
        "in_scope_refused": sum(1 for r in in_scope
                                if r["result"].get("refused")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-web", action="store_true",
                        help="禁用联网搜索，评测本地链路（推荐，结果稳定）")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.no_web:
        # 评测本地链路时屏蔽联网，避免网络波动污染指标
        import orchestrator as orch_module
        orch_module.web_search.search = lambda *a, **k: []

    from orchestrator import orchestrator  # noqa: E402

    cases = build_cases(args.limit, args.seed)
    rows = []
    for i, case in enumerate(cases, 1):
        result = orchestrator.handle(case["question"])
        score = score_case(case, result)
        rows.append({"case": case, "result": result, "score": score})
        print(f"[{i:>2}/{len(cases)}] {case['kind']:<12} "
              f"passed={score['passed']} "
              f"support={score.get('support_ratio')} :: {case['question'][:26]}")

    summary = summarize(rows)
    out_dir = args.out or os.path.join(
        os.path.dirname(__file__), "evaluation", "results",
        datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已写入: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
