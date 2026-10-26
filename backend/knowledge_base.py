"""
知识库检索模块（RAG 的 Retrieval 部分）
==========================================
对应 PRD 第四章「知识库策略：实时联网 + 主题轮换」与检索 Agent 的检索来源。

三层知识获取机制中的第①层：本地精选知识库。
本模块负责从本地 JSON 知识库中，根据用户问题检索出最相关的条目。

检索策略（轻量实现，无需向量数据库）：
- 中文分词后做关键词重叠打分 + 标题/预览加权
- 命中则返回 Top-K，供下游 Agent 生成答案
- 未命中返回空，触发「联网检索」或「拒答」（见 PRD 4.2）
"""

import json
import os
import re

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 中文功能词/停用词：几乎每条知识都含这些字，用于检索打分会造成大量误匹配
# （例如“屎是苦的吗”里的“是/的/吗”会命中无关条目）。查询分词时剔除它们。
_STOPWORDS = set(
    "的了吗呢吧啊呀哦嗯是不在和与会我你他她它这那也都就而能个"
    "上下中为以之其等及或且被把让给到从对向于但却只还又再很太更最"
    "没可真假么什怎样着过要想说问"
)


def _core_text(text: str) -> str:
    """去掉标点和功能词，留下“实义字串”，用于判断查询与标题是否近似原句。"""
    text = re.sub(r"[^\w一-鿿]", "", text.lower())
    return "".join(ch for ch in text if ch not in _STOPWORDS)


def _load_all():
    """加载并合并所有知识库文件（基础库 + 扩充库）。"""
    items = []
    for fname in ("knowledge.json", "knowledge_extra.json"):
        path = os.path.join(_DATA_DIR, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                items.extend(json.load(f))
    return items


class KnowledgeBase:
    def __init__(self):
        self.items = _load_all()
        print(f"[KnowledgeBase] 已加载 {len(self.items)} 条知识")

    def _tokenize(self, text: str):
        """
        轻量中文分词：二元组（bigram）切分 + 英文/数字整体 token。

        为什么不用单字（unigram）：单字会把"蓝色"拆成「蓝」「色」两个独立字，
        导致"天空是蓝色的"误命中"血液是蓝色的"（都含这两个字）。这是知识库外
        自由问题被误判为"命中"的根因。

        为什么过滤含功能字的 bigram：切字会切出"是蓝""色的""么天"这类跨词边界
        的垃圾组合，必须丢弃。只保留两个字符都是实义字的 bigram。
        """
        text = re.sub(r"[^\w一-鿿]", " ", text.lower())
        tokens = set()
        for word in text.split():
            if re.match(r"[a-z0-9]+", word):
                tokens.add(word)          # 英文/数字整体作为一个 token
                continue
            chars = list(word)
            if len(chars) == 1:
                if chars[0] not in _STOPWORDS:
                    tokens.add(chars[0])
                continue
            for i in range(len(chars) - 1):
                bigram = chars[i] + chars[i + 1]
                if bigram[0] in _STOPWORDS or bigram[1] in _STOPWORDS:
                    continue              # 过滤含功能字的跨词垃圾 bigram
                tokens.add(bigram)
        return tokens

    def search(self, query: str, top_k: int = 3):
        """
        检索最相关的知识条目。

        Returns:
            List[dict]：按相关度降序的知识条目（含 _score 字段）
        """
        # 剔除停用词后再打分：否则“是/的/吗”这类高频功能词会让几乎所有条目都命中
        q_tokens = self._tokenize(query) - _STOPWORDS
        # 有效关键词个数（用于后面的相对阈值判断）
        n_q = len(q_tokens)
        if n_q == 0:
            return []

        q_core = _core_text(query)
        scored = []
        for item in self.items:
            title = item.get("title", "")
            title_tokens = self._tokenize(title)
            body_tokens = self._tokenize(
                item.get("preview", "") + item.get("answer", "") + item.get("truth", "")
            )
            title_overlap = len(q_tokens & title_tokens)
            body_overlap = len(q_tokens & body_tokens)
            # 标题命中权重更高
            title_hit = title_overlap * 3
            body_hit = body_overlap
            score = title_hit + body_hit

            # 近似原句命中：问题骨架（去功能词）出现在标题里时给强加成。
            # 解决“地球是平的吗”这类短问题只剩 1 个 bigram、被泛化条目抢到 top-1。
            title_core = _core_text(title)
            exact = False
            if q_core and len(q_core) >= 2 and q_core in title_core:
                exact = True
                score += 6
                if q_core == title_core:          # 问题骨架与标题骨架完全一致
                    score += 4
                elif title_core.startswith(q_core):
                    score += 2

            # 相关性门槛（Bug C 修复）：改用 bigram 后，要求「标题命中 ≥2 个
            # 独立实义 bigram」才算强命中，或正文覆盖足够多查询关键词。
            # 实测：真实命中 title_bi≥2（如"章鱼有几个心脏"→命中"章鱼""心脏"），
            # 误命中 title_bi≤1（如"天空是蓝色"只命中"蓝色"1 个）——门槛定为 2 可彻底区分。
            # 短查询（有效 token 只有 1 个）放宽门槛：例如“地球是平的吗”经停用词
            # 过滤后只剩「地球」，若仍要求标题命中 ≥2 就会漏掉正确答案。
            if n_q == 1:
                strong = title_overlap >= 1
            else:
                strong = (
                    title_overlap >= 2
                    or body_overlap >= max(3, n_q)
                )
            if strong and score > 0:
                scored.append((score, item, title_overlap, body_overlap, exact))

        # 同分时依次比较标题覆盖、正文覆盖，避免稳定地让错误条目排第一。
        # （“人的血液是蓝色的吗”曾与章鱼条目同分 8，旧代码按文件顺序取到章鱼。）
        scored.sort(key=lambda x: (x[0], x[2], x[3]), reverse=True)
        results = []
        for score, item, title_overlap, body_overlap, exact in scored[:top_k]:
            r = dict(item)
            r["_score"] = score
            r["_title_overlap"] = title_overlap
            r["_body_overlap"] = body_overlap
            r["_n_q"] = n_q
            r["_exact"] = exact
            results.append(r)
        return results

    def get_by_category(self, category: str):
        """按分类返回全部条目（用于首页卡片展示）。"""
        return [it for it in self.items if it.get("category") == category]

    def get_by_id(self, item_id: int):
        for it in self.items:
            if it.get("id") == item_id:
                return it
        return None


# 单例
kb = KnowledgeBase()
