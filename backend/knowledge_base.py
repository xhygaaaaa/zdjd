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
        """极简中文分词：去标点后按字 + 保留英文单词。"""
        text = re.sub(r"[^\w一-鿿]", " ", text.lower())
        tokens = set()
        for word in text.split():
            if re.match(r"[a-z0-9]+", word):
                tokens.add(word)          # 英文/数字整体作为一个 token
            else:
                tokens.update(list(word))  # 中文按单字
        return tokens

    def search(self, query: str, top_k: int = 3):
        """
        检索最相关的知识条目。

        Returns:
            List[dict]：按相关度降序的知识条目（含 _score 字段）
        """
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scored = []
        for item in self.items:
            title_tokens = self._tokenize(item.get("title", ""))
            body_tokens = self._tokenize(
                item.get("preview", "") + item.get("answer", "") + item.get("truth", "")
            )
            # 标题命中权重更高
            title_hit = len(q_tokens & title_tokens) * 3
            body_hit = len(q_tokens & body_tokens)
            score = title_hit + body_hit
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, item in scored[:top_k]:
            r = dict(item)
            r["_score"] = score
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
