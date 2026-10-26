"""
联网搜索 Agent（本地知识库兜底）
==================================
本地知识库没有可靠依据时，走这条路：先联网搜，再基于搜索结果组织回答。

设计原则（防幻觉）：
1. **只在本地检索为空时触发**，不让搜索结果盖过精心整理的本地条目。
2. 回答必须基于搜到的片段，并要求标出 [1][2] 编号。
3. 有学术/权威来源 → 置信度高；只有普通网页 → 明确写“无法确认来源”。
4. 搜索链路**不下真假结论**（verdict 恒为 None），因为它没有经过人工核校。
5. 搜到的结果必须先过**相关性校验**：标题/摘要里连一个查询词都不出现，
   判定为搜索引擎跑偏，直接丢弃——否则会出现“搜南京联成，返回南京市旅游”
   却因为域名是 gov.cn 而误判为“权威来源”的荒唐情况。

为什么爬 HTML 而不是接搜索 API：
- 演示环境不需要额外 key，克隆即可跑；
- 生产替换成 Bing Web Search / Tavily / Serper 只需要改 `_providers`。
"""

import html
import re
import urllib.parse

import requests

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# 权威域名（论文库 / 政府 / 高校 / 国际组织）。命中即视为“有来源支撑”。
AUTHORITATIVE_DOMAINS = (
    "nature.com", "science.org", "sciencedirect.com", "cell.com",
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "thelancet.com",
    "bmj.com", "nejm.org", "springer.com", "ieee.org", "arxiv.org",
    "who.int", "cdc.gov", "nih.gov", "un.org",
    "cnki.net", "wanfangdata.com.cn", "cqvip.com",
    "gov.cn", "edu.cn", "ac.cn", "cas.cn",
)

# 提问句式里对检索没帮助的部分，搜之前先删掉
_QUERY_NOISE = (
    "请问", "我想知道", "我想问", "告诉我", "帮我", "查一下", "搜一下",
    "是什么", "是谁", "是多少", "是多久", "怎么样", "怎么回事", "为什么",
    "怎么", "如何", "有哪些", "有什么", "多少", "多久", "吗", "呢",
    "？", "?", "。", "！", "!",
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[一-鿿]{2,}")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub("", text))).strip()


def _bigrams(text: str) -> set:
    """查询词 → 二元组，用于判断搜索结果是不是真的和问题相关。"""
    s = re.sub(r"[^\w一-鿿]", "", text.lower())
    return {s[i:i + 2] for i in range(len(s) - 1)}


class WebSearchAgent:
    """最小可用的网页搜索：百度 HTML 优先，Bing 兜底。"""

    UI_NOISE = ("扫码下载", "百度App", "登录", "注册", "下载百度")

    def __init__(self, top_k: int = 5, timeout: float = 6.0):
        self.top_k = top_k
        self.timeout = timeout

    # ------------------------------------------------------------------
    @staticmethod
    def build_query(query: str) -> str:
        """去掉疑问句式，留下实体和关键词，搜索引擎更容易命中。"""
        q = query.strip()
        for noise in _QUERY_NOISE:
            q = q.replace(noise, "")
        q = re.sub(r"\s+", " ", q).strip(" 的")
        return q or query

    # ------------------------------------------------------------------
    def _get(self, url: str) -> str:
        # 必须把空格编码成 %20：requests 的 params 会把空格转成 '+',
        # Bing 会把 '+' 当成字面字符，于是返回完全不相干的结果。
        return requests.get(
            url,
            headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"},
            timeout=self.timeout,
        ).text

    def _baidu(self, q: str) -> str:
        return self._get("https://www.baidu.com/s?wd=" + urllib.parse.quote(q))

    def _bing(self, q: str) -> str:
        return self._get("https://cn.bing.com/search?q=" + urllib.parse.quote(q))

    def _sogou(self, q: str) -> str:
        return self._get("https://www.sogou.com/web?query=" + urllib.parse.quote(q))

    def _so360(self, q: str) -> str:
        return self._get("https://www.so.com/s?q=" + urllib.parse.quote(q))

    # ------------------------------------------------------------------
    def _parse_baidu(self, page: str) -> list:
        out = []
        for m in re.finditer(r"<h3[^>]*>(.*?)</h3>", page, re.S):
            block = m.group(1)
            a = re.search(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
            if not a:
                continue
            url, title = a.group(1), _clean(a.group(2))
            if not url.startswith("http") or not title:
                continue
            if any(k in title for k in self.UI_NOISE):
                continue
            # 摘要：h3 之后的一小段正文，够用即可
            tail = page[m.end():m.end() + 1200]
            snip = _clean(re.sub(r"<script.*?</script>", "", tail, flags=re.S))[:220]
            out.append({"title": title, "url": url, "snippet": snip})
        return out

    def _parse_bing(self, page: str) -> list:
        out = []
        for block in re.findall(r'<li class="b_algo".*?</li>', page, re.S):
            m = re.search(r'<h2>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
            if not m:
                continue
            url, title = m.group(1), _clean(m.group(2))
            if not url.startswith("http") or not title:
                continue
            sm = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
            snip = _clean(sm.group(1))[:220] if sm else ""
            out.append({"title": title, "url": url, "snippet": snip})
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve(url: str, timeout: float = 3.0) -> str:
        """
        搜索引擎给的是 /link?url= 跳转链接，域名永远是搜索引擎自己，
        没法判断来源权威性，用户点开也多一跳。这里跟一次重定向拿真实地址。
        """
        try:
            resp = requests.get(
                url, headers={"User-Agent": UA}, timeout=timeout,
                allow_redirects=True, stream=True,
            )
            final = resp.url
            resp.close()
            return final or url
        except Exception:
            return url

    @staticmethod
    def _is_authoritative(url: str) -> bool:
        host = urllib.parse.urlparse(url).netloc.lower()
        return any(host == d or host.endswith("." + d) for d in AUTHORITATIVE_DOMAINS)

    @staticmethod
    def _relevant(item: dict, query_bigrams: set, raw_query: str) -> bool:
        """
        相关性闸门：标题/摘要里至少出现一个查询二元组。
        短查询（≤4 字）只要求整句或单个词出现，否则太严。
        """
        if not query_bigrams:
            return True
        text = (item.get("title", "") + item.get("snippet", "")).lower()
        hits = sum(1 for bg in query_bigrams if bg in text)
        need = 1 if len(query_bigrams) <= 3 else 2
        return hits >= need

    # ------------------------------------------------------------------
    def _pubmed(self, q: str, limit: int = 3) -> list:
        """
        PubMed 免费接口（无需 key）：中文网页几乎不会落在学术域名上，
        所以要真正实现“有论文支撑 = 高置信度”，必须单独查一次论文库。
        """
        try:
            r = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": q, "retmode": "json", "retmax": limit},
                timeout=self.timeout,
            )
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            s = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
                timeout=self.timeout,
            )
            data = s.json().get("result", {})
        except Exception:
            return []

        # 只有标题和期刊名，模型没法据此回答，必须把摘要取回来
        abstracts = self._pubmed_abstracts(ids[:2])

        out = []
        for pid in ids:
            info = data.get(pid, {})
            title = (info.get("title") or "").strip()
            if not title:
                continue
            journal = info.get("fulljournalname") or info.get("source") or ""
            year = (info.get("pubdate") or "")[:4]
            abstract = abstracts.get(pid, "")
            snippet = abstract or f"PubMed 收录论文 · {journal} {year}"
            out.append({
                "title": f"{title}（{journal} {year}）".strip(),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                "snippet": snippet,
                "authoritative": True,
            })
        return out

    def _pubmed_abstracts(self, ids: list) -> dict:
        """efetch 取摘要纯文本，按 PMID 分块；失败返回空字典，不影响主流程。"""
        try:
            text = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
                timeout=self.timeout + 3,
            ).text
        except Exception:
            return {}

        # text 模式不带 PMID 标记，没法把摘要和结果对上，所以用 XML 解析
        result = {}
        for record in re.split(r"<PubmedArticle>", text)[1:]:
            pmid = re.search(r"<PMID[^>]*>(\d+)</PMID>", record)
            if not pmid:
                continue
            parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", record, re.S)
            body = _WS_RE.sub(" ", _clean(" ".join(parts))).strip()
            if body:
                result[pmid.group(1)] = body[:900]
        return result

    def _parse_sogou(self, page: str) -> list:
        out = []
        for m in re.finditer(r"<h3[^>]*>(.*?)</h3>", page, re.S):
            a = re.search(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', m.group(1), re.S)
            if not a:
                continue
            url, title = a.group(1), _clean(a.group(2))
            if title and len(title) >= 4 and "搜狗" not in title:
                tail = _clean(page[m.end():m.end() + 900])[:220]
                out.append({"title": title, "url": url, "snippet": tail})
        return out

    # ------------------------------------------------------------------
    def search(self, query: str, en_query: str = "") -> list:
        """
        返回相关搜索结果；所有引擎都跑偏或网络异常时返回空列表（上层据此拒答）。

        搜索引擎对长问句很不友好（“南京联成的主营业务是什么”会返回旅游攻略），
        所以调用方应先把问题改写成短检索式（见 QueryRewriteAgent）。
        """
        q = self.build_query(query)
        query_bigrams = _bigrams(q)

        # 论文库优先：命中就是最高质量的来源，且不受中文相关性闸门限制
        papers = self._pubmed(en_query) if en_query else []
        results = list(papers)
        if len(results) >= self.top_k:
            return results[:self.top_k]

        # Bing 排第一：它给的是直达链接，引用列表里用户点开就是原文；
        # 搜狗/360 给的是 /link?url= 跳转链接，域名永远是搜索引擎自己，无法判断权威性。
        providers = (
            (self._bing, self._parse_bing, "https://cn.bing.com"),
            (self._so360, self._parse_sogou, "https://www.so.com"),
            (self._sogou, self._parse_sogou, "https://www.sogou.com"),
            (self._baidu, self._parse_baidu, "https://www.baidu.com"),
        )

        seen = {r["url"] for r in results}
        for fetch, parse, base in providers:
            try:
                raw = parse(fetch(q))
            except Exception:
                continue
            for item in raw:
                url = html.unescape(item["url"])
                if url.startswith("/"):
                    url = base + url
                item["url"] = url
                if url in seen:
                    continue
                seen.add(url)
                if not self._relevant(item, query_bigrams, q):
                    continue
                # 只解析前 3 条的跳转链接，避免每条都多一次网络往返
                host = urllib.parse.urlparse(url).netloc.lower()
                if len(results) < 3 and any(
                    h in host for h in ("so.com", "sogou.com", "baidu.com")
                ):
                    url = self._resolve(url)
                    item["url"] = url
                item["authoritative"] = self._is_authoritative(url)
                results.append(item)
                if len(results) >= self.top_k:
                    return results
            # 这个引擎已经给了足够多的相关结果就不再换源
            if len(results) >= 2:
                break
        return results


web_search = WebSearchAgent()
