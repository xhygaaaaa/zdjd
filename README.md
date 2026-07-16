# 真的假的?! —— AI 冷知识 & 辟谣探索机

> 一个基于**多 Agent 协同**与 **RAG 检索**的 AI 科普问答系统。用户既能探索有趣的冷知识，也能验证网络流传说法的真假，并追溯误解的起源。
>
> 前端复刻科技感深色 UI，后端用 FastAPI 编排多个职责单一的 Agent，内置 200 条精选知识库，支持三级置信度防幻觉机制。

![tech](https://img.shields.io/badge/backend-FastAPI-009688) ![llm](https://img.shields.io/badge/LLM-MiniMax%20%7C%20DeepSeek-orange) ![arch](https://img.shields.io/badge/architecture-Multi--Agent%20%2B%20RAG-blue)

---

## ✨ 项目亮点

| 能力 | 说明 | 对应代码 |
|------|------|---------|
| **多 Agent 协同** | 总调度 Agent 判断问题类型，动态分发到「冷知识链路」或「辟谣链路」 | `orchestrator.py` |
| **RAG 检索** | 本地精选知识库 + 关键词加权检索，命中优先用高质量结构化答案 | `knowledge_base.py` |
| **误解双重溯源** | 辟谣时不仅给真相，还追溯「误解从哪来」「为什么流传」（核心差异化） | `agents.py::MisconceptionOriginAgent` |
| **三级置信度防幻觉** | 🟢确证 / 🟡趣味参考 / 🔴存疑，无可靠资料时诚实拒答 | `agents.py::ReviewAgent` |
| **可插拔大模型** | 默认 mock 模式开箱即跑；填 Key 即接入 MiniMax，一行配置切 DeepSeek | `llm_client.py` |

---

## 🏗️ 系统架构

```
                          用户提问
                             │
                    ┌────────▼─────────┐
                    │  Orchestrator    │  判断：冷知识 or 辟谣？
                    │  (总调度 Agent)   │
                    └────────┬─────────┘
              ┌──────────────┴──────────────┐
       冷知识链路                        辟谣链路
              │                             │
     检索→科普→溯源→延伸           检索→辟谣→误解溯源→溯源→延伸
              └──────────────┬──────────────┘
                    ┌────────▼─────────┐
                    │   审查 Agent      │  评估置信度，防幻觉
                    └────────┬─────────┘
                             │
              答案 + 真相溯源 + 误解溯源 + 相关推荐
```

> 架构完整对应 [PRD 文档](./docs/PRD.md) 第五章。为控制真实 API 成本，执行层将 8 个 Agent 合并为 2-3 次大模型调用，但每个 Agent 的职责与 Prompt 均独立定义，便于讲解与横向扩展。

---

## 📁 项目结构

```
zhende-jiade/
├── frontend/
│   └── index.html          # 前端页面（三栏布局 + 搜索问答 + 分类浏览）
├── backend/
│   ├── main.py             # FastAPI 入口，3 个 REST 接口
│   ├── orchestrator.py     # 总调度 Agent：路由判断 + 链路编排
│   ├── agents.py           # 检索/科普/辟谣/溯源/审查等 Agent
│   ├── llm_client.py       # 大模型客户端（mock / MiniMax / DeepSeek）
│   ├── knowledge_base.py   # RAG 检索
│   └── data/
│       ├── knowledge.json        # 基础知识库
│       └── knowledge_extra.json  # 扩充知识库（合计约 200 条）
├── docs/
│   └── PRD.md              # 完整产品需求文档
├── requirements.txt
├── .env.example            # 大模型配置模板
└── README.md
```

---

## 🚀 快速开始

### 1. 安装依赖
```bash
cd backend
pip install -r ../requirements.txt
```

### 2. 启动服务（默认 mock 模式，无需 API Key）
```bash
uvicorn main:app --reload --port 8000
```

### 3. 打开浏览器
访问 **http://localhost:8000/** 即可使用。

### 4. （可选）接入真实大模型
复制 `.env.example` 为 `.env`，填入密钥：
```bash
LLM_PROVIDER=minimax
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://api.minimax.chat/v1
LLM_MODEL=abab6.5s-chat
```
重启服务即可切换到真实大模型。切 DeepSeek 只需改 4 个环境变量。

---

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/categories` | 分类列表及数量 |
| GET  | `/api/knowledge/{category}` | 按分类取知识卡片 |
| POST | `/api/ask` | 多 Agent 问答（核心） |

**请求示例**
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "鱼的记忆只有7秒吗？"}'
```

**响应示例**
```json
{
  "route": "debunk",
  "verdict": "fake",
  "answer": "鱼的记忆可达数月……",
  "truth": "鱼的记忆可达数月甚至更长",
  "origin": "被《海底总动员》多莉角色强化传播",
  "why": "简单有趣的说法更易传播",
  "source": "维基百科 · 动物行为学研究",
  "confidence": "high",
  "confidence_label": "🟢 确证 · 有权威来源",
  "related": ["🐙 章鱼有3个心脏和蓝色血液", "..."],
  "route_hit": 1
}
```

---

## 🧠 技术要点（面试可讲）

1. **为什么用多 Agent 而非单次 Prompt？**
   任务拆分让每个环节职责单一、Prompt 更聚焦，也便于针对性优化和替换（如单独升级辟谣 Agent）。总调度按问题类型动态编排链路，冷知识和辟谣走不同路径。

2. **如何防止大模型幻觉？**
   三道防线：① 强制基于检索资料回答；② 审查 Agent 标注三级置信度；③ 无可靠来源时诚实拒答而非编造。

3. **知识库为什么不预存全部？**
   采用「精选库 + 检索 + 主题轮换」策略——本地存高质量精选，命中优先返回结构化答案；未命中降级到大模型生成，体现 RAG「按需检索」的核心思想。

4. **成本与效果如何权衡？**
   8 个 Agent 的完整链路在执行层合并为 2-3 次调用：检索/翻译/溯源/延伸靠本地完成，只有生成和审查消耗 LLM，兼顾架构清晰度与调用成本。

---

## 📄 License

MIT — 仅供学习与作品展示。知识库内容标注了来源与置信度，部分标记为「趣味参考」，请勿作为严肃学术依据。
