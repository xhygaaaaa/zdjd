"""
FastAPI 服务入口
==================
对应 PRD 11.2「实现路径 阶段2：自建前端 + 接 API」。

提供 3 个接口：
- GET  /api/categories          分类列表（首页左侧导航）
- GET  /api/knowledge/{cat}     按分类取知识卡片（首页中间卡片）
- POST /api/ask                 多 Agent 问答（核心接口）

运行：
    cd backend
    pip install -r ../requirements.txt
    uvicorn main:app --reload --port 8000
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 加载 .env（可选依赖，没装也能跑）
# 注意：服务从 backend/ 启动，而 .env 在其上一级目录 zhende-jiade/，
# python-dotenv 默认只在当前工作目录找，必须显式指定路径，否则密钥读不到会退回 mock。
try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(_ENV_PATH)
except ImportError:
    pass

from knowledge_base import kb
from orchestrator import orchestrator

app = FastAPI(title="真的假的?! AI冷知识&辟谣探索机", version="1.0")

# 允许前端本地直接访问（开发环境放开跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# PRD 4.3 主题轮换：分类元数据
CATEGORIES = [
    {"key": "ocean", "name": "海洋动物", "icon": "🐙"},
    {"key": "space", "name": "宇宙天文", "icon": "🌌"},
    {"key": "body", "name": "人体奥秘", "icon": "🧠"},
    {"key": "plant", "name": "植物世界", "icon": "🌱"},
    {"key": "history", "name": "历史冷知识", "icon": "🏛️"},
    {"key": "food", "name": "食物真相", "icon": "🍯"},
    {"key": "debunk", "name": "经典辟谣", "icon": "🔍"},
]


class AskRequest(BaseModel):
    query: str


@app.get("/api/categories")
def get_categories():
    """返回分类列表及每类数量。"""
    result = []
    for c in CATEGORIES:
        c2 = dict(c)
        c2["count"] = len(kb.get_by_category(c["key"]))
        result.append(c2)
    return {"categories": result}


@app.get("/api/knowledge/{category}")
def get_knowledge(category: str):
    """按分类返回知识卡片列表。"""
    return {"items": kb.get_by_category(category)}


@app.post("/api/ask")
def ask(req: AskRequest):
    """
    核心问答接口：调用 Orchestrator 走多 Agent 链路。
    返回结构见 orchestrator.handle 的文档。
    """
    if not req.query.strip():
        return {"error": "问题不能为空"}
    result = orchestrator.handle(req.query.strip())
    result["provider"] = orchestrator_provider()
    return result


def orchestrator_provider():
    from llm_client import llm
    return llm.provider


# ---------------------------------------------------------------------------
# 托管前端静态文件：访问 http://localhost:8000/ 直接打开页面
# ---------------------------------------------------------------------------
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")
