# ==========================================
# api.py - NMAP-AI Unified API (FastAPI)
# Comprehension + Complexity + Easy(RAG Neo4j)
# Single server: http://localhost:8000
# ==========================================

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Easy (RAG / Neo4j) ---
from Agents.Agent_easy.rag_engine import NeoConnection, NmapRAGPipeline, IntentClassifier

# --- Complexity ---
from Agents.Agent_complexity.complexity_slm_word2vec import ComplexityClassifierSLM

# --- Comprehension ---
from Agents.Agent_comprehension.nmap_agent_embeddings import NMAPEmbeddingAgent
from Agents.Agent_hard.agent_diffusion import HardDiffusionAgent


# =========================
# Models
# =========================

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    nodes_in_graph: Optional[int] = None
    agents: Dict[str, str]
    timestamp: str


# =========================
# Globals (state)
# =========================

neo_connection: Optional[NeoConnection] = None
rag_pipeline: Optional[NmapRAGPipeline] = None
complexity_agent: Optional[ComplexityClassifierSLM] = None
comprehension_agent: Optional[NMAPEmbeddingAgent] = None
hard_agent: Optional[HardDiffusionAgent] = None


def _abs_path_from_backend(*parts: str) -> str:
    """Build absolute path relative to backend/ directory (api.py location)."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(backend_dir, *parts))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle (no deprecated on_event)."""
    global neo_connection, rag_pipeline, complexity_agent, comprehension_agent, hard_agent

    print("\n" + "=" * 70)
    print("🚀 NMAP-AI Unified API - Starting")
    print("=" * 70)

    # 1) Init Neo4j + RAG
    try:
        neo_connection = NeoConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        rag_pipeline = NmapRAGPipeline(neo_connection.driver)
        print("✅ Neo4j + RAG ready")
    except Exception as exc:
        print(f"❌ Neo4j/RAG init failed: {exc}")
        raise

    # 2) Init Comprehension Agent
    try:
        domain_path = _abs_path_from_backend("Agents", "Agent_comprehension", "nmap_domain.txt")
        if not os.path.exists(domain_path):
            raise FileNotFoundError(f"nmap_domain.txt not found at: {domain_path}")
        comprehension_agent = NMAPEmbeddingAgent(domain_path)
        print("✅ Comprehension agent ready")
    except Exception as exc:
        print(f"❌ Comprehension init failed: {exc}")
        comprehension_agent = None  # keep API running; endpoint will return 503

    # 3) Init Complexity Agent
    try:
        complexity_agent = ComplexityClassifierSLM()
        complexity_agent.train()
        print("✅ Complexity agent ready")
    except Exception as exc:
        print(f"❌ Complexity init failed: {exc}")
        complexity_agent = None

    # 4) Init Hard Diffusion Agent
    try:
        hard_agent = HardDiffusionAgent(seed=42)
        print("Hard diffusion agent ready")
    except Exception as exc:
        print(f"Hard diffusion init failed: {exc}")
        hard_agent = None

    print("📚 Swagger UI: http://localhost:8000/docs")
    print("=" * 70 + "\n")

    yield

    # Shutdown
    if neo_connection:
        try:
            neo_connection.close()
            print("✅ Neo4j connection closed")
        except Exception:
            pass


app = FastAPI(
    title="NMAP-AI Unified API",
    description="Comprehension + Complexity + RAG(Easy) under one FastAPI server.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (keep open for frontend dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Helpers
# =========================

def _require(agent: Any, name: str):
    if agent is None:
        raise HTTPException(status_code=503, detail=f"{name} not initialized")


# =========================
# Routes
# =========================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "NMAP-AI Unified API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/comprehension", "/complexity", "/generate"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    # Neo4j count (best-effort)
    nodes_count = None
    neo_ok = False

    if neo_connection and getattr(neo_connection, "driver", None):
        try:
            with neo_connection.driver.session() as session:
                r = session.run("MATCH (n) RETURN COUNT(n) AS count")
                nodes_count = int(r.single()["count"])
            neo_ok = True
        except Exception:
            neo_ok = False

    return {
        "status": "healthy" if neo_ok else "degraded",
        "neo4j_connected": neo_ok,
        "nodes_in_graph": nodes_count,
        "agents": {
            "comprehension": "ready" if comprehension_agent else "not_ready",
            "complexity": "ready" if complexity_agent else "not_ready",
            "easy_rag": "ready" if rag_pipeline else "not_ready",
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/comprehension", tags=["Comprehension"])
async def comprehension(req: QueryRequest):
    _require(comprehension_agent, "Comprehension agent")
    try:
        data = comprehension_agent.understand_query(req.query)
        return {
            "success": True,
            "query": req.query,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehension error: {str(e)}")


@app.post("/complexity", tags=["Complexity"])
async def complexity(req: QueryRequest):
    _require(complexity_agent, "Complexity agent")
    try:
        result = complexity_agent.classify(req.query)  # returns dict {"level":..., ...}
        return {
            "success": True,
            "query": req.query,
            "complexity": result,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Complexity error: {str(e)}")


@app.post("/generate", tags=["Main"])
async def generate(req: QueryRequest):
    _require(comprehension_agent, "Comprehension agent")
    _require(complexity_agent, "Complexity agent")

    # 1) comprehension firewall
    try:
        comp_data = comprehension_agent.understand_query(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehension error: {str(e)}")

    if not comp_data.get("is_relevant", False):
        return {
            "success": False,
            "type": "out_of_context",
            "message": "This assistant only handles Nmap-related requests.",
            "comprehension": comp_data,
        }

    # 2) complexity routing
    try:
        complexity = complexity_agent.classify(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Complexity error: {str(e)}")
    level = complexity.get("level")

    if level == "easy":
        _require(rag_pipeline, "RAG pipeline")
        try:
            result = rag_pipeline.process_query(req.query)
            result["success"] = True if "success" not in result else result["success"]
            return {
                "success": True,
                "query": req.query,
                "comprehension": comp_data,
                "complexity": complexity,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

    if level == "medium":
        return {
            "success": False,
            "query": req.query,
            "comprehension": comp_data,
            "complexity": complexity,
            "error": "MEDIUM agent not implemented yet",
            "timestamp": datetime.now().isoformat(),
        }

    if level == "hard":
        _require(hard_agent, "Hard diffusion agent")
        target = IntentClassifier().extract_target(req.query)
        if not target:
            return {
                "success": False,
                "query": req.query,
                "comprehension": comp_data,
                "complexity": complexity,
                "error": "No target specified. Please provide an IP address or hostname.",
                "timestamp": datetime.now().isoformat(),
            }
        try:
            result = hard_agent.generate(req.query, target)
            return {
                "success": bool(result.get("ok")),
                "query": req.query,
                "comprehension": comp_data,
                "complexity": complexity,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Hard diffusion error: {str(e)}")

    raise HTTPException(status_code=500, detail=f"Unknown complexity level: {level}")
