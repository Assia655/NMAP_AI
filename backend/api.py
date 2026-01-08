# ==========================================
# api1.py - NMAP-AI Unified API (FastAPI)
# Comprehension + Complexity + Easy(RAG Neo4j) + Validation
# Single server: http://localhost:8000
# ==========================================

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Easy (RAG / Neo4j) ---
from Agents.Agent_easy.rag_engine import NeoConnection, NmapRAGPipeline, IntentClassifier

# --- Complexity ---
from Agents.Agent_complexity.complexity_slm_word2vec import ComplexityClassifierSLM

# --- Comprehension ---
from Agents.Agent_comprehension.nmap_agent_embeddings import NMAPEmbeddingAgent
from APIs.api_medium import MediumAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# Models
# =========================

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    validation_api_connected: bool
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
medium_agent: Optional[MediumAgent] = None


def _abs_path_from_backend(*parts: str) -> str:
    """Build absolute path relative to backend/ directory (api1.py location)."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(backend_dir, *parts))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle (no deprecated on_event)."""
    global neo_connection, rag_pipeline, complexity_agent, comprehension_agent, medium_agent

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
        logger.info("✅ Neo4j + RAG ready")
    except Exception as exc:
        logger.error(f"❌ Neo4j/RAG init failed: {exc}")
        raise

    # 2) Init Comprehension Agent
    try:
        domain_path = _abs_path_from_backend("Agents", "Agent_comprehension", "nmap_domain.txt")
        if not os.path.exists(domain_path):
            raise FileNotFoundError(f"nmap_domain.txt not found at: {domain_path}")
        comprehension_agent = NMAPEmbeddingAgent(domain_path)
        logger.info("✅ Comprehension agent ready")
    except Exception as exc:
        logger.error(f"❌ Comprehension init failed: {exc}")
        comprehension_agent = None

    # 3) Init Complexity Agent
    try:
        complexity_agent = ComplexityClassifierSLM()
        complexity_agent.train()
        logger.info("✅ Complexity agent ready")
    except Exception as exc:
        logger.error(f"❌ Complexity init failed: {exc}")
        complexity_agent = None

    # 4) Init Medium Agent (T5 + LoRA)
    try:
        medium_agent = MediumAgent()
        medium_agent.load()
        logger.info(f"✅ Medium/hard agent ready on {medium_agent.device}")
    except Exception as exc:
        logger.error(f"❌ Medium/hard agent init failed: {exc}")
        medium_agent = None

    # 5) Test Validation API connection
    try:
        test_validation_connection()
        logger.info("✅ Validation API connection successful")
    except Exception as exc:
        logger.warning(f"⚠️ Validation API not reachable: {exc}")
        logger.warning("System will continue but validation will fail")

    logger.info("📚 Swagger UI: http://localhost:8000/docs")
    logger.info("=" * 70 + "\n")

    yield

    # Shutdown
    if neo_connection:
        try:
            neo_connection.close()
            logger.info("✅ Neo4j connection closed")
        except Exception:
            pass


app = FastAPI(
    title="NMAP-AI Unified API",
    description="Comprehension + Complexity + RAG(Easy) + Validation under one FastAPI server.",
    version="2.0.0",
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
    ready = agent is not None
    if ready and hasattr(agent, "is_ready"):
        ready = bool(agent.is_ready)
    if not ready:
        raise HTTPException(status_code=503, detail=f"{name} not initialized")

def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

import requests

# Configuration for validation API
VALIDATION_API_URL = "http://192.168.56.106:9000/validate"
VALIDATION_TIMEOUT = 10  # seconds

def test_validation_connection():
    """Test connection to validation API"""
    try:
        response = requests.get(
            "http://192.168.56.106:9000/health",
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        raise ConnectionError(f"Cannot connect to validation API: {str(e)}")

def validate_command_with_kali(command: str, complexity: str) -> dict:
    """
    Send command to Kali VM for validation
    
    Args:
        command: The Nmap command to validate
        complexity: The complexity level (easy/medium/hard)
        
    Returns:
        dict with validation results
    """
    try:
        logger.info(f"[VALIDATION] Sending command to Kali: {command[:50]}...")
        logger.info(f"[VALIDATION] Complexity: {complexity}")
        logger.info(f"[VALIDATION] Target URL: {VALIDATION_API_URL}")
        
        response = requests.post(
            VALIDATION_API_URL,
            json={
                "command": command,
                "complexity": complexity
            },
            timeout=VALIDATION_TIMEOUT
        )
        
        logger.info(f"[VALIDATION] Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"[VALIDATION] HTTP error: {response.status_code}")
            logger.error(f"[VALIDATION] Response: {response.text}")
            return {
                "status": "unknown",
                "valid": False,
                "safe": True,
                "requires_privilege": False,
                "errors": [f"Validation API returned status {response.status_code}"],
                "warnings": [],
                "suggestions": []
            }
        
        validation_result = response.json()
        logger.info(f"[VALIDATION] Result: {validation_result.get('status')}")
        logger.info(f"[VALIDATION] Valid: {validation_result.get('valid')}")
        
        if validation_result.get('errors'):
            logger.warning(f"[VALIDATION] Errors: {validation_result.get('errors')}")
        if validation_result.get('warnings'):
            logger.info(f"[VALIDATION] Warnings: {validation_result.get('warnings')}")
            
        return validation_result
        
    except requests.exceptions.Timeout:
        logger.error("[VALIDATION] Request timed out")
        return {
            "status": "unknown",
            "valid": False,
            "safe": True,
            "requires_privilege": False,
            "errors": ["Validation request timed out"],
            "warnings": ["Validation API is not responding"],
            "suggestions": ["Check if validation API is running on Kali VM"]
        }
    except requests.exceptions.ConnectionError as e:
        logger.error(f"[VALIDATION] Connection error: {str(e)}")
        return {
            "status": "unknown",
            "valid": False,
            "safe": True,
            "requires_privilege": False,
            "errors": [f"Cannot connect to validation API: {str(e)}"],
            "warnings": ["Make sure Kali VM is running and validation API is started"],
            "suggestions": [
                "Start validation API on Kali: uvicorn validation_api:app --host 0.0.0.0 --port 9000",
                "Check network connectivity to 192.168.56.106:9000"
            ]
        }
    except Exception as e:
        logger.error(f"[VALIDATION] Unexpected error: {str(e)}", exc_info=True)
        return {
            "status": "unknown",
            "valid": False,
            "safe": True,
            "requires_privilege": False,
            "errors": [f"Validation error: {str(e)}"],
            "warnings": [],
            "suggestions": []
        }

def normalize_validation(validation: dict) -> dict:
    """
    Normalize validation response to consistent format
    
    Args:
        validation: Raw validation response from Kali
        
    Returns:
        Normalized validation dict with verdict
    """
    status = str(validation.get("status", "unknown")).lower()
    
    # Map status to verdict
    mapping = {
        "valid": "VALID",
        "invalid": "INVALID",
        "repairable": "REPAIRABLE",
        "unsafe": "UNSAFE",
        "privilege_required": "REPAIRABLE",  # Treat privilege requirement as repairable
        "unknown": "UNKNOWN"
    }
    
    verdict = mapping.get(status, "UNKNOWN")
    
    # Handle requires_privilege flag
    if validation.get("requires_privilege") and status == "valid":
        verdict = "REPAIRABLE"
        if "suggestions" not in validation:
            validation["suggestions"] = []
        if "Add 'sudo' prefix" not in str(validation["suggestions"]):
            validation["suggestions"].append("Add 'sudo' prefix for privileged operations")
    
    return {
        "verdict": verdict,
        "raw": validation,
        "details": {
            "errors": validation.get("errors", []),
            "warnings": validation.get("warnings", []),
            "suggestions": validation.get("suggestions", []),
            "safe": validation.get("safe", True),
            "requires_privilege": validation.get("requires_privilege", False)
        }
    }

def generate_with_validation(
    query: str,
    initial_level: str,
    comp_data: Dict[str, Any]
):
    """
    Generate a command and validate it via Kali.
    Retry if validator says REPAIRABLE.
    
    Args:
        query: User's natural language query
        initial_level: Initial complexity level (easy/medium/hard)
        comp_data: Comprehension data
        
    Returns:
        dict with generation and validation results
    """
    
    tried_levels = set()
    level = initial_level
    last_command = None
    last_validation = None
    attempts = []

    for attempt in range(3):  # max 3 attempts
        tried_levels.add(level)
        logger.info(f"[GENERATION] Attempt {attempt + 1}, Level: {level}")

        # -------- GENERATION --------
        try:
            if level == "easy":
                _require(rag_pipeline, "RAG pipeline")
                gen = rag_pipeline.process_query(query)
                command = gen.get("command")
                generation_method = "RAG"
            else:
                _require(medium_agent, "Medium agent")
                gen = medium_agent.generate(query)
                command = gen.get("command") or gen.get("raw_command")
                generation_method = "T5+LoRA"
            
            logger.info(f"[GENERATION] Method: {generation_method}")
            logger.info(f"[GENERATION] Command: {command}")
            
        except Exception as e:
            logger.error(f"[GENERATION] Error: {str(e)}", exc_info=True)
            command = None

        if not command:
            last_validation = {
                "verdict": "INVALID",
                "raw": {"status": "invalid"},
                "details": {
                    "errors": ["Model failed to produce a command"],
                    "warnings": [],
                    "suggestions": ["Try rephrasing your query"]
                }
            }
            attempts.append({
                "level": level,
                "command": None,
                "validation": last_validation
            })
            break

        # -------- VALIDATION --------
        raw_validation = validate_command_with_kali(command, level)
        normalized = normalize_validation(raw_validation)

        last_command = command
        last_validation = normalized
        
        attempts.append({
            "level": level,
            "command": command,
            "validation": normalized
        })

        verdict = normalized["verdict"]
        logger.info(f"[VALIDATION] Verdict: {verdict}")

        # ✅ VALID → SUCCESS
        if verdict == "VALID":
            logger.info("[RESULT] Command validated successfully")
            return {
                "success": True,
                "command": command,
                "validation": normalized,
                "final_complexity": level,
                "attempts": attempts,
            }

        # 🟡 REPAIRABLE → retry with different agent
        if verdict == "REPAIRABLE":
            logger.info("[RESULT] Command is repairable, trying different level")
            # Try switching between easy and medium
            if level == "easy":
                level = "medium"
            elif level == "medium":
                level = "easy"
            else:
                level = "medium"
                
            if level in tried_levels:
                logger.warning("[RESULT] Already tried this level, stopping")
                break
            continue

        # ❌ Everything else → STOP
        logger.warning(f"[RESULT] Command validation failed: {verdict}")
        break

    # -------- FINAL FAILURE --------
    logger.error("[RESULT] All generation attempts failed")
    return {
        "success": False,
        "command": last_command,
        "validation": last_validation,
        "final_complexity": level,
        "attempts": attempts,
        "message": "Command generation failed validation after multiple attempts",
    }


# =========================
# Routes
# =========================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "NMAP-AI Unified API v2.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/comprehension", "/complexity", "/generate"],
        "validation_api": VALIDATION_API_URL
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
    
    # Test validation API
    validation_ok = False
    try:
        validation_ok = test_validation_connection()
    except Exception:
        validation_ok = False

    return {
        "status": "healthy" if (neo_ok and validation_ok) else "degraded",
        "neo4j_connected": neo_ok,
        "validation_api_connected": validation_ok,
        "nodes_in_graph": nodes_count,
        "agents": {
            "comprehension": "ready" if comprehension_agent else "not_ready",
            "complexity": "ready" if complexity_agent else "not_ready",
            "easy_rag": "ready" if rag_pipeline else "not_ready",
            "medium": "ready" if medium_agent and getattr(medium_agent, "is_ready", False) else "not_ready",
            "validation": "ready" if validation_ok else "not_ready",
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
        logger.error(f"Comprehension error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comprehension error: {str(e)}")


@app.post("/complexity", tags=["Complexity"])
async def complexity(req: QueryRequest):
    _require(complexity_agent, "Complexity agent")
    try:
        result = complexity_agent.classify(req.query)
        return {
            "success": True,
            "query": req.query,
            "complexity": result,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Complexity error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Complexity error: {str(e)}")

@app.post("/generate", tags=["Main"])
async def generate(req: QueryRequest):
    """
    Main endpoint: Comprehension → Complexity → Generation → Validation
    """
    _require(comprehension_agent, "Comprehension agent")
    _require(complexity_agent, "Complexity agent")
    
    logger.info(f"[REQUEST] Query: {req.query}")

    # 1) Comprehension firewall
    try:
        comp_data = comprehension_agent.understand_query(req.query)
        logger.info(f"[COMPREHENSION] Relevance: {comp_data.get('is_relevant')}")
    except Exception as e:
        logger.error(f"Comprehension error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comprehension error: {str(e)}")

    confidence = _to_float(comp_data.get("confidence", 0))
    similarity = _to_float(comp_data.get("similarity", 0))

    is_relevant = (
            comp_data.get("is_relevant") is True
            or confidence >= 0.3
            or similarity >= 0.3
            or bool(comp_data.get("keywords"))
    )

    if not is_relevant:
        logger.info("[COMPREHENSION] Query out of context")
        return {
            "success": False,
            "type": "out_of_context",
            "message": "This assistant only handles Nmap-related requests.",
            "comprehension": comp_data,
            "timestamp": datetime.now().isoformat(),
        }

    # 2) Complexity classification
    try:
        complexity = complexity_agent.classify(req.query)
        level = complexity.get("level")
        logger.info(f"[COMPLEXITY] Level: {level}")
    except Exception as e:
        logger.error(f"Complexity error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Complexity error: {str(e)}")

    # 3) Generate + validate + retry
    result = generate_with_validation(
        query=req.query,
        initial_level=level,
        comp_data=comp_data
    )
    
    logger.info(f"[FINAL] Success: {result.get('success')}")
    logger.info(f"[FINAL] Command: {result.get('command')}")

    return {
        "success": result.get("success"),
        "query": req.query,
        "comprehension": comp_data,
        "complexity": complexity,
        "final_complexity": result.get("final_complexity"),
        "attempts": result.get("attempts"),
        "result": {
            "command": result.get("command"),
            "validation": result.get("validation"),
        },
        "timestamp": datetime.now().isoformat(),
    }
