# ==========================================
# api_medium.py - NMAP-AI Medium Agent API (FastAPI)
# Medium (T5 + LoRA) + Rule-based Nmap command correction
# ==========================================

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
except ImportError:
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    PeftModel = None


# =========================
# Models
# =========================

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")


class HealthResponse(BaseModel):
    status: str
    medium_agent_ready: bool
    timestamp: str


# =========================
# Globals / helpers
# =========================

VALID_NMAP = re.compile(r"^nmap\s+.+")
FORBIDDEN_TOKENS = ["--subnet", "target", "hosts.pvv", "hosts.txt"]


def _abs_backend_path(*parts: str) -> str:
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    return os.path.abspath(os.path.join(backend_dir, *parts))


def apply_medium_rules(prompt: str, cmd: str):
    """
    Apply lightweight safety/intent rules after T5 generation so the command
    stays aligned with the user's wording and common Nmap defaults.
    """
    original_cmd = cmd
    p = prompt.lower()

    if "top ports" in p and "-p-" in cmd:
        cmd = cmd.replace("-p-", "").strip()
    if "default and vuln" in p:
        cmd = re.sub(r"--script\s+\S+", "--script default,vuln", cmd)
        if "--script" not in cmd:
            cmd = f"nmap --script default,vuln {cmd.replace('nmap', '').strip()}"
    if "default scripts" in p and "vuln" not in p:
        if "--script" not in cmd:
            cmd = f"nmap --script default {cmd.replace('nmap', '').strip()}"
    if "ping scan" in p:
        cmd = re.sub(r"-sV|-O|-p-|--script\s+\S+", "", cmd).strip()
        if "-sn" not in cmd:
            cmd = f"nmap -sn {cmd.replace('nmap', '').strip()}"
    if "service detection" in p or "services" in p:
        if "-sV" not in cmd:
            cmd = cmd.replace("nmap", "nmap -sV")

    corrected = cmd != original_cmd
    return cmd.strip(), corrected


def is_valid_nmap(cmd: str) -> bool:
    if not VALID_NMAP.match(cmd):
        return False
    for bad in FORBIDDEN_TOKENS:
        if bad in cmd:
            return False
    return True


class MediumAgent:
    """
    T5 + LoRA medium-tier agent with post-generation rules.
    """

    def __init__(
        self,
        base_model: str = "google-t5/t5-small",
        lora_path: Optional[str] = None,
    ):
        self.base_model = base_model
        self.lora_path = lora_path or _abs_backend_path("Agents", "Agent_medium", "T5", "T5_qlora-nmap")
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.model: Optional[PeftModel] = None
        self.tokenizer: Optional[Any] = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self):
        if not (torch and AutoTokenizer and AutoModelForSeq2SeqLM and PeftModel):
            raise RuntimeError("Medium agent dependencies are missing (torch/transformers/peft).")

        bnb_config = None
        if self.device == "cuda":
            if BitsAndBytesConfig is None:
                raise RuntimeError("bitsandbytes is required for 4-bit CUDA loading.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.lora_path)

    def generate(self, query: str) -> Dict[str, Any]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if not self.is_ready:
            raise RuntimeError("Medium agent not initialized.")

        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=1,
                do_sample=False,
            )
        raw_cmd = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("\n")[0].strip()
        final_cmd, corrected = apply_medium_rules(query, raw_cmd)
        valid = is_valid_nmap(final_cmd)
        return {
            "command": final_cmd,
            "raw_command": raw_cmd,
            "corrected": corrected,
            "valid": valid,
        }


# =========================
# FastAPI app
# =========================

app = FastAPI(
    title="NMAP-AI Medium API",
    description="Medium Agent (T5 + LoRA) for Nmap command generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

medium_agent = MediumAgent()


@app.on_event("startup")
async def startup_event():
    global medium_agent
    try:
        medium_agent.load()
        print(f"ok. Medium agent ready on {medium_agent.device}")
    except Exception as e:
        print(f"Medium agent init failed: {e}")


def _require(agent: Any, name: str):
    ready = agent is not None and (not hasattr(agent, "is_ready") or agent.is_ready)
    if not ready:
        raise HTTPException(status_code=503, detail=f"{name} not initialized")


# =========================
# Routes
# =========================

@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "healthy" if medium_agent.is_ready else "degraded",
        "medium_agent_ready": medium_agent.is_ready,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/generate")
async def generate(req: QueryRequest):
    _require(medium_agent, "Medium agent")
    try:
        result = medium_agent.generate(req.query)
        response = {
            "success": bool(result["valid"]),
            "query": req.query,
            "result": {
                "command": result["command"] if result["valid"] else None,
                "raw_command": result["raw_command"],
                "corrected": result["corrected"],
                "valid": result["valid"],
            },
            "timestamp": datetime.now().isoformat(),
        }
        if not result["valid"]:
            response["warning"] = "Generated command failed validation; please refine the query."
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Medium agent error: {str(e)}")
