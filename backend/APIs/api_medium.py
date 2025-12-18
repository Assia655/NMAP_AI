# ==========================================
# api_medium.py - NMAP-AI Medium Agent API (FastAPI)
# Medium (T5 + LoRA) + Rule-based Nmap command correction
# ==========================================

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel

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
# Globals
# =========================

medium_agent_model: PeftModel = None
medium_agent_tokenizer: AutoTokenizer = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rule-based corrections
VALID_NMAP = re.compile(r"^nmap\s+.+")
FORBIDDEN_TOKENS = ["--subnet", "target", "hosts.pvv", "hosts.txt"]

def apply_medium_rules(prompt: str, cmd: str):
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


    corrected = (cmd != original_cmd)
    return cmd.strip(), corrected

def is_valid_nmap(cmd: str) -> bool:
    if not VALID_NMAP.match(cmd):
        return False
    for bad in FORBIDDEN_TOKENS:
        if bad in cmd:
            return False
    return True

# =========================
# FastAPI app
# =========================

app = FastAPI(title="NMAP-AI Medium API",
              description="Medium Agent (T5 + LoRA) for Nmap command generation",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Startup
# =========================

@app.on_event("startup")
async def startup_event():
    global medium_agent_model, medium_agent_tokenizer
    try:
        BASE_MODEL = "google-t5/t5-small"
        LORA_PATH = os.path.join("Agents", "Agent_medium", "T5", "T5_qlora-nmap")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map="auto" if DEVICE=="cuda" else None
        )
        medium_agent_model = PeftModel.from_pretrained(base_model, LORA_PATH)
        medium_agent_model.to(DEVICE)
        medium_agent_model.eval()
        medium_agent_tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
        print(f"✅ Medium agent ready on {DEVICE}")
    except Exception as e:
        print(f"❌ Medium agent init failed: {e}")
        medium_agent_model = None
        medium_agent_tokenizer = None

# =========================
# Helpers
# =========================

def _require(agent: Any, name: str):
    if agent is None:
        raise HTTPException(status_code=503, detail=f"{name} not initialized")

# =========================
# Routes
# =========================

@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "healthy" if medium_agent_model else "degraded",
        "medium_agent_ready": medium_agent_model is not None,
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/generate")
async def generate(req: QueryRequest):
    _require(medium_agent_model, "Medium agent")
    _require(medium_agent_tokenizer, "Medium tokenizer")
    try:
        inputs = medium_agent_tokenizer(req.query, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = medium_agent_model.generate(
                **inputs, max_new_tokens=60, num_beams=1, do_sample=False
            )
        raw_cmd = medium_agent_tokenizer.decode(outputs[0], skip_special_tokens=True).split("\n")[0].strip()
        final_cmd, corrected = apply_medium_rules(req.query, raw_cmd)
        valid = is_valid_nmap(final_cmd)
        return {
            "success": True,
            "query": req.query,
            "raw_command": raw_cmd,
            "final_command": final_cmd,
            "corrected": corrected,
            "valid": valid,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Medium agent error: {str(e)}")
