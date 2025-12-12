# ==========================================
# api.py
# NMAP-AI RAG API - Single Endpoint
# FastAPI + Swagger - Simple & Clean
# ==========================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn

# Import du RAG Engine
from Agents.Agent_easy.rag_engine import NeoConnection, NmapRAGPipeline

# ========== MODELS ==========

class QueryRequest(BaseModel):
    """Requête utilisateur en langage naturel"""
    query: str = Field(
        ..., 
        description="Natural language query",
        example="Scan 192.168.1.1 for open ports quickly"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {"query": "Scan 192.168.1.1 for open ports"},
                {"query": "Detect services on 10.0.0.1 quickly"},
                {"query": "Stealth scan ports 80,443 on scanme.nmap.org"},
                {"query": "Check if 172.16.0.1 is alive"}
            ]
        }


class NmapResponse(BaseModel):
    """Réponse complète avec commande Nmap générée"""
    success: bool = Field(..., description="Statut de la génération")
    query: str = Field(..., description="Requête originale")
    intent: Optional[str] = Field(None, description="Intention détectée")
    confidence: Optional[float] = Field(None, description="Confiance (0-1)")
    target: Optional[str] = Field(None, description="Cible extraite")
    ports: Optional[str] = Field(None, description="Ports extraits")
    command: Optional[str] = Field(None, description="Commande Nmap générée")
    explanation: Optional[str] = Field(None, description="Explication détaillée")
    requires_root: Optional[bool] = Field(None, description="Nécessite root/sudo")
    warnings: Optional[List[str]] = Field(None, description="Avertissements")
    validation: Optional[Dict[str, Any]] = Field(None, description="Validation")
    kg_context: Optional[Dict[str, Any]] = Field(None, description="Contexte du Knowledge Graph")
    timestamp: Optional[str] = Field(None, description="Timestamp")
    error: Optional[str] = Field(None, description="Message d'erreur")
    suggestion: Optional[str] = Field(None, description="Suggestion si erreur")


# ========== FASTAPI APP ==========

app = FastAPI(
    title="🎯 NMAP-AI RAG API",
    description="""
    ## 🚀 NMAP-AI - Knowledge Graph RAG Engine (Zero-Shot)
    
    **Convertissez du langage naturel en commandes Nmap valides et optimisées !**
    
    ### ✨ Fonctionnalités
    - 🧠 **Intelligence Artificielle** : Classification d'intention zero-shot
    - 📊 **Knowledge Graph** : Neo4j avec 200+ nœuds (Scans, Options, Relations)
    - ✅ **Validation** : Détection automatique des conflits et dépendances
    - 🎓 **Apprentissage** : Exemples tirés du Knowledge Graph
    - ⚡ **Rapide** : Génération en <100ms
    
    ### 🎯 Architecture
    ```
    User Query → Intent Classifier → Knowledge Graph RAG → Command Generator → Validator → Response
    ```
    
    ### 📝 Exemples de requêtes
    - "Scan 192.168.1.1 for open ports"
    - "Detect services on 10.0.0.1 quickly"
    - "Stealth scan ports 80,443 on scanme.nmap.org"
    - "Check if 172.16.0.1 is alive"
    - "Detect operating system of 192.168.1.50"
    
    ### 🔗 Knowledge Graph
    - **12 Scans** : -sS, -sT, -sU, -sF, -sN, -sX, -sA, -sW, -sn, -O, -sV, -A
    - **20+ Options** : Timing, Ports, Output, Evasion
    - **Relations** : CONFLICTS_WITH, COMPATIBLE_WITH, WORKS_WITH
    - **Validations** : Privilèges, Dépendances, Warnings
    
    ### 🎓 Tier: EASY (Zero-Shot)
    - Pas de fine-tuning
    - 100% basé sur le Knowledge Graph
    - Génération par règles + RAG
    """,
    version="1.0.0",
    contact={
        "name": "NMAP-AI Team",
        "email": "contact@nmap-ai.dev"
    },
    license_info={
        "name": "MIT License",
    }
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== GLOBAL STATE ==========

neo_connection = None
rag_pipeline = None


# ========== STARTUP/SHUTDOWN ==========

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global neo_connection, rag_pipeline
    
    try:
        print("\n" + "="*70)
        print("🚀 NMAP-AI RAG API - Starting")
        print("="*70)
        
        # Connexion Neo4j
        neo_connection = NeoConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        
        # Initialiser le RAG Pipeline
        rag_pipeline = NmapRAGPipeline(neo_connection.driver)
        
        print("✅ RAG Pipeline initialized successfully!")
        print("📚 Swagger UI: http://localhost:8000/docs")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt"""
    global neo_connection
    
    if neo_connection:
        neo_connection.close()
        print("\n✅ Neo4j connection closed")


# ========== MAIN ENDPOINT ==========

@app.post(
    "/generate",
    response_model=NmapResponse,
    summary="🎯 Génère une commande Nmap",
    description="""
    ## 🎯 Endpoint Principal - Génération de Commande Nmap
    
    **Processus complet :**
    
    1. **Intent Classification** : Détecte l'intention (port_scan, service_detection, os_detection, etc.)
    2. **Parameter Extraction** : Extrait la cible (IP/hostname), les ports, les options
    3. **Knowledge Graph RAG** : Interroge Neo4j pour récupérer :
       - Les scans recommandés pour cet intent
       - Les options compatibles
       - Les conflits potentiels
       - Les dépendances et privilèges requis
       - Les exemples similaires
    4. **Command Generation** : Construit la commande Nmap optimale
    5. **Validation** : Vérifie la syntaxe, les conflits, les dépendances
    
    **Réponse :**
    - `command` : Commande Nmap prête à exécuter
    - `explanation` : Explication détaillée de chaque option
    - `warnings` : Avertissements (privilèges, dépendances)
    - `kg_context` : Contexte complet du Knowledge Graph
    - `validation` : Score de validation (0-1)
    
    **Exemples :**
    ```json
    {"query": "Scan 192.168.1.1 for open ports"}
    → "nmap -sS 192.168.1.1"
    
    {"query": "Detect services on 10.0.0.1 quickly"}
    → "nmap -sS -sV -T4 10.0.0.1"
    
    {"query": "Stealth scan ports 80,443 on scanme.nmap.org"}
    → "nmap -sS -T1 -p 80,443 scanme.nmap.org"
    ```
    """,
    response_description="Commande Nmap générée avec contexte complet",
    tags=["🎯 Main Endpoint"]
)
async def generate_nmap_command(request: QueryRequest):
    """
    🎯 Génère une commande Nmap à partir d'une requête en langage naturel
    
    **Knowledge Graph RAG :**
    - Utilise Neo4j pour récupérer le contexte complet
    - Exploite les relations : CONFLICTS_WITH, COMPATIBLE_WITH, WORKS_WITH
    - Valide avec les règles du graphe (Validation, Privilege, Dependency)
    
    **Zero-Shot :**
    - Pas de modèle ML entraîné
    - Classification par patterns de mots-clés
    - Génération par règles + traversée du graphe
    """
    
    if not rag_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="RAG Pipeline not initialized. Check Neo4j connection."
        )
    
    try:
        # Process query avec le RAG Pipeline
        result = rag_pipeline.process_query(request.query)
        
        # Return response
        return NmapResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )


# ========== HEALTH CHECK ==========

@app.get(
    "/health",
    summary="🏥 Health Check",
    description="Vérifie l'état de l'API et de Neo4j",
    tags=["Health"]
)
async def health_check():
    """Health check - Vérifie Neo4j et le RAG Pipeline"""
    
    if not neo_connection or not neo_connection.driver:
        raise HTTPException(status_code=503, detail="Neo4j not connected")
    
    try:
        with neo_connection.driver.session() as session:
            result = session.run("MATCH (n) RETURN COUNT(n) as count")
            count = result.single()['count']
        
        return {
            "status": "healthy",
            "neo4j_connected": True,
            "nodes_in_graph": count,
            "rag_pipeline": "ready",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j error: {str(e)}"
        )


# ========== ROOT ==========

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - Informations sur l'API"""
    return {
        "message": "🎯 NMAP-AI RAG API",
        "version": "1.0.0",
        "tier": "EASY (Zero-Shot)",
        "docs": "/docs",
        "health": "/health",
        "main_endpoint": "/generate",
        "description": "Convert natural language to Nmap commands using Knowledge Graph RAG"
    }


# ========== RUN ==========

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 NMAP-AI RAG API - Starting Server")
    print("="*70)
    print("📚 Swagger UI: http://localhost:8000/docs")
    print("📖 ReDoc: http://localhost:8000/redoc")
    print("🔌 Main Endpoint: POST http://localhost:8000/generate")
    print("="*70 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )