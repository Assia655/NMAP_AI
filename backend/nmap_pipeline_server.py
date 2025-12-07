"""
Serveur Flask - Agent 2 Complexity
==============================================

Port: 5003 (Agent Complexity)

Installation:
    pip install flask flask-cors sentence-transformers scikit-learn transformers torch

Usage:
    python complexity_server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from agent_complexity_slm_embeddings import ComplexityClassifierSLM

app = Flask(__name__)
CORS(app)


# ==========================================
# INITIALISATION AGENT 2: COMPLEXITY
# ==========================================

print("\n🚀 Initialisation de l'Agent 2 - Complexity (SLM + Embeddings)...")
classifier = ComplexityClassifierSLM()
metrics = classifier.train()
print(f"✅ Agent 2 Complexity prêt ! (Accuracy: {metrics['accuracy']*100:.2f}%)\n")


# ==========================================
# CONFIGURATION DES ROUTES
# ==========================================

ROUTING_CONFIG = {
    'easy': {
        'port': 5004,
        'model': 'KG-RAG (Knowledge Graph)',
        'color': '🟢'
    },
    'medium': {
        'port': 5005,
        'model': 'LoRA fine-tuned (T5-small / Phi-4)',
        'color': '🟠'
    },
    'hard': {
        'port': 5006,
        'model': 'Diffusion-based synthesis',
        'color': '🔴'
    }
}


# ==========================================
# ENDPOINTS API
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de santé de l'Agent 2"""
    return jsonify({
        "status": "ok",
        "agent": "Agent 2 - Complexity",
        "port": 5003,
        "method": metrics['method'],
        "accuracy": round(metrics['accuracy'], 3),
        "device": metrics['device']
    }), 200


@app.route("/classify", methods=["POST"])
def classify():
    """
    Endpoint principal : Classification de complexité
    
    Body JSON:
    {
        "query": "Scan all ports using SYN scan"
    }
    
    Response:
    {
        "query": "Scan all ports using SYN scan",
        "level": "medium"
    }
    """
    data = request.get_json()
    
    # Validation
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    query = data.get("query") or data.get("user_query")
    
    if not query:
        return jsonify({
            "error": "Missing 'query' field",
            "example": {"query": "nmap -p 80 192.168.1.1"}
        }), 400
    
    try:
        # ===================================
        # CLASSIFICATION (Agent 2)
        # ===================================
        classification_result = classifier.classify(query)
        level = classification_result['level']
        
        # ===================================
        # RÉPONSE SIMPLE (query + level)
        # ===================================
        response = {
            "query": query,
            "level": level
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Erreur lors de la classification"
        }), 500


@app.route("/classify-detailed", methods=["POST"])
def classify_detailed():
    """
    Endpoint détaillé : Classification avec toutes les infos
    
    Response complète avec probabilités, raisonnement, etc.
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    query = data.get("query") or data.get("user_query")
    
    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400
    
    try:
        # Classification complète
        result = classifier.classify(query)
        
        # Ajouter les infos de routing
        route_info = ROUTING_CONFIG[result['level']]
        result['routing'] = {
            'port': route_info['port'],
            'model': route_info['model'],
            'color': route_info['color']
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  AGENT 2 - COMPLEXITY SERVER")
    print("="*70)
    print("\n🎯 Rôle:")
    print("  → Classifie les requêtes Nmap en 3 niveaux: EASY/MEDIUM/HARD")
    print("\n📡 Endpoints:")
    print("  → GET  /health              - Statut de l'agent")
    print("  → POST /classify            - Classification simple (query + level)")
    print("  → POST /classify-detailed   - Classification détaillée (avec toutes les infos)")
    print("\n🧠 Méthode:")
    print(f"  → {metrics['method']}")
    print(f"  → Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  → Embedding dimension: {metrics['embedding_dim']}")
    print(f"  → Device: {metrics['device']}")
    print("\n🎯 Routing:")
    for level, info in ROUTING_CONFIG.items():
        print(f"  {info['color']} {level.upper():6} → Port {info['port']} ({info['model']})")
    print("\n🌐 Serveur démarré sur http://localhost:5003")
    print("="*70 + "\n")
    
    app.run(host="0.0.0.0", port=5003, debug=False)