"""
Serveur Flask - Agent Complexity avec Word2Vec
===============================================

Port: 5003

Installation:
    pip install flask flask-cors numpy gensim

Usage:
    py -3.12 complexity_server_word2vec.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from agent_complexity_word2vec import ComplexityClassifierSLM

app = Flask(__name__)
CORS(app)


# ==========================================
# INITIALISATION
# ==========================================

print("\n Initialisation - SLM + Word2Vec...")
classifier = ComplexityClassifierSLM()
metrics = classifier.train()
print(f" Agent prêt ! (Accuracy: {metrics['accuracy']*100:.2f}%)\n")


# ==========================================
# CONFIGURATION
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
# ENDPOINTS
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    """Health check"""
    return jsonify({
        "status": "ok",
        "agent": "Complexity - SLM + Word2Vec",
        "port": 5003,
        "method": metrics['method'],
        "accuracy": round(metrics['accuracy'], 3),
        "device": metrics['device']
    }), 200


@app.route("/classify", methods=["POST"])
def classify():
    """
    Classification simple
    
    Body: {"query": "Scan all ports"}
    Response: {"query": "...", "level": "medium"}
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    query = data.get("query") or data.get("user_query")
    
    if not query:
        return jsonify({
            "error": "Missing 'query' field",
            "example": {"query": "nmap -p 80 192.168.1.1"}
        }), 400
    
    try:
        result = classifier.classify(query)
        level = result['level']
        
        response = {
            "query": query,
            "level": level
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Erreur classification"
        }), 500


@app.route("/classify-detailed", methods=["POST"])
def classify_detailed():
    """
    Classification détaillée avec toutes les infos
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    query = data.get("query") or data.get("user_query")
    
    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400
    
    try:
        result = classifier.classify(query)
        
        # Ajouter routing info
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
    print("  AGENT COMPLEXITY - SLM + WORD2VEC")
    print("="*70)
    print("\n Rôle:")
    print("  → Classification EASY/MEDIUM/HARD")
    print("\n Endpoints:")
    print("  → GET  /health")
    print("  → POST /classify")
    print("  → POST /classify-detailed")
    print("\n Méthode:")
    print(f"  → {metrics['method']}")
    print(f"  → Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  → Vocab: {metrics['vocab_size']} mots")
    print(f"  → Device: {metrics['device']}")
    print("\n Routing:")
    for level, info in ROUTING_CONFIG.items():
        print(f"  {info['color']} {level.upper():6} → Port {info['port']}")
    print("\n Serveur: http://localhost:5003")
    print("="*70 + "\n")
    
    app.run(host="0.0.0.0", port=5003, debug=False)