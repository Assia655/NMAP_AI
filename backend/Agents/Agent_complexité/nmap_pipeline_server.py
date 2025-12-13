"""
Serveur Flask - Pipeline Complet NMAP-AI
=========================================

Architecture complète:
1. Agent Compréhension (vérifie si Nmap)
2. Agent Complexité (SLM + Embeddings)
3. Routage automatique vers EASY/MEDIUM/HARD

Port: 5000 (serveur principal)

Installation:
    pip install flask flask-cors sentence-transformers scikit-learn

Usage:
    python nmap_pipeline_server.py
"""

from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from backend.Agents.Agent_complexité.agent_complexity_slm_embeddings import ComplexityClassifierSLM
import requests
import re

app = Flask(__name__)
CORS(app)


# ==========================================
# AGENT 1: COMPREHENSION
# ==========================================

def load_nmap_domain():
    """Charge le vocabulaire Nmap"""
    nmap_keywords = [
        'nmap', 'scan', 'port', 'ports', 'host', 'target', 'network',
        'ping', 'tcp', 'udp', 'syn', 'ack', 'fin', 'version', 'os',
        'detect', 'detection', 'script', 'nse', 'firewall', 'bypass',
        'stealth', 'fragmentation', 'decoy', 'timing', 'aggressive',
        '-sS', '-sT', '-sU', '-sV', '-O', '-A', '-p', '-Pn', '-f',
        'scanner', 'enumerate', 'discovery', 'reconnaissance'
    ]
    return nmap_keywords


def check_nmap_context(query):
    """
    Vérifie si la requête concerne Nmap
    
    Returns:
        dict: {is_nmap: bool, confidence: float}
    """
    nmap_keywords = load_nmap_domain()
    query_lower = query.lower()
    
    # Compter les mots Nmap présents
    matches = sum(1 for keyword in nmap_keywords if keyword in query_lower)
    
    # Calcul de confiance
    confidence = min(1.0, matches / 3)  # 3+ mots = haute confiance
    is_nmap = confidence > 0.3
    
    return {
        'is_nmap': is_nmap,
        'confidence': confidence,
        'matches': matches
    }


# ==========================================
# AGENT 2: COMPLEXITY (SLM + Embeddings)
# ==========================================

print("\n🚀 Initialisation de l'agent de complexité (SLM + Embeddings)...")
classifier = ComplexityClassifierSLM()
metrics = classifier.train()
print(f"✅ Agent Complexité prêt ! (Accuracy: {metrics['accuracy']*100:.2f}%)\n")


# ==========================================
# CONFIGURATION DES PAGES
# ==========================================

PAGE_CONFIG = {
    'easy': {
        'url': 'http://localhost:5004/process',
        'port': 5004,
        'model': 'KG-RAG (Knowledge Graph)',
        'color': '🟢'
    },
    'medium': {
        'url': 'http://localhost:5005/process',
        'port': 5005,
        'model': 'LoRA fine-tuned (T5-small / Phi-4)',
        'color': '🟠'
    },
    'hard': {
        'url': 'http://localhost:5006/process',
        'port': 5006,
        'model': 'Diffusion-based synthesis',
        'color': '🔴'
    }
}


def send_to_target_server(query, level, classification):
    """
    Envoie la requête au serveur cible (5004/5005/5006)
    
    Args:
        query (str): Requête utilisateur
        level (str): Niveau (easy/medium/hard)
        classification (dict): Résultat de classification
        
    Returns:
        dict: Résultat de l'envoi
    """
    target = PAGE_CONFIG[level]
    
    try:
        response = requests.post(
            target['url'],
            json={
                'query': query,
                'classification': classification
            },
            timeout=5
        )
        
        if response.status_code == 200:
            return {
                'success': True,
                'message': f'✅ Requête envoyée au serveur {level.upper()} (port {target["port"]})',
                'response': response.json()
            }
        else:
            return {
                'success': False,
                'message': f'❌ Serveur {level.upper()} a retourné une erreur',
                'status_code': response.status_code
            }
            
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'message': f'❌ Serveur {level.upper()} (port {target["port"]}) non disponible. Lancez: python test_{level}_server.py'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'❌ Erreur: {str(e)}'
        }


# ==========================================
# ENDPOINTS API
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de santé"""
    return jsonify({
        "status": "ok",
        "pipeline": "comprehension + complexity",
        "port": 5000,
        "agents": {
            "comprehension": "active",
            "complexity": "active (Embeddings)"
        },
        "accuracy": round(metrics['accuracy'], 3)
    }), 200


@app.route("/process", methods=["POST"])
def process_query():
    """
    Endpoint principal : Pipeline complet
    
    Body JSON:
    {
        "query": "Scan all ports using SYN scan"
    }
    
    OU
    
    {
        "user_query": "Scan all ports using SYN scan"
    }
    
    Response:
    {
        "comprehension": {
            "is_nmap": true,
            "confidence": 0.8
        },
        "classification": {
            "level": "medium",
            "confidence": 0.94
        },
        "routing": {
            "target_server": "localhost:5005",
            "port": 5005,
            "model": "LoRA fine-tuned"
        },
        "delivery": {
            "success": true,
            "message": "✅ Requête envoyée..."
        }
    }
    """
    data = request.get_json()
    
    # Vérifier que le body existe
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    # Accepter 'query' OU 'user_query'
    user_query = data.get("query") or data.get("user_query")
    
    if not user_query:
        return jsonify({
            "error": "Missing 'query' or 'user_query' field",
            "example": {
                "query": "nmap -p 80 192.168.1.1"
            }
        }), 400
    
    try:
        # ===================================
        # ÉTAPE 1: COMPREHENSION
        # ===================================
        comprehension_result = check_nmap_context(user_query)
        
        if not comprehension_result['is_nmap']:
            return jsonify({
                "error": "Requête non-Nmap",
                "message": "Votre demande ne concerne pas Nmap. Exemple: 'Scanner les ports ouverts sur 192.168.1.1'",
                "comprehension": comprehension_result
            }), 400
        
        # ===================================
        # ÉTAPE 2: COMPLEXITY (SLM + Embeddings)
        # ===================================
        classification_result = classifier.classify(user_query)
        level = classification_result['level']
        
        # ===================================
        # ÉTAPE 3: ROUTAGE ET ENVOI
        # ===================================
        page_info = PAGE_CONFIG[level]
        
        # Envoyer automatiquement au serveur cible
        send_result = send_to_target_server(
            user_query, 
            level, 
            classification_result
        )
        
        response = {
            "query": user_query,
            "comprehension": comprehension_result,
            "classification": {
                "level": classification_result['level'],
                "confidence": classification_result['confidence'],
                "probabilities": classification_result['probabilities'],
                "reasoning": classification_result['reasoning'],
                "recommended_model": classification_result['recommended_model'],
                "method": classification_result['method']
            },
            "routing": {
                "target_server": f"localhost:{page_info['port']}",
                "port": page_info['port'],
                "model": page_info['model'],
                "color": page_info['color'],
                "url": page_info['url']
            },
            "delivery": send_result,
            "message": f"{page_info['color']} Classification: {level.upper()} ({classification_result['confidence']*100:.0f}%) → {send_result['message']}"
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Erreur lors du traitement"
        }), 500


@app.route("/comprehension", methods=["POST"])
def comprehension_only():
    """Endpoint uniquement pour la compréhension"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    # Accepter 'query' OU 'user_query'
    user_query = data.get("query") or data.get("user_query")
    
    if not user_query:
        return jsonify({
            "error": "Missing 'query' or 'user_query' field"
        }), 400
    
    result = check_nmap_context(user_query)
    
    return jsonify(result), 200


@app.route("/complexity", methods=["POST"])
def complexity_only():
    """Endpoint uniquement pour la complexité"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    # Accepter 'query' OU 'user_query'
    user_query = data.get("query") or data.get("user_query")
    
    if not user_query:
        return jsonify({
            "error": "Missing 'query' or 'user_query' field"
        }), 400
    
    try:
        result = classifier.classify(user_query)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  NMAP-AI PIPELINE SERVER")
    print("="*70)
    print("\n🔗 Pipeline complet:")
    print("  1️⃣  Agent Compréhension (vérifie si Nmap)")
    print("  2️⃣  Agent Complexité (SLM + Embeddings)")
    print("  3️⃣  Routage automatique (EASY/MEDIUM/HARD)")
    print("\n📡 Endpoints:")
    print("  → GET  /health           - Statut du pipeline")
    print("  → POST /process          - Pipeline complet (recommandé)")
    print("  → POST /comprehension    - Compréhension seule")
    print("  → POST /complexity       - Complexité seule")
    print("\n🧠 Agent Complexité:")
    print(f"  → Méthode: {metrics['method']}")
    print(f"  → Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  → Dimension: {metrics['embedding_dim']}")
    print(f"  → Device: {metrics['device']}")
    print("\n🎯 Routage vers:")
    for level, info in PAGE_CONFIG.items():
        print(f"  {info['color']} {level.upper():6} → {info['url']} (port {info['port']})")
    print("\n🌐 Serveur démarré sur http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False)