"""
Serveur de Test EASY - Port 5004
=================================

Ce serveur reçoit les requêtes EASY et affiche les informations.

Usage:
    python test_easy_server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Compteur de requêtes
request_count = 0


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "level": "EASY",
        "port": 5004,
        "requests_received": request_count
    }), 200


@app.route("/process", methods=["POST"])
def process():
    """Reçoit et affiche les requêtes EASY"""
    global request_count
    request_count += 1
    
    data = request.get_json()
    query = data.get('query', data.get('user_query', 'Unknown'))
    
    # Afficher dans le terminal
    print("\n" + "="*60)
    print(f" REQUÊTE EASY REÇUE #{request_count}")
    print("="*60)
    print(f" Heure: {datetime.now().strftime('%H:%M:%S')}")
    print(f" Query: {query}")
    print(f" Classification: {data.get('classification', {})}")
    print(f" Port: 5004")
    print(f" Modèle: KG-RAG (Knowledge Graph)")
    print("="*60 + "\n")
    
    return jsonify({
        "status": "received",
        "level": "EASY",
        "port": 5004,
        "query": query,
        "message": "✅ Requête EASY reçue avec succès",
        "request_number": request_count
    }), 200


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🟢 SERVEUR TEST EASY")
    print("="*60)
    print("  Port: 5004")
    print("  Niveau: EASY")
    print("  Modèle: KG-RAG (Knowledge Graph)")
    print("\n  ⏳ En attente de requêtes...")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5004, debug=False)