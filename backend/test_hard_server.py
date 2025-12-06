"""
Serveur de Test HARD - Port 5006
=================================

Ce serveur reçoit les requêtes HARD et affiche les informations.

Usage:
    python test_hard_server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

request_count = 0


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "level": "HARD",
        "port": 5006,
        "requests_received": request_count
    }), 200


@app.route("/process", methods=["POST"])
def process():
    """Reçoit et affiche les requêtes HARD"""
    global request_count
    request_count += 1
    
    data = request.get_json()
    query = data.get('query', data.get('user_query', 'Unknown'))
    
    # Afficher dans le terminal
    print("\n" + "="*60)
    print(f"🔴 REQUÊTE HARD REÇUE #{request_count}")
    print("="*60)
    print(f"⏰ Heure: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📝 Query: {query}")
    print(f"📊 Classification: {data.get('classification', {})}")
    print(f"🎯 Port: 5006")
    print(f"🤖 Modèle: Diffusion-based synthesis")
    print("="*60 + "\n")
    
    return jsonify({
        "status": "received",
        "level": "HARD",
        "port": 5006,
        "query": query,
        "message": "✅ Requête HARD reçue avec succès",
        "request_number": request_count
    }), 200


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🔴 SERVEUR TEST HARD")
    print("="*60)
    print("  Port: 5006")
    print("  Niveau: HARD")
    print("  Modèle: Diffusion-based synthesis")
    print("\n  ⏳ En attente de requêtes...")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5006, debug=False)