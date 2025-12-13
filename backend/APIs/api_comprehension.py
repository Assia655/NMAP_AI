# ===============================
#  NMAP Comprehension Agent API
#  Version corrigée et stable
# ===============================

from flask import Flask, request, jsonify
from flasgger import Swagger
import os
import sys

# === Assurer l'import correct du backend ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# === Import de l'agent ===
try:
    from Agents.Agent_comprehension.nmap_agent_embeddings import NMAPEmbeddingAgent
    print("✅ Importation du NMAPEmbeddingAgent réussie")
except Exception as e:
    print(f"❌ ERREUR import agent : {e}")
    NMAPEmbeddingAgent = None

app = Flask(__name__)

swagger = Swagger(app, template={
    "info": {
        "title": "NMAP Comprehension Agent API",
        "description": "Analyse sémantique intelligente pour détecter les requêtes NMAP",
        "version": "2.0.0"
    }
})

comprehension_agent = None


# =============================================
# 🔥 Fonction pour charger l'agent avec bon chemin
# =============================================
def create_app():
    global comprehension_agent
    print("\n==============================")
    print("🚀 Initialisation de l'Agent NMAP Comprehension")
    print("==============================")

    # 1 — Récupérer le chemin du dossier backend
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 2 — Construire le chemin ABSOLU du fichier nmap_domain.txt
    agent_path = os.path.join(
        backend_dir,
        "Agents",
        "Agent_comprehension",
        "nmap_domain.txt"
    )

    print(f"📁 Chemin ABSOLU du fichier : {agent_path}")
    print(f"📌 Fichier existe ? → {os.path.exists(agent_path)}")

    try:
        comprehension_agent = NMAPEmbeddingAgent(agent_path)
        print("✅ Agent chargé avec succès !")

    except Exception as e:
        print(f"❌ ERREUR lors du chargement de l'agent : {e}")
        comprehension_agent = None

    return app


# =============================================
#  📌 ROUTE PRINCIPALE : /comprehension
# =============================================
@app.route("/comprehension", methods=["POST"])
def comprehension():
    global comprehension_agent

    if comprehension_agent is None:
        return jsonify({
            "success": False,
            "error": "Agent non initialisé",
            "message": "Impossible de charger nmap_domain.txt ou spaCy"
        }), 500

    try:
        data = request.get_json(silent=True)

        if not data or "user_query" not in data:
            return jsonify({
                "success": False,
                "error": "Champ manquant",
                "message": "Le champ 'user_query' est obligatoire"
            }), 400

        user_query = data["user_query"].strip()

        if not user_query:
            return jsonify({
                "success": False,
                "error": "Requête vide"
            }), 400

        result = comprehension_agent.understand_query(user_query)

        return jsonify({
            "success": True,
            "data": result
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Erreur interne",
            "message": str(e)
        }), 500


@app.route("/")
def home():
    return """
    <h1>NMAP Comprehension Agent API</h1>
    <p>Swagger UI : <a href="/apidocs">/apidocs</a></p>
    <p>Endpoint : POST /comprehension</p>
    """


# =============================================
#  🚀 LANCEMENT DU SERVEUR
# =============================================
if __name__ == "__main__":
    app = create_app()
    print("🌍 Serveur lancé : http://localhost:5001")
    print("📘 Swagger UI : http://localhost:5001/apidocs")
    app.run(host="0.0.0.0", port=5001, debug=False)
