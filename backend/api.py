# api.py  ← Fichier corrigé pour Flask 3.0+
from flask import Flask, request, jsonify
from flasgger import Swagger
import os
import sys

# Ajouter le répertoire courant pour importer l'agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import de ton agent
try:
    from nmap_agent_embeddings import NMAPEmbeddingAgent
    print("Agent importé avec succès")
except Exception as e:
    print(f"Erreur import agent : {e}")
    NMAPEmbeddingAgent = None

app = Flask(__name__)

# Configuration Swagger
swagger = Swagger(app, template={
    "info": {
        "title": "NMAP Comprehension Agent API",
        "description": "Analyse sémantique intelligente pour détecter les requêtes Nmap",
        "version": "2.0.0"
    }
})

# Variable globale pour l'agent
comprehension_agent = None


# CHANGEMENT ICI : on charge l'agent au démarrage, PAS avec before_first_request
def create_app():
    global comprehension_agent
    print("Initialisation de l'agent NMAP...")
    try:
        comprehension_agent = NMAPEmbeddingAgent("nmap_domain.txt")
        print("Agent chargé avec succès !")
    except FileNotFoundError:
        print("Fichier nmap_domain.txt introuvable !")
        comprehension_agent = None
    except Exception as e:
        print(f"Erreur chargement agent : {e}")
        comprehension_agent = None
    return app


@app.route("/comprehension", methods=["POST"])
def comprehension():
    """
    Analyser une requête utilisateur pour vérifier si elle est liée à NMAP
    ---
    tags:
      - Analysis
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            user_query:
              type: string
              example: "scanner le port 80"
          required:
            - user_query
    responses:
      200:
        description: Analyse réussie
      400:
        description: Erreur de validation
      500:
        description: Erreur serveur
    """
    global comprehension_agent

    if comprehension_agent is None:
        return jsonify({
            "success": False,
            "error": "Agent non initialisé",
            "message": "Fichier nmap_domain.txt manquant ou spaCy non installé"
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
                "error": "Requête vide",
                "message": "La requête ne peut pas être vide"
            }), 400

        result = comprehension_agent.understand_query(user_query)

        return jsonify({
            "success": True,
            "data": {
                "is_relevant": result['is_relevant'],
                "decision": result['decision'],
                "query": result['query'],
                "confidence": result['confidence'],
                "scores": result['scores'],
                "analysis": result['analysis']
            }
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
    <pre>
{
  "user_query": "scanner les ports 80 et 443 sur 192.168.1.0/24"
}
    </pre>
    """


# Lancement propre pour Flask 3.0+
if __name__ == "__main__":
    # On charge l'agent AVANT de lancer le serveur
    app = create_app()
    print("Démarrage du serveur sur http://localhost:5001")
    print("Swagger UI → http://localhost:5001/apidocs")
    app.run(host="0.0.0.0", port=5001, debug=False)