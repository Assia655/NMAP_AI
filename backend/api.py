"""Unified Flask app exposing comprehension and hard diffusion endpoints."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flasgger import Swagger

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from Agents.Agent_comprehension.nmap_agent_embeddings import NMAPEmbeddingAgent
except Exception as exc:
    print(f"[api] Failed to import NMAPEmbeddingAgent: {exc}")
    NMAPEmbeddingAgent = None  # type: ignore

from APIs.api_hard import api_hard


def build_swagger(app: Flask) -> None:
    Swagger(
        app,
        template={
            "info": {
                "title": "NMAP AI API",
                "description": "Comprehension and hard diffusion agents",
                "version": "2.0.0",
            }
        },
    )


def init_comprehension_agent() -> NMAPEmbeddingAgent | None:
    if NMAPEmbeddingAgent is None:
        return None
    domain_path = BASE_DIR / "Agents" / "Agent_comprehension" / "nmap_domain.txt"
    try:
        agent = NMAPEmbeddingAgent(str(domain_path))
        print(f"[api] Comprehension agent loaded from {domain_path}")
        return agent
    except FileNotFoundError:
        print(f"[api] Domain file not found: {domain_path}")
    except Exception as exc:
        print(f"[api] Failed to load comprehension agent: {exc}")
    return None


def create_app() -> Flask:
    app = Flask(__name__)
    build_swagger(app)
    comprehension_agent = init_comprehension_agent()
    app.register_blueprint(api_hard, url_prefix="/api")

    @app.route("/comprehension", methods=["POST"])
    def comprehension():
        if comprehension_agent is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Agent not initialized",
                        "message": "nmap_domain.txt missing or spaCy not installed",
                    }
                ),
                500,
            )

        data = request.get_json(silent=True) or {}
        user_query = (data.get("user_query") or "").strip()
        if not user_query:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Missing field",
                        "message": "The 'user_query' field is required",
                    }
                ),
                400,
            )

        try:
            result = comprehension_agent.understand_query(user_query)
        except Exception as exc:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Internal error",
                        "message": str(exc),
                    }
                ),
                500,
            )

        return jsonify({"success": True, "data": result}), 200

    @app.route("/")
    def home():
        return """
        <h1>NMAP AI API</h1>
        <p>Swagger UI: <a href="/apidocs">/apidocs</a></p>
        <ul>
          <li>POST /comprehension</li>
          <li>POST /api/hard</li>
        </ul>
        """

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    print(f"[api] Server starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
