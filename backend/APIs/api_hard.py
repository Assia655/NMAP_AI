from flask import Blueprint, request, jsonify

# Use package-relative import so it works when run from backend/api.py
from Agents.Agent_hard import HardDiffusionAgent

api_hard = Blueprint("api_hard", __name__)
agent = HardDiffusionAgent(seed=42)


@api_hard.route("/hard", methods=["POST"])
def hard_agent():
    data = request.get_json(silent=True) or {}

    query = (data.get("query") or "").strip()
    target = (data.get("target") or "").strip()

    if not query or not target:
        return jsonify({"ok": False, "error": "Missing 'query' or 'target'"}), 400

    result = agent.generate(query, target)
    return jsonify(result), 200
