from flask import Blueprint, request, jsonify
from Agents.Agent_complexité.agent_complexity_word2vec import ComplexityClassifier

api_complexity = Blueprint("api_complexity", __name__)
classifier = ComplexityClassifier()

@api_complexity.route("/complexity", methods=["POST"])
def complexity():
    data = request.get_json(silent=True) or {}
    query = data.get("user_query", "").strip()

    if not query:
        return jsonify({"success": False, "error": "Missing user_query"}), 400

    level, score = classifier.predict(query)

    return jsonify({
        "success": True,
        "complexity": level,
        "confidence": score
    }), 200
