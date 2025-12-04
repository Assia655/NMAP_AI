from flask import Flask, request, jsonify
from agent_comprehension import comprehension_agent

app = Flask(__name__)

@app.route("/comprehension", methods=["POST"])
def comprehension():
    data = request.get_json()

    if not data or "user_query" not in data:
        return jsonify({"error": "Missing 'user_query'"}), 400

    user_query = data["user_query"]

    # Appeler ton agent statique TF-IDF
    result = comprehension_agent(user_query)

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
