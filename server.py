"""
server.py — Flask communication layer between the JS browser game and eee.py.

Endpoints:
  GET  /state         → returns the latest game state pushed by the JS game
  POST /action        → stores an action string for the agent to consume
  POST /update_state  → JS game pushes a new GameState JSON payload

The Flask server and the EEE agent loop run concurrently via threading.
"""

import threading
from flask import Flask, request, jsonify
from flask_cors import CORS

import eee

app = Flask(__name__)
CORS(app)

# Module-level state shared between Flask endpoints and the agent
_current_state: dict | None = None
_pending_action: str | None = None
_state_lock = threading.Lock()
_action_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Flask endpoints
# ---------------------------------------------------------------------------

@app.get("/state")
def get_state():
    with _state_lock:
        if _current_state is None:
            return jsonify({"error": "No state available yet"}), 404
        return jsonify(_current_state)


@app.post("/update_state")
def update_state():
    payload = request.get_json(force=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON"}), 400
    with _state_lock:
        global _current_state
        _current_state = payload
    return jsonify({"status": "ok"})


@app.post("/action")
def receive_action():
    payload = request.get_json(force=True)
    if payload is None or "action" not in payload:
        return jsonify({"error": "Missing 'action' field"}), 400
    with _action_lock:
        global _pending_action
        _pending_action = payload["action"]
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flask_thread = threading.Thread(
        target=lambda: app.run(host="localhost", port=5000, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()

    eee.run_agent(game_id="test_game_01", max_turns=50)
