from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

from flask import Flask, jsonify, request, send_from_directory, abort

BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
DESC_FILE = OUTPUTS_DIR / "stage_descriptions.json"

app = Flask(__name__, static_folder=None)


@app.get("/api/stage_descriptions")
def get_stage_descriptions():
    if not DESC_FILE.exists():
        return jsonify({}), 200
    try:
        data = json.loads(DESC_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    return jsonify(data), 200


@app.post("/api/stage_descriptions")
def save_stage_descriptions():
    try:
        data: Dict[str, Any] = request.get_json(force=True, silent=False)  # expect a dict
        if not isinstance(data, dict):
            return jsonify({"error": "JSON must be an object {stage: description}"}), 400
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        DESC_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Static files under outputs/
@app.get("/")
def index():
    # List directory or redirect to dashboard if present
    dashboard = OUTPUTS_DIR / "franjas_dashboard.html"
    if dashboard.exists():
        return send_from_directory(OUTPUTS_DIR, "franjas_dashboard.html")
    # fallback: directory listing
    files = sorted([p.name for p in OUTPUTS_DIR.iterdir()])
    return ("<h1>Outputs</h1><ul>" + "".join(f"<li><a href='/{name}'>{name}</a></li>" for name in files) + "</ul>")


@app.get("/<path:filename>")
def serve_outputs(filename: str):
    target = OUTPUTS_DIR / filename
    if not target.exists():
        abort(404)
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    # Run on 0.0.0.0 to make it accessible to browser preview
    app.run(host="0.0.0.0", port=8001, debug=False)
