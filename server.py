from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

from flask import Flask, jsonify, request, send_from_directory, abort
import yaml
import argparse
from datetime import datetime

from hospital_sim.simulation import HospitalSim, load_config, SimConfig, StageConfig
from optimize import run_optimization

BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
DESC_FILE = OUTPUTS_DIR / "stage_descriptions.json"

app = Flask(__name__, static_folder=None)


# ---- History helpers (placed after OUTPUTS_DIR definition) ----
HISTORY_DIR = OUTPUTS_DIR / "history"

def _now_stamp() -> str:
    # Safe for filenames
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def save_history(kind: str, df, kpi: Dict[str, Any], best: Dict[str, Any] | None = None):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = _now_stamp()
    run_dir = HISTORY_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    # Write files
    df.to_csv(run_dir / "franjas_patients.csv", index=False)
    (run_dir / "franjas_kpis.json").write_text(json.dumps(kpi, indent=2), encoding="utf-8")
    if best is not None:
        (run_dir / "best_trial.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    # Snapshot current base config
    if BASE_CONFIG.exists():
        (run_dir / "config_snapshot.yaml").write_text(BASE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    # Snapshot optimized config if present
    opt_path = CONFIGS_DIR / "optimized.yaml"
    if opt_path.exists():
        (run_dir / "optimized.yaml").write_text(opt_path.read_text(encoding="utf-8"), encoding="utf-8")
    # Update index
    index_path = HISTORY_DIR / "index.json"
    index = []
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = []
    entry = {
        "timestamp": ts,
        "kind": kind,
        "kpis": kpi,
        "paths": {
            "dir": f"history/{ts}",
            "patients": f"history/{ts}/franjas_patients.csv",
            "kpis": f"history/{ts}/franjas_kpis.json",
            "best": f"history/{ts}/best_trial.json" if best is not None else None,
            "config": f"history/{ts}/config_snapshot.yaml" if (run_dir / "config_snapshot.yaml").exists() else None,
            "optimized": f"history/{ts}/optimized.yaml" if (run_dir / "optimized.yaml").exists() else None,
        },
    }
    index.insert(0, entry)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

@app.get("/api/history")
def api_history():
    index_path = HISTORY_DIR / "index.json"
    if not index_path.exists():
        return jsonify([]), 200
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        data = []
    return jsonify(data), 200


@app.get("/api/base_yaml")
def api_base_yaml():
    if not BASE_CONFIG.exists():
        return jsonify({"error": "base.yaml not found"}), 404
    text = BASE_CONFIG.read_text(encoding="utf-8")
    return jsonify({"content": text, "path": str(BASE_CONFIG)}), 200

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


@app.get("/api/optimized_yaml")
def api_optimized_yaml():
    path = CONFIGS_DIR / "optimized.yaml"
    if not path.exists():
        return jsonify({"error": "optimized.yaml not found"}), 404
    text = path.read_text(encoding="utf-8")
    return jsonify({"content": text, "path": str(path)}), 200


@app.get("/download/optimized.yaml")
def download_optimized_yaml():
    path = CONFIGS_DIR / "optimized.yaml"
    if not path.exists():
        abort(404)
    return send_from_directory(CONFIGS_DIR, "optimized.yaml", as_attachment=True)


@app.post("/api/apply_optimized")
def apply_optimized():
    src = CONFIGS_DIR / "optimized.yaml"
    if not src.exists():
        return jsonify({"error": "optimized.yaml not found"}), 404
    # Overwrite base.yaml with optimized contents
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    BASE_CONFIG.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return jsonify({"ok": True, "applied_to": str(BASE_CONFIG)}), 200


@app.post("/api/recalc")
def recalc():
    try:
        # Load config and run simulation
        if not BASE_CONFIG.exists():
            return jsonify({"error": "configs/base.yaml not found"}), 404
        cfg = load_config(str(BASE_CONFIG))
        sim = HospitalSim(cfg)
        df = sim.run()
        kpi = sim.kpis(df)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        # Timestamped prefix optional; keep canonical filenames for dashboard
        df.to_csv(OUTPUTS_DIR / "franjas_patients.csv", index=False)
        (OUTPUTS_DIR / "franjas_kpis.json").write_text(json.dumps(kpi, indent=2), encoding="utf-8")
        # Save history
        save_history("recalc", df, kpi)
        return jsonify({"ok": True, "kpis": kpi}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/optimize")
def optimize_api():
    try:
        if not BASE_CONFIG.exists():
            return jsonify({"error": "configs/base.yaml not found"}), 404
        payload = request.get_json(force=True, silent=False) or {}
        n_trials = int(payload.get("n_trials", 30))
        capacity_bounds = payload.get("capacity_bounds") or {}
        weights = payload.get("weights") or {"los_mean": 1.0, "wait_p95": 0.1, "cap_penalty_per_server": 0.0}
        optimize_mode = (payload.get("optimize_mode") or "fixed").lower()  # 'fixed' or 'schedule'
        stages_to_opt = payload.get("stages_to_opt")  # optional list

        # Run optimization (fixed vs schedule)
        if optimize_mode == "schedule":
            from optimize import run_optimization_schedule
            study = run_optimization_schedule(
                config_path=str(BASE_CONFIG),
                n_trials=n_trials,
                stages_to_opt=stages_to_opt,
                capacity_bounds=capacity_bounds,
                weights=weights,
                storage=None,
            )
        else:
            study = run_optimization(
                config_path=str(BASE_CONFIG),
                n_trials=n_trials,
                stages_to_opt=stages_to_opt,
                capacity_bounds=capacity_bounds,
                weights=weights,
                storage=None,
            )
        best = {"value": study.best_trial.value, "params": study.best_trial.params}

        # Confirm simulation using best params
        base_cfg = load_config(str(BASE_CONFIG))
        new_stages = []
        for s in base_cfg.stages:
            if optimize_mode == "schedule":
                # Reconstruct 24h schedule if present in params; fallback to existing
                sched_keys = [f"cap_{s.name}_h{h:02d}" for h in range(24)]
                has_all = all(k in study.best_trial.params for k in sched_keys)
                if has_all:
                    sched = [int(study.best_trial.params[k]) for k in sched_keys]
                    base_cap = max(sched)
                    new_stages.append(StageConfig(name=s.name, capacity=int(base_cap), service_time=s.service_time, capacity_schedule=sched))
                else:
                    new_stages.append(s)
            else:
                cap = study.best_trial.params.get(f"cap_{s.name}", s.capacity)
                new_stages.append(StageConfig(name=s.name, capacity=int(cap), service_time=s.service_time, capacity_schedule=s.capacity_schedule))
        trial_cfg = SimConfig(
            seed=base_cfg.seed,
            sim_hours=base_cfg.sim_hours,
            warmup_hours=base_cfg.warmup_hours,
            stages=new_stages,
            arrivals_per_hour=base_cfg.arrivals_per_hour,
            arrivals_profile=base_cfg.arrivals_profile,
            hourly_arrivals=base_cfg.hourly_arrivals,
        )
        sim = HospitalSim(trial_cfg)
        df = sim.run()
        kpi = sim.kpis(df)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        # Save outputs
        df.to_csv(OUTPUTS_DIR / "franjas_patients.csv", index=False)
        (OUTPUTS_DIR / "franjas_kpis.json").write_text(json.dumps(kpi, indent=2), encoding="utf-8")
        # Save best trial summary
        (OUTPUTS_DIR / "best_trial.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

        # Save optimized configuration YAML
        optimized_cfg = {
            "seed": base_cfg.seed,
            "sim_hours": base_cfg.sim_hours,
            "warmup_hours": base_cfg.warmup_hours,
            "arrivals_per_hour": base_cfg.arrivals_per_hour,
            "arrivals_profile": base_cfg.arrivals_profile,
            "hourly_arrivals": base_cfg.hourly_arrivals,
            "stages": [
                {"name": st.name, "capacity": st.capacity, "service_time": st.service_time, **({"capacity_schedule": st.capacity_schedule} if getattr(st, "capacity_schedule", None) else {})}
                for st in trial_cfg.stages
            ],
        }
        (CONFIGS_DIR / "optimized.yaml").write_text(
            yaml.safe_dump(optimized_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        # Save history for optimize
        save_history("optimize", df, kpi, best)

        return jsonify({"ok": True, "best": best, "kpis": kpi, "optimized_config": "configs/optimized.yaml", "best_trial": "outputs/best_trial.json", "mode": optimize_mode}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/smooth_optimized")
def smooth_optimized():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        window = int(payload.get("window", 3))
        if window < 1:
            return jsonify({"error": "window must be >= 1"}), 400
        opt_path = CONFIGS_DIR / "optimized.yaml"
        if not opt_path.exists():
            return jsonify({"error": "optimized.yaml not found"}), 404
        cfg = yaml.safe_load(opt_path.read_text(encoding="utf-8")) or {}
        stages = cfg.get("stages", [])
        changed = False
        for st in stages:
            sched = st.get("capacity_schedule")
            if isinstance(sched, list) and len(sched) == 24:
                # circular moving average
                sm = []
                for i in range(24):
                    acc = 0.0
                    for k in range(-(window//2), window - (window//2)):
                        acc += float(sched[(i + k) % 24])
                    sm.append(int(round(acc / window)))
                st["capacity_schedule"] = sm
                st["capacity"] = int(max(sm))
                changed = True
        if not changed:
            return jsonify({"error": "no schedules to smooth"}), 400
        opt_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return jsonify({"ok": True, "optimized_config": "configs/optimized.yaml"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Config hourly_arrivals in configs/base.yaml ---
CONFIGS_DIR = BASE_DIR / "configs"
BASE_CONFIG = CONFIGS_DIR / "base.yaml"


@app.get("/api/config_hourly")
def get_config_hourly():
    if not BASE_CONFIG.exists():
        return jsonify({"hourly_arrivals": None, "error": "configs/base.yaml not found"}), 404
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    hourly = cfg.get("hourly_arrivals")
    return jsonify({"hourly_arrivals": hourly}), 200


@app.post("/api/config_hourly")
def set_config_hourly():
    try:
        data = request.get_json(force=True, silent=False)
        arr = data.get("hourly_arrivals")
        if not isinstance(arr, list) or len(arr) != 24:
            return jsonify({"error": "hourly_arrivals must be a list of 24 numbers"}), 400
        # coerce to float
        arr = [float(x) for x in arr]
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        cfg = {}
        if BASE_CONFIG.exists():
            with open(BASE_CONFIG, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        cfg["hourly_arrivals"] = arr
        # Ensure arrivals_per_hour/profile remain if present
        with open(BASE_CONFIG, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    # Run on 0.0.0.0 to make it accessible to browser preview
    app.run(host="0.0.0.0", port=args.port, debug=False)
