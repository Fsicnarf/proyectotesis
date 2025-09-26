# Hospital Dynamic Simulation and Capacity Optimization

This repository provides a discrete-event simulation (DES) of patient flow in a hospital using SimPy, plus an optimization loop (Optuna) to search for optimal capacities per stage.

Time unit: hours.

## Project Structure

- `requirements.txt` — Python dependencies.
- `hospital_sim/` — Python package with the simulation engine.
  - `simulation.py` — core DES model and KPI computation.
  - `__init__.py` — package exports.
- `configs/` — YAML configuration files.
  - `base.yaml` — example configuration.
- `cli.py` — command-line interface to run simulations and export results.
- `optimize.py` — Optuna-based capacity optimization.

## Install

Use Python 3.10+ recommended.

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Run a Simulation

```bash
python cli.py run --config configs/base.yaml --outdir outputs --prefix test1
```

This command writes:

- `outputs/test1_patients.csv` — detailed per-patient data.
- `outputs/test1_kpis.json` — KPIs (throughput, LOS mean/p95, stage wait mean/p95).

## Configuration

Edit `configs/base.yaml` to change:

- `sim_hours`, `warmup_hours`, `seed`.
- `arrivals_per_hour` (Poisson arrivals, base rate).
- Time-of-day arrivals (franjas horarias) — choose one of:
  - `arrivals_profile` (24 valores): perfil multiplicativo por hora del día. La tasa efectiva es `arrivals_per_hour * arrivals_profile[hour]`.
  - `hourly_arrivals` (24 valores): tasas absolutas por hora; si está presente, anula `arrivals_per_hour` y `arrivals_profile`.
- `stages`: list of stages, each with `name`, `capacity`, and `service_time` distribution:
  - Supported distributions: `exponential` (params: `rate` or `mean`), `normal` (`mean`, `std`), `lognormal` (`mean`, `sigma`), `uniform` (`low`, `high`).

Example (franjas horarias):

```yaml
arrivals_per_hour: 8
arrivals_profile: [
  0.3, 0.3, 0.3, 0.3,
  0.4, 0.5,
  0.8, 1.0, 1.2, 1.3,
  1.4, 1.5, 1.5,
  1.3, 1.2, 1.1,
  1.0, 0.9, 0.8,
  0.7, 0.6, 0.5, 0.4,
  0.3
]

# Overrides the profile if provided
# hourly_arrivals: [4,4,4,4, 5,6, 8,9,10,11, 12,12,12, 11,10,9, 8,7,6, 5,5,4,3, 3]
```

## KPI Definitions

- `throughput`: number of patients with non-null departure time after warmup.
- `los_mean`, `los_p95`: length of stay statistics (arrival to departure).
- `wait_{stage}_mean`, `wait_{stage}_p95`: waiting time before each stage.

## Optimization

Run Optuna to optimize capacities:

```bash
python optimize.py --config configs/base.yaml --trials 50
```

- Objective minimizes a weighted sum of LOS and stage wait P95, with an optional soft penalty per server.
- Edit weights and bounds in `optimize.py` by passing parameters to `run_optimization()` if using as a library, or modify the defaults.

Example (as library):

```python
from optimize import run_optimization
study = run_optimization("configs/base.yaml", n_trials=50)
print(study.best_trial.params)
```

## Notes and Next Steps

- The baseline model is a simple sequential flow (triage -> consultation -> diagnostics -> treatment). You can add branching paths or probabilities by modifying `HospitalSim.patient_flow()`.
- To track utilization precisely, consider adding resource monitors to record busy times.
- Validate parameters with real hospital data to calibrate distributions.
- Arrivals use a non-homogeneous Poisson process (NHPP) with thinning, using a 24h ciclo para `arrivals_profile`/`hourly_arrivals`.
