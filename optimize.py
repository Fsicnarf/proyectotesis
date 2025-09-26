from pathlib import Path
from typing import Dict, Any, List

import optuna

from hospital_sim.simulation import HospitalSim, load_config, SimConfig, StageConfig


def objective_factory(base_cfg: SimConfig, stages_to_opt: List[str], capacity_bounds: Dict[str, Any], weights: Dict[str, float]):
    def objective(trial: optuna.Trial) -> float:
        # Create a modified copy of config each trial
        new_stages = []
        for s in base_cfg.stages:
            if s.name in stages_to_opt:
                lb, ub = capacity_bounds.get(s.name, (1, 10))
                cap = trial.suggest_int(f"cap_{s.name}", lb, ub)
                new_stages.append(StageConfig(name=s.name, capacity=cap, service_time=s.service_time))
            else:
                new_stages.append(s)
        trial_cfg = SimConfig(
            seed=base_cfg.seed,
            sim_hours=base_cfg.sim_hours,
            warmup_hours=base_cfg.warmup_hours,
            arrivals_per_hour=base_cfg.arrivals_per_hour,
            stages=new_stages,
        )

        sim = HospitalSim(trial_cfg)
        df = sim.run()
        kpi = sim.kpis(df)

        # Weighted objective: minimize LOS mean and P95 waits across stages
        # cost = w1*los_mean + w2*los_p95 + sum_s (w_wait * wait_p95_s)
        cost = 0.0
        cost += weights.get("los_mean", 1.0) * kpi.get("los_mean", 0.0)
        cost += weights.get("los_p95", 0.0) * kpi.get("los_p95", 0.0)
        w_wait = weights.get("wait_p95", 0.0)
        for s in trial_cfg.stages:
            cost += w_wait * kpi.get(f"wait_{s.name}_p95", 0.0)

        # Optional soft capacity penalty: discourage overcapacity
        cap_penalty_w = weights.get("cap_penalty_per_server", 0.0)
        if cap_penalty_w > 0:
            total_cap = sum(st.capacity for st in trial_cfg.stages)
            cost += cap_penalty_w * total_cap

        return float(cost)

    return objective


def run_optimization(config_path: str, n_trials: int = 40, stages_to_opt: List[str] = None,
                     capacity_bounds: Dict[str, Any] = None, weights: Dict[str, float] = None,
                     storage: str = None, study_name: str = "hospital_capacity_opt"):
    base_cfg = load_config(config_path)

    if stages_to_opt is None:
        stages_to_opt = [s.name for s in base_cfg.stages]
    if capacity_bounds is None:
        capacity_bounds = {s: (1, 10) for s in stages_to_opt}
    if weights is None:
        weights = {"los_mean": 1.0, "wait_p95": 0.1, "cap_penalty_per_server": 0.0}

    sampler = optuna.samplers.TPESampler(seed=base_cfg.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name, storage=storage, load_if_exists=bool(storage))

    study.optimize(objective_factory(base_cfg, stages_to_opt, capacity_bounds, weights), n_trials=n_trials)

    return study


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Optimize hospital capacities using Optuna")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///opt.db)")
    args = parser.parse_args()

    study = run_optimization(config_path=args.config, n_trials=args.trials, storage=args.storage)

    print("Best trial:")
    print(json.dumps({
        "value": study.best_trial.value,
        "params": study.best_trial.params
    }, indent=2))
