import math
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple

import numpy as np
import pandas as pd
import simpy
import yaml


# -----------------------------
# Utility: Distributions
# -----------------------------

Distribution = Callable[[random.Random], float]


def make_distribution(cfg: Dict[str, Any]) -> Distribution:
    kind = cfg.get("type", "exponential").lower()
    params = cfg.get("params", {})

    def expo(rng: random.Random) -> float:
        rate = params.get("rate")
        mean = params.get("mean")
        if rate is not None:
            return rng.expovariate(rate)
        if mean is not None and mean > 0:
            return rng.expovariate(1.0 / mean)
        raise ValueError("Exponential distribution requires 'rate' or positive 'mean'.")

    def normal(rng: random.Random) -> float:
        mu = params.get("mean", 1.0)
        sigma = params.get("std", 0.1)
        return max(0.0, rng.gauss(mu, sigma))

    def lognormal(rng: random.Random) -> float:
        mean = params.get("mean", 1.0)
        sigma = params.get("sigma", 0.25)
        # Convert mean/sigma in real space to mu/s for log-space approx
        # Using approximation: if X~logN(mu,s), E[X]=exp(mu+s^2/2)
        # We'll invert approximately:
        s = sigma
        mu = math.log(max(1e-9, mean)) - 0.5 * s * s
        return rng.lognormvariate(mu, s)

    def uniform(rng: random.Random) -> float:
        a = params.get("low", 0.0)
        b = params.get("high", 1.0)
        return rng.uniform(a, b)

    if kind == "exponential":
        return expo
    if kind == "normal":
        return normal
    if kind == "lognormal":
        return lognormal
    if kind == "uniform":
        return uniform

    raise ValueError(f"Unsupported distribution type: {kind}")


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class StageConfig:
    name: str
    capacity: int
    service_time: Dict[str, Any]
    capacity_schedule: Optional[List[int]] = None  # Optional 24-length hourly capacities


@dataclass
class SimConfig:
    seed: int
    sim_hours: float
    warmup_hours: float
    stages: List[StageConfig]
    arrivals_per_hour: float
    arrivals_profile: Optional[List[float]] = None  # 24-length multipliers (optional)
    hourly_arrivals: Optional[List[float]] = None   # 24-length absolute rates (optional, overrides profile)


@dataclass
class PatientRecord:
    id: int
    arrival_time: float
    stage_waits: Dict[str, float] = field(default_factory=dict)
    stage_service_times: Dict[str, float] = field(default_factory=dict)
    departure_time: Optional[float] = None

    @property
    def los(self) -> Optional[float]:
        if self.departure_time is None:
            return None
        return self.departure_time - self.arrival_time


# -----------------------------
# Hospital Simulation
# -----------------------------

class HospitalSim:
    def __init__(self, config: SimConfig):
        self.config = config
        self.env = simpy.Environment()
        self.rng = random.Random(config.seed)

        # Resources by stage
        self.stages: List[StageConfig] = config.stages
        # Compute max capacity per stage considering schedules (if any)
        self.max_caps: Dict[str, int] = {}
        for s in self.stages:
            if s.capacity_schedule is not None:
                if len(s.capacity_schedule) != 24:
                    raise ValueError(f"capacity_schedule for stage {s.name} must have length 24")
                max_c = max(int(s.capacity), int(max(s.capacity_schedule)))
            else:
                max_c = int(s.capacity)
            self.max_caps[s.name] = max(1, max_c)

        self.resources: Dict[str, simpy.Resource] = {
            s.name: simpy.Resource(self.env, capacity=self.max_caps[s.name]) for s in self.stages
        }
        self.service_dists: Dict[str, Distribution] = {
            s.name: make_distribution(s.service_time) for s in self.stages
        }

        # Data capture
        self.patients: List[PatientRecord] = []
        self._patient_seq = 0
        # Blockers to emulate reduced capacity
        self._blockers: Dict[str, List[simpy.events.Process]] = {s.name: [] for s in self.stages}
        self._blocker_reqs: Dict[str, List[simpy.events.Event]] = {s.name: [] for s in self.stages}

        # Start capacity managers if schedules present
        for s in self.stages:
            if s.capacity_schedule is not None:
                self.env.process(self.capacity_manager(s))

    # ---------------
    # Arrival process
    # ---------------
    def rate_at(self, t: float) -> float:
        """Return arrival rate (per hour) at simulation time t (hours), with 24h periodicity."""
        # Determine hour of day 0..23
        hour = int(math.floor(t % 24))
        if self.config.hourly_arrivals is not None:
            rates = self.config.hourly_arrivals
            if len(rates) != 24:
                raise ValueError("hourly_arrivals must have length 24")
            return max(0.0, float(rates[hour]))
        base = max(0.0, float(self.config.arrivals_per_hour))
        if self.config.arrivals_profile is not None:
            prof = self.config.arrivals_profile
            if len(prof) != 24:
                raise ValueError("arrivals_profile must have length 24")
            return base * max(0.0, float(prof[hour]))
        return base

    def lambda_max(self) -> float:
        """Upper bound of rate over the 24h cycle (for thinning)."""
        if self.config.hourly_arrivals is not None:
            return max(0.0, max(self.config.hourly_arrivals))
        base = max(0.0, float(self.config.arrivals_per_hour))
        if self.config.arrivals_profile is not None:
            return base * max(0.0, max(self.config.arrivals_profile))
        return base

    def arrival_generator(self):
        """Non-homogeneous Poisson process via thinning with 24h periodic rate."""
        lam_max = self.lambda_max()
        if lam_max <= 0:
            # No arrivals at all; just wait until the end of simulation.
            yield self.env.timeout(self.config.sim_hours)
            return
        while True:
            # Propose candidate arrival with homogeneous rate lam_max
            delta = self.rng.expovariate(lam_max)
            yield self.env.timeout(delta)
            t = self.env.now
            lam_t = self.rate_at(t)
            accept_prob = 0.0 if lam_max <= 0 else min(1.0, lam_t / lam_max)
            if self.rng.random() <= accept_prob:
                self._patient_seq += 1
                pid = self._patient_seq
                pr = PatientRecord(id=pid, arrival_time=t)
                self.patients.append(pr)
                self.env.process(self.patient_flow(pr))

    # ----------------------
    # Capacity schedule logic
    # ----------------------
    def desired_capacity(self, stage: StageConfig, t: float) -> int:
        if stage.capacity_schedule is None:
            return int(stage.capacity)
        hour = int(math.floor(t % 24))
        return int(stage.capacity_schedule[hour])

    def capacity_manager(self, stage: StageConfig):
        name = stage.name
        res = self.resources[name]
        max_cap = self.max_caps[name]

        def make_blocker():
            req = res.request()
            self._blocker_reqs[name].append(req)
            yield req
            try:
                # Hold server indefinitely until interrupted
                yield self.env.timeout(float('inf'))
            except simpy.Interrupt:
                pass
            finally:
                # release and remove
                res.release(req)

        def adjust():
            desired = self.desired_capacity(stage, self.env.now)
            desired = max(0, min(desired, max_cap))
            # number of blockers needed to reduce effective capacity
            need_blockers = max(0, max_cap - desired)
            cur_blockers = len(self._blockers[name])
            if cur_blockers < need_blockers:
                # start additional blockers
                for _ in range(need_blockers - cur_blockers):
                    p = self.env.process(make_blocker())
                    self._blockers[name].append(p)
            elif cur_blockers > need_blockers:
                # interrupt extra blockers
                n_remove = cur_blockers - need_blockers
                for _ in range(n_remove):
                    p = self._blockers[name].pop()
                    p.interrupt()

        # Initial adjust
        adjust()
        while True:
            # wait until next hour boundary
            next_hour = math.floor(self.env.now) + 1
            if next_hour <= self.env.now:
                next_hour = self.env.now + 1
            yield self.env.timeout(next_hour - self.env.now)
            adjust()

    # -------------------
    # Patient flow process
    # -------------------
    def patient_flow(self, record: PatientRecord):
        # sequentially pass through each configured stage
        for stage in self.stages:
            start_wait = self.env.now
            with self.resources[stage.name].request() as req:
                yield req  # wait for resource
                wait = self.env.now - start_wait
                record.stage_waits[stage.name] = wait

                # service
                service_time = self.service_dists[stage.name](self.rng)
                record.stage_service_times[stage.name] = service_time
                yield self.env.timeout(service_time)
        record.departure_time = self.env.now

    # ----
    # Run
    # ----
    def run(self) -> pd.DataFrame:
        # Start arrivals
        self.env.process(self.arrival_generator())
        sim_end = self.config.sim_hours
        self.env.run(until=sim_end)

        # Build dataframe
        rows = []
        for p in self.patients:
            rows.append(
                {
                    "id": p.id,
                    "arrival_time": p.arrival_time,
                    **{f"wait_{k}": v for k, v in p.stage_waits.items()},
                    **{f"service_{k}": v for k, v in p.stage_service_times.items()},
                    "departure_time": p.departure_time,
                    "los": p.los,
                }
            )
        df = pd.DataFrame(rows)
        return df

    # ---------
    # KPIs
    # ---------
    def kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        warmup = self.config.warmup_hours
        # Filter out patients who arrived before warmup cutoff
        if warmup > 0:
            df_kpi = df[df["arrival_time"] >= warmup].copy()
        else:
            df_kpi = df.copy()

        kpis: Dict[str, Any] = {}
        # LOS
        if not df_kpi.empty:
            kpis["throughput"] = int(df_kpi["departure_time"].notna().sum())
            kpis["los_mean"] = float(df_kpi["los"].mean())
            kpis["los_p95"] = float(df_kpi["los"].quantile(0.95))
        else:
            kpis["throughput"] = 0
            kpis["los_mean"] = float("nan")
            kpis["los_p95"] = float("nan")

        # Waits per stage
        for s in self.stages:
            col = f"wait_{s.name}"
            if col in df_kpi:
                kpis[f"{col}_mean"] = float(df_kpi[col].mean())
                kpis[f"{col}_p95"] = float(df_kpi[col].quantile(0.95))
        return kpis


# -----------------------------
# Config helpers
# -----------------------------

def load_config(path: str) -> SimConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    stages = [
        StageConfig(
            name=s["name"],
            capacity=int(s.get("capacity", 1)),
            service_time=s["service_time"],
            capacity_schedule=s.get("capacity_schedule"),
        )
        for s in cfg["stages"]
    ]

    sc = SimConfig(
        seed=int(cfg.get("seed", 42)),
        sim_hours=float(cfg.get("sim_hours", 24.0)),
        warmup_hours=float(cfg.get("warmup_hours", 0.0)),
        arrivals_per_hour=float(cfg.get("arrivals_per_hour", 5.0)),
        arrivals_profile=cfg.get("arrivals_profile"),
        hourly_arrivals=cfg.get("hourly_arrivals"),
        stages=stages,
    )
    return sc
