# -*- coding: utf-8 -*-
"""
Battery Thermal Simulator
=========================
Simulates battery temperature dynamics during a charge/discharge cycle.
Physics model:
  dT/dt = (Q_heat - Q_cool) / C_thermal
  Q_heat  = heat_load (W)
  Q_cool  = flow_rate * c_fluid * rho * (T - T_ambient)
  noise   = Gaussian perturbation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BatteryConfig:
    # Thermal capacitance of the battery pack (J/°C)
    C_thermal: float = 5000.0
    # Fluid heat capacity (J/kg·°C) – water + nanoparticle blend
    c_fluid: float = 4200.0
    # Fluid density (kg/m³)
    rho_fluid: float = 1000.0
    # Ambient / inlet coolant temperature (°C)
    T_ambient: float = 25.0
    # Safe operating band
    T_min: float = 15.0
    T_max: float = 45.0
    # Time step (seconds)
    dt: float = 1.0
    # Gaussian noise std (°C)
    noise_std: float = 0.3


class BatterySimulator:
    """
    Discrete-time thermal simulator for a battery pack.

    State
    -----
    temperature : float  – current pack temperature (°C)

    Inputs
    ------
    heat_load  : float  – electrical heat dissipation (W)
    flow_rate  : float  – coolant volumetric flow (L/min → converted internally)
    nano_pct   : float  – nanoparticle % (boosts effective heat capacity)
    """

    def __init__(self, T_init: float = 25.0, config: Optional[BatteryConfig] = None, seed: int = 42):
        self.cfg = config or BatteryConfig()
        self.temperature = T_init
        self.time = 0.0
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _effective_heat_capacity(self, nano_pct: float) -> float:
        """Nanoparticles increase specific heat capacity linearly (up to ~15%)."""
        boost = 1.0 + 0.015 * nano_pct          # e.g. 1% nano → +1.5%
        return self.cfg.C_thermal * boost

    def _cooling_power(self, flow_rate_lpm: float, nano_pct: float) -> float:
        """Heat removed by coolant (W).  flow_rate in L/min → m³/s."""
        flow_m3s = (flow_rate_lpm / 1000.0) / 60.0     # L/min → m³/s
        delta_T = max(self.temperature - self.cfg.T_ambient, 0.0)
        # effective fluid capacity with nanoparticles
        c_eff = self.cfg.c_fluid * (1.0 + 0.012 * nano_pct)
        return c_eff * self.cfg.rho_fluid * flow_m3s * delta_T

    def step(self, heat_load: float, flow_rate: float, nano_pct: float = 1.0) -> dict:
        """Advance one time step and return the new state."""
        C = self._effective_heat_capacity(nano_pct)
        Q_cool = self._cooling_power(flow_rate, nano_pct)

        dT = (heat_load - Q_cool) / C * self.cfg.dt
        noise = self.rng.normal(0.0, self.cfg.noise_std)
        self.temperature = self.temperature + dT + noise
        self.time += self.cfg.dt

        record = {
            "time": self.time,
            "temperature": round(self.temperature, 3),
            "heat_load": heat_load,
            "flow_rate": flow_rate,
            "nano_pct": nano_pct,
            "Q_cool": round(Q_cool, 2),
        }
        self.history.append(record)
        return record

    def reset(self, T_init: float = 25.0):
        self.temperature = T_init
        self.time = 0.0
        self.history.clear()

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ------------------------------------------------------------------
# Heat-load schedule: charge ramp → hold → discharge → rest
# ------------------------------------------------------------------
def heat_load_schedule(t: float) -> float:
    """Returns heat load (W) as a function of simulation time (s)."""
    if t < 30:
        return 200.0 + 4.0 * t          # ramp up (charging)
    elif t < 55:
        return 320.0                     # peak hold
    elif t < 75:
        return 320.0 - 8.0 * (t - 55)   # discharge taper
    else:
        return 150.0                     # rest / idle


# ------------------------------------------------------------------
# Generate synthetic training dataset
# ------------------------------------------------------------------
def generate_training_data(n_samples: int = 3000, seed: int = 0) -> pd.DataFrame:
    """
    Produce a dataset of (heat_load, flow_rate, nano_pct, T_current) → T_next
    by running the simulator with random inputs.
    """
    rng = np.random.default_rng(seed)
    cfg = BatteryConfig(noise_std=0.2)
    sim = BatterySimulator(T_init=25.0, config=cfg, seed=seed)

    rows = []
    T = 25.0
    for _ in range(n_samples):
        hl  = rng.uniform(50, 500)
        fr  = rng.uniform(0.5, 10.0)
        np_ = rng.uniform(0.0, 3.0)
        sim.temperature = T
        rec = sim.step(hl, fr, np_)
        rows.append({
            "heat_load":  hl,
            "flow_rate":  fr,
            "nano_pct":   np_,
            "T_current":  T,
            "T_next":     rec["temperature"],
        })
        T = rec["temperature"]
        # Random reset to cover full temperature range
        if rng.random() < 0.05:
            T = rng.uniform(15, 50)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick sanity check
    sim = BatterySimulator(T_init=25.0, seed=7)
    for t in range(1, 21):
        hl = heat_load_schedule(t)
        rec = sim.step(hl, flow_rate=3.0, nano_pct=1.0)
        print(f"t={t:3d}s  T={rec['temperature']:5.2f}°C  HL={rec['heat_load']:5.0f}W  Qcool={rec['Q_cool']:.1f}W")
