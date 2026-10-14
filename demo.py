# -*- coding: utf-8 -*-
"""
Demo: AI Agent vs. No Cooling vs. Fixed Cooling
================================================
Runs three scenarios for 100 time steps and plots the results.

Scenarios:
  1. No cooling       – flow_rate = FLOW_MIN the entire run
  2. Fixed cooling    – flow_rate = 5.0 L/min constant
  3. AI agent         – LangGraph agent adapts flow each step

Output: plots/demo_comparison.png
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from simulator import BatterySimulator, BatteryConfig, heat_load_schedule
from train_model import load_model, predict_next_temp

N_STEPS  = 100
T_INIT   = 25.0
T_MIN    = 15.0
T_MAX    = 45.0
NANO_PCT = 1.0

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# Baseline simulations (no LangGraph overhead)
# ──────────────────────────────────────────────────────────────────

def run_baseline(flow_rate: float, n_steps: int = N_STEPS, seed: int = 1) -> pd.DataFrame:
    cfg = BatteryConfig()
    sim = BatterySimulator(T_init=T_INIT, config=cfg, seed=seed)
    for t in range(1, n_steps + 1):
        hl = heat_load_schedule(float(t))
        sim.step(hl, flow_rate, NANO_PCT)
    return sim.get_dataframe()


# ──────────────────────────────────────────────────────────────────
# Agent simulation with adaptive control
# ──────────────────────────────────────────────────────────────────

def run_agent_direct(model, n_steps: int = N_STEPS, seed: int = 1) -> pd.DataFrame:
    """
    Runs the agent logic directly (without full LangGraph streaming)
    for speed and reliable step collection.
    """
    FLOW_MIN = 0.5
    FLOW_MAX = 10.0

    cfg = BatteryConfig()
    sim = BatterySimulator(T_init=T_INIT, config=cfg, seed=seed)
    flow = 3.0
    memory = []

    for t in range(1, n_steps + 1):
        hl = heat_load_schedule(float(t))

        # ── Read ─────────────────────────────────────────────────
        rec = sim.step(hl, flow, NANO_PCT)
        T   = rec["temperature"]

        # ── Predict ──────────────────────────────────────────────
        T_pred = predict_next_temp(model, T, hl, flow, NANO_PCT)

        # ── Decide (proportional + predictive) ───────────────────
        headroom = T_MAX - T_pred
        urgency  = 1.0 - headroom / (T_MAX - T_MIN)
        urgency  = max(0.0, min(1.0, urgency))
        flow_new = FLOW_MIN + urgency * (FLOW_MAX - FLOW_MIN)

        # ── Safety check ─────────────────────────────────────────
        flow_new = max(FLOW_MIN, min(FLOW_MAX, flow_new))
        T_pred2  = predict_next_temp(model, T, hl, flow_new, NANO_PCT)
        if T_pred2 > T_MAX:
            flow_new = FLOW_MAX          # emergency max cooling

        flow = round(flow_new, 2)
        memory.append({"time": t, "temperature": T, "flow": flow})

    df = sim.get_dataframe()
    df["flow_agent"] = [m["flow"] for m in memory]
    return df


# ──────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────

def plot_comparison(df_none, df_fixed, df_agent):
    fig = plt.figure(figsize=(16, 10), facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    gs  = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
                   left=0.08, right=0.96, top=0.90, bottom=0.08)

    PANEL_BG  = "#1a1d27"
    GRID_CLR  = "#2a2d3a"
    TEXT_CLR  = "#e8eaf6"
    ACCENT    = "#7c83fd"

    C_NONE    = "#ff5252"     # red  – no cooling
    C_FIXED   = "#ffb300"     # amber – fixed
    C_AGENT   = "#69f0ae"     # green – agent

    safe_alpha = 0.12

    times = df_none["time"].values

    # ── Panel A : Temperature comparison ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(PANEL_BG)
    ax1.fill_between(times, T_MIN, T_MAX, color=ACCENT, alpha=safe_alpha,
                     label="Safe zone (15–45°C)")
    ax1.axhline(T_MAX, color=ACCENT, ls="--", lw=0.8, alpha=0.5)
    ax1.axhline(T_MIN, color=ACCENT, ls="--", lw=0.8, alpha=0.5)

    ax1.plot(times, df_none["temperature"],  color=C_NONE,  lw=2,   label="No cooling")
    ax1.plot(times, df_fixed["temperature"], color=C_FIXED, lw=2,   label="Fixed flow (5 L/min)")
    ax1.plot(times, df_agent["temperature"], color=C_AGENT, lw=2.5, label="AI Agent (adaptive)")

    ax1.set_title("Battery Temperature Over Time — Scenario Comparison",
                  color=TEXT_CLR, fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("Temperature (°C)", color=TEXT_CLR)
    ax1.set_xlabel("Simulation Time (s)", color=TEXT_CLR)
    ax1.tick_params(colors=TEXT_CLR)
    ax1.grid(color=GRID_CLR, lw=0.6)
    for sp in ax1.spines.values(): sp.set_color(GRID_CLR)
    leg = ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
                     loc="upper left", fontsize=9)

    # ── Panel B : Heat load ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(PANEL_BG)
    heat_vals = [heat_load_schedule(t) for t in times]
    ax2.fill_between(times, heat_vals, alpha=0.3, color="#ff7043")
    ax2.plot(times, heat_vals, color="#ff7043", lw=2)
    ax2.set_title("Heat Load Profile (W)", color=TEXT_CLR, fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (s)", color=TEXT_CLR)
    ax2.set_ylabel("Heat Load (W)", color=TEXT_CLR)
    ax2.tick_params(colors=TEXT_CLR)
    ax2.grid(color=GRID_CLR, lw=0.6)
    for sp in ax2.spines.values(): sp.set_color(GRID_CLR)

    # ── Panel C : Agent flow rate ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PANEL_BG)
    ax3.plot(times, df_agent["flow_agent"], color=C_AGENT, lw=2, label="Agent flow")
    ax3.axhline(5.0, color=C_FIXED, lw=1.5, ls="--", alpha=0.7, label="Fixed (5 L/min)")
    ax3.axhline(0.5, color=C_NONE,  lw=1.5, ls="--", alpha=0.7, label="Min (0.5 L/min)")
    ax3.fill_between(times, df_agent["flow_agent"], 0, alpha=0.15, color=C_AGENT)
    ax3.set_title("AI Agent — Coolant Flow Rate", color=TEXT_CLR, fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (s)", color=TEXT_CLR)
    ax3.set_ylabel("Flow Rate (L/min)", color=TEXT_CLR)
    ax3.set_ylim(0, 11)
    ax3.tick_params(colors=TEXT_CLR)
    ax3.grid(color=GRID_CLR, lw=0.6)
    for sp in ax3.spines.values(): sp.set_color(GRID_CLR)
    ax3.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8)

    # ── Panel D : Violations summary bar chart ────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(PANEL_BG)
    scenarios    = ["No Cooling", "Fixed (5 L/min)", "AI Agent"]
    violations   = [
        int((df_none["temperature"]  > T_MAX).sum()),
        int((df_fixed["temperature"] > T_MAX).sum()),
        int((df_agent["temperature"] > T_MAX).sum()),
    ]
    colors_bar = [C_NONE, C_FIXED, C_AGENT]
    bars = ax4.bar(scenarios, violations, color=colors_bar, edgecolor=GRID_CLR, width=0.5)
    for bar, v in zip(bars, violations):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(v), ha="center", va="bottom", color=TEXT_CLR, fontsize=12, fontweight="bold")
    ax4.set_title("Over-temperature Violations", color=TEXT_CLR, fontsize=11, fontweight="bold")
    ax4.set_ylabel("Steps above 45°C", color=TEXT_CLR)
    ax4.tick_params(colors=TEXT_CLR)
    ax4.grid(axis="y", color=GRID_CLR, lw=0.6)
    for sp in ax4.spines.values(): sp.set_color(GRID_CLR)

    # ── Panel E : Statistics table ────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(PANEL_BG)
    ax5.axis("off")

    stats = []
    for label, df in [("No Cooling", df_none), ("Fixed (5 L/min)", df_fixed), ("AI Agent", df_agent)]:
        T = df["temperature"]
        stats.append([
            label,
            f"{T.mean():.1f}°C",
            f"{T.max():.1f}°C",
            f"{T.min():.1f}°C",
            f"{(T > T_MAX).sum()} / {N_STEPS}",
        ])

    table = ax5.table(
        cellText=stats,
        colLabels=["Scenario", "Mean T", "Max T", "Min T", "Violations"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor(PANEL_BG if r > 0 else "#22253a")
        cell.set_edgecolor(GRID_CLR)
        cell.set_text_props(color=TEXT_CLR)

    ax5.set_title("Performance Summary", color=TEXT_CLR, fontsize=11,
                  fontweight="bold", pad=12)

    # ── Super-title ───────────────────────────────────────────────
    fig.suptitle("🔋 AI Battery Cooling Agent — Demo Results",
                 color=TEXT_CLR, fontsize=16, fontweight="bold", y=0.97)

    out_path = os.path.join(PLOTS_DIR, "demo_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[demo] Plot saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AI BATTERY COOLING AGENT — DEMO")
    print("=" * 60)

    # Step 1 – ensure model is trained
    print("\n[1/4] Loading (or training) prediction model …")
    model, model_name = load_model()
    print(f"      Model ready: {model_name}")

    # Step 2 – baselines
    print("\n[2/4] Running no-cooling baseline …")
    df_none  = run_baseline(flow_rate=0.5)

    print("[2/4] Running fixed-flow baseline …")
    df_fixed = run_baseline(flow_rate=5.0)

    # Step 3 – agent
    print("\n[3/4] Running AI agent …")
    df_agent = run_agent_direct(model)

    # Step 4 – results
    print("\n[4/4] Generating comparison plot …")
    out = plot_comparison(df_none, df_fixed, df_agent)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for label, df in [("No cooling  ", df_none), ("Fixed 5 L/m ", df_fixed), ("AI Agent    ", df_agent)]:
        T     = df["temperature"]
        viols = int((T > T_MAX).sum())
        print(f"  {label}  mean={T.mean():.1f}°C  max={T.max():.1f}°C  violations={viols}/{N_STEPS}")

    print(f"\n✅ All done!  Plot → {out}")
    return out


if __name__ == "__main__":
    main()
