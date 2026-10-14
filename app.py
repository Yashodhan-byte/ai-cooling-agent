"""
Streamlit Dashboard — AI Battery Cooling Agent  ✦ Ultra Edition
=================================================================
Run:  python3 -m streamlit run app.py
"""

import os
import sys
import time
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import streamlit as st

# ──────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔋 AI Battery Cooling Agent — Mission Control",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

from simulator import BatterySimulator, BatteryConfig, heat_load_schedule
from train_model import load_model, predict_next_temp

# ── Design tokens ─────────────────────────────────────────────────
T_MIN    = 15.0
T_MAX    = 45.0
FLOW_MIN = 0.5
FLOW_MAX = 10.0

BG       = "#090d16"
SURFACE  = "#0e1424"
CARD     = "#131a2f"
CARD2    = "#18223c"
BORDER   = "#232e4d"
ACCENT   = "#6366f1"   # Indigo
ACCENT2  = "#8b5cf6"   # Violet
CYAN     = "#06b6d4"   # Coolant Cyan
GREEN    = "#10b981"   # Safe Emerald
AMBER    = "#f59e0b"   # Warning Amber
RED      = "#ef4444"   # Critical Rose
TEXT     = "#f1f5f9"
MUTED    = "#94a3b8"

C_NONE   = "#ef4444"
C_FIXED  = "#f59e0b"
C_AGENT  = "#10b981"
PANEL_BG = "#0e1424"
GRID_CLR = "#1e2942"
TEXT_CLR = "#f1f5f9"

# Pump power model constants (approx W at given flow in L/min)
# P_pump = alpha * (flow_rate)^2.2 (quadratic/cubic hydraulic curve)
PUMP_POWER_COEFF = 2.4   # Watts per (L/min)^1.8

def calc_pump_power(flow_lpm: float) -> float:
    return PUMP_POWER_COEFF * (flow_lpm ** 1.8)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: {BG};
    color: {TEXT};
}}
.stApp {{ background-color: {BG}; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {SURFACE}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0b1120 0%, #060913 100%);
    border-right: 1px solid {BORDER};
}}

/* Streamlit Button Styling */
.stButton > button {{
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.88rem;
    padding: 10px 18px;
    letter-spacing: 0.3px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 18px rgba(99,102,241,0.3);
    width: 100%;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.5);
    border-color: rgba(255,255,255,0.3);
}}

/* Card container */
.glass-panel {{
    background: linear-gradient(145deg, {CARD} 0%, rgba(14,20,36,0.85) 100%);
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    backdrop-filter: blur(12px);
}}

/* Metric Cards */
.metric-card {{
    background: linear-gradient(145deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 18px 16px 14px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s;
    position: relative;
    overflow: hidden;
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {CYAN}, {ACCENT}, {ACCENT2});
    opacity: 0.8;
}}
.metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(0,0,0,0.5);
    border-color: rgba(99,102,241,0.4);
}}
.metric-value {{
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.5px;
}}
.metric-label {{
    font-size: 0.68rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}}
.metric-sub {{
    font-size: 0.72rem;
    color: {MUTED};
    margin-top: 5px;
    font-family: 'JetBrains Mono', monospace;
}}
.metric-icon {{ font-size: 1.35rem; margin-bottom: 4px; display: block; }}

/* Progress Bar */
.prog-container {{
    background: {SURFACE};
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 10px 0 16px 0;
    border: 1px solid {BORDER};
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
}}
.prog-fill {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, {CYAN}, {ACCENT}, {GREEN});
    transition: width 0.35s ease;
    box-shadow: 0 0 12px rgba(16,185,129,0.5);
}}

/* Thermal & Flow Gauges */
.gauge-box {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}}
.gauge-bar-bg {{
    background: {SURFACE};
    border-radius: 8px;
    height: 14px;
    width: 100%;
    overflow: hidden;
    position: relative;
    border: 1px solid {BORDER};
}}
.gauge-bar-val {{
    height: 100%;
    border-radius: 8px;
    transition: width 0.4s ease;
}}

/* Memory Log Entries */
.memory-entry {{
    background: {CARD};
    border-left: 4px solid {ACCENT};
    padding: 8px 14px;
    margin: 5px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    color: {TEXT};
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}}
.memory-entry.danger {{
    border-left-color: {RED};
    background: rgba(239,68,68,0.08);
}}
.memory-entry.warn {{
    border-left-color: {AMBER};
}}
.memory-entry:hover {{ background: {CARD2}; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDER};
    border-radius: 12px 12px 0 0;
    gap: 6px;
    padding: 6px 8px 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {MUTED};
    font-weight: 700;
    font-size: 0.85rem;
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    border: none;
    transition: all 0.2s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT};
    background: rgba(255,255,255,0.04);
}}
.stTabs [aria-selected="true"] {{
    background: {CARD} !important;
    color: {CYAN} !important;
    border-bottom: 2px solid {CYAN} !important;
}}

/* Badges */
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}}
.badge-run {{
    background: rgba(16,185,129,0.15);
    color: {GREEN};
    border: 1px solid rgba(16,185,129,0.35);
    box-shadow: 0 0 10px rgba(16,185,129,0.2);
}}
.badge-idle {{
    background: rgba(148,163,184,0.12);
    color: {MUTED};
    border: 1px solid rgba(148,163,184,0.25);
}}
.badge-alert {{
    background: rgba(239,68,68,0.15);
    color: {RED};
    border: 1px solid rgba(239,68,68,0.35);
}}

/* Report Card */
.report-card {{
    background: linear-gradient(145deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 22px;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}}
.report-title {{
    font-size: 1.05rem;
    font-weight: 800;
    color: {TEXT};
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* Callouts & headers */
.section-header {{
    font-size: 0.76rem;
    font-weight: 800;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 1.8px;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 8px;
    margin: 18px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}

hr {{ border: none; border-top: 1px solid {BORDER}; margin: 18px 0; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────
def init_session():
    if "model" not in st.session_state:
        with st.spinner("🤖 Initializing Physics & Machine Learning Agent…"):
            model, name = load_model()
            st.session_state.model = model
            st.session_state.model_name = name

    defaults = dict(
        sim_agent=None, sim_none=None, sim_fixed=None,
        running=False, step=0, flow_agent=FLOW_MIN, memory=[],
        agent_temps=[], none_temps=[], fixed_temps=[],
        agent_flows=[], times=[], heat_loads=[],
        q_cool_agent=[], q_cool_fixed=[],
        pump_power_agent=[], pump_power_fixed=[],
        violations_a=0, violations_n=0, violations_f=0,
        n_steps=100, fixed_flow=5.0, T_init=25.0,
        nano_pct=1.0, profile="Dynamic Cycle",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_simulation(n_steps, fixed_flow, T_init, nano_pct=1.0):
    cfg = BatteryConfig()
    st.session_state.sim_agent = BatterySimulator(T_init=T_init, config=cfg, seed=99)
    st.session_state.sim_none  = BatterySimulator(T_init=T_init, config=cfg, seed=99)
    st.session_state.sim_fixed = BatterySimulator(T_init=T_init, config=cfg, seed=99)
    st.session_state.step               = 0
    st.session_state.flow_agent         = 3.0
    st.session_state.memory             = []
    st.session_state.agent_temps        = []
    st.session_state.none_temps         = []
    st.session_state.fixed_temps        = []
    st.session_state.agent_flows        = []
    st.session_state.times              = []
    st.session_state.heat_loads         = []
    st.session_state.q_cool_agent       = []
    st.session_state.q_cool_fixed       = []
    st.session_state.pump_power_agent   = []
    st.session_state.pump_power_fixed   = []
    st.session_state.violations_a       = 0
    st.session_state.violations_n       = 0
    st.session_state.violations_f       = 0
    st.session_state.n_steps            = n_steps
    st.session_state.fixed_flow         = fixed_flow
    st.session_state.T_init             = T_init
    st.session_state.nano_pct           = nano_pct


# ── Custom Workload Profiles ─────────────────────────────────────
def get_heat_load(t: float, profile: str) -> float:
    if profile == "Dynamic Cycle":
        return heat_load_schedule(t)
    elif profile == "Extreme Fast-Charge (Peak 450W)":
        if t < 40:
            return 250.0 + 5.0 * t  # ramps up to 450W
        elif t < 70:
            return 450.0
        else:
            return max(150.0, 450.0 - 10.0 * (t - 70))
    elif profile == "Constant Highway Cruise (280W)":
        return 280.0 + 15.0 * math.sin(t / 10.0)
    elif profile == "Urban Stop & Go (Pulsed)":
        pulse = 380.0 if (int(t) // 10) % 2 == 1 else 140.0
        return pulse
    return heat_load_schedule(t)


# ── Agent logic ───────────────────────────────────────────────────
def agent_decide(T_current, heat_load, current_flow, model, nano_pct):
    T_pred   = predict_next_temp(model, T_current, heat_load, current_flow, nano_pct)
    headroom = T_MAX - T_pred
    urgency  = 1.0 - headroom / (T_MAX - T_MIN)
    urgency  = max(0.0, min(1.0, urgency))
    flow_new = FLOW_MIN + urgency * (FLOW_MAX - FLOW_MIN)
    T_pred2  = predict_next_temp(model, T_current, heat_load, flow_new, nano_pct)
    if T_pred2 > T_MAX:
        flow_new = FLOW_MAX
    return round(max(FLOW_MIN, min(FLOW_MAX, flow_new)), 2), T_pred


def advance_one_step(profile="Dynamic Cycle"):
    s  = st.session_state
    t  = s.step + 1
    hl = get_heat_load(float(t), profile)
    nano = s.nano_pct

    # Agent Step
    rec_a = s.sim_agent.step(hl, s.flow_agent, nano)
    T_a   = rec_a["temperature"]
    flow_new, pred_t = agent_decide(T_a, hl, s.flow_agent, s.model, nano)
    s.flow_agent = flow_new

    # Baselines
    rec_n = s.sim_none.step(hl, FLOW_MIN, nano)
    rec_f = s.sim_fixed.step(hl, s.fixed_flow, nano)

    p_agent = calc_pump_power(flow_new)
    p_fixed = calc_pump_power(s.fixed_flow)

    s.times.append(t)
    s.heat_loads.append(hl)
    s.agent_temps.append(T_a)
    s.none_temps.append(rec_n["temperature"])
    s.fixed_temps.append(rec_f["temperature"])
    s.agent_flows.append(flow_new)
    s.q_cool_agent.append(rec_a["Q_cool"])
    s.q_cool_fixed.append(rec_f["Q_cool"])
    s.pump_power_agent.append(p_agent)
    s.pump_power_fixed.append(p_fixed)

    s.memory.append({
        "t": t, "T": T_a, "flow": flow_new, "hl": hl,
        "q_cool": rec_a["Q_cool"], "pred_T": pred_t, "power": p_agent
    })
    if len(s.memory) > 30:
        s.memory = s.memory[-30:]

    if T_a > T_MAX:                      s.violations_a += 1
    if rec_n["temperature"] > T_MAX:     s.violations_n += 1
    if rec_f["temperature"] > T_MAX:     s.violations_f += 1
    s.step += 1


# ── Chart styling helper ──────────────────────────────────────────
def _style_ax(ax, fig=None):
    ax.set_facecolor(PANEL_BG)
    if fig:
        fig.patch.set_facecolor(BG)
    ax.grid(color=GRID_CLR, lw=0.7, linestyle="--", alpha=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)


def make_temp_chart():
    s = st.session_state
    if not s.times:
        return None
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=BG)
    _style_ax(ax, fig)
    ax.fill_between(s.times, T_MIN, T_MAX, color=ACCENT, alpha=0.06, label="Safe Band (15–45°C)")
    ax.axhline(T_MAX, color=RED,  ls="--", lw=1.2, alpha=0.8)
    ax.axhline(T_MIN, color=CYAN, ls="--", lw=1.2, alpha=0.8)
    ax.text(s.times[-1], T_MAX + 0.4, f" {T_MAX}°C Upper Limit", color=RED,  fontsize=7, fontweight="bold")
    ax.text(s.times[-1], T_MIN - 0.4, f" {T_MIN}°C Lower Limit", color=CYAN, fontsize=7, va="top", fontweight="bold")

    ax.plot(s.times, s.none_temps,  color=C_NONE,  lw=1.6, alpha=0.85, label="No Active Cooling (0.5 L/min)")
    ax.plot(s.times, s.fixed_temps, color=C_FIXED, lw=1.6, alpha=0.85,
            label=f"Fixed Flow ({s.fixed_flow} L/min)")
    ax.fill_between(s.times, s.agent_temps, alpha=0.15, color=C_AGENT)
    ax.plot(s.times, s.agent_temps, color=C_AGENT, lw=2.6, label="🤖 AI Agent (Adaptive)", zorder=5)

    ax.set_xlabel("Elapsed Time (seconds)", color=MUTED, fontsize=8)
    ax.set_ylabel("Pack Temperature (°C)", color=MUTED, fontsize=8)
    ax.set_title("Battery Temperature — Live Multi-Strategy Comparison", color=TEXT_CLR, fontsize=11,
                 fontweight="bold", pad=12)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT_CLR, fontsize=8,
              loc="upper left", framealpha=0.9)
    fig.tight_layout(pad=1.4)
    return fig


def make_flow_chart():
    s = st.session_state
    if not s.times:
        return None
    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor=BG)
    _style_ax(ax, fig)
    ax.fill_between(s.times, s.agent_flows, FLOW_MIN, alpha=0.2, color=CYAN)
    ax.plot(s.times, s.agent_flows, color=CYAN, lw=2.2, label="AI Dynamic Flow (L/min)")
    ax.axhline(s.fixed_flow, color=C_FIXED, ls="--", lw=1.2, alpha=0.8,
               label=f"Fixed Baseline ({s.fixed_flow} L/min)")
    ax.set_xlabel("Elapsed Time (seconds)", color=MUTED, fontsize=8)
    ax.set_ylabel("Coolant Flow (L/min)", color=MUTED, fontsize=8)
    ax.set_title("Coolant Flow Modulation Profile", color=TEXT_CLR, fontsize=11,
                 fontweight="bold", pad=12)
    ax.set_ylim(-0.2, FLOW_MAX + 1.2)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT_CLR, fontsize=8, framealpha=0.9)
    fig.tight_layout(pad=1.4)
    return fig


def make_energy_chart():
    s = st.session_state
    if not s.times:
        return None
    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor=BG)
    _style_ax(ax, fig)
    cum_agent_joules = np.cumsum(s.pump_power_agent)
    cum_fixed_joules = np.cumsum(s.pump_power_fixed)
    ax.plot(s.times, cum_agent_joules, color=GREEN, lw=2.2, label="AI Agent Cumulative Pump Energy (J)")
    ax.plot(s.times, cum_fixed_joules, color=AMBER, lw=2.0, ls="--", label="Fixed Flow Cumulative Pump Energy (J)")
    ax.fill_between(s.times, cum_agent_joules, cum_fixed_joules, color=GREEN, alpha=0.08)
    ax.set_xlabel("Elapsed Time (seconds)", color=MUTED, fontsize=8)
    ax.set_ylabel("Parasitic Energy (Joules)", color=MUTED, fontsize=8)
    ax.set_title("Parasitic Pump Energy Consumption", color=TEXT_CLR, fontsize=11, fontweight="bold", pad=12)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT_CLR, fontsize=8, framealpha=0.9)
    fig.tight_layout(pad=1.4)
    return fig


def make_violation_bar():
    s = st.session_state
    fig, ax = plt.subplots(figsize=(5.5, 3.4), facecolor=BG)
    _style_ax(ax, fig)
    cats = ["No Cooling", f"Fixed ({s.fixed_flow} L/m)", "AI Agent"]
    vals = [s.violations_n, s.violations_f, s.violations_a]
    bars = ax.bar(cats, vals, color=[C_NONE, C_FIXED, C_AGENT], edgecolor=BORDER,
                  linewidth=1, width=0.48, zorder=3)
    ymax = max(vals) if max(vals) > 0 else 1
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.03,
                str(v), ha="center", color=TEXT_CLR, fontsize=12, fontweight="bold")
    ax.set_ylabel("Safety Violations (Steps > 45°C)", color=MUTED, fontsize=8)
    ax.set_title("Thermal Safety Violations", color=TEXT_CLR, fontsize=10,
                 fontweight="bold", pad=12)
    ax.set_ylim(0, ymax * 1.35 + 1)
    fig.tight_layout(pad=1.4)
    return fig


def make_scatter():
    s = st.session_state
    if len(s.times) < 5:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.4), facecolor=BG)
    _style_ax(ax, fig)
    sc = ax.scatter(s.agent_flows, s.agent_temps, c=s.heat_loads,
                    cmap="plasma", s=28, alpha=0.85, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
    cbar.set_label("Heat Load (W)", color=MUTED, fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)
    ax.axhline(T_MAX, color=RED, ls="--", lw=1.0, alpha=0.7, label=f"Critical {T_MAX}°C")
    ax.set_xlabel("Flow Rate (L/min)", color=MUTED, fontsize=8)
    ax.set_ylabel("Battery Temp (°C)", color=MUTED, fontsize=8)
    ax.set_title("Cooling Efficiency: Flow vs Temp", color=TEXT_CLR,
                 fontsize=10, fontweight="bold", pad=10)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT_CLR, fontsize=8)
    fig.tight_layout(pad=1.4)
    return fig


# ── UI Components ─────────────────────────────────────────────────
def metric_card(icon, value, label, color=ACCENT, sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <span class="metric-icon">{icon}</span>
        <div class="metric-value" style="color:{color};">{value}</div>
        <div class="metric-label">{label}</div>
        {sub_html}
    </div>
    """


def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:12px 0 16px;">
            <div style="font-size:2.2rem; filter:drop-shadow(0 0 10px rgba(99,102,241,0.5));">🔋</div>
            <div style="font-size:1.1rem; font-weight:800; color:{TEXT}; letter-spacing:0.5px;">
                AI Cooling Agent
            </div>
            <div style="font-size:0.72rem; color:{CYAN}; font-weight:600; margin-top:2px;">
                ✦ MISSION CONTROL v2.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        model_name = st.session_state.get("model_name", "GradientBoostingRegressor")
        st.markdown(f"""
        <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:12px;
                    padding:10px 14px; margin-bottom:14px; text-align:center;">
            <div style="font-size:0.65rem; color:{MUTED}; text-transform:uppercase;
                        letter-spacing:1.5px; margin-bottom:2px;">Prediction Engine</div>
            <div style="font-size:0.85rem; font-weight:700; color:{CYAN};">{model_name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="section-header">⚡ Workload Scenario</div>', unsafe_allow_html=True)
        profile = st.selectbox(
            "Heat Load Profile",
            [
                "Dynamic Cycle",
                "Extreme Fast-Charge (Peak 450W)",
                "Constant Highway Cruise (280W)",
                "Urban Stop & Go (Pulsed)"
            ],
            index=0,
            help="Choose battery thermal stress scenario"
        )

        nano_pct = st.slider(
            "Nanoparticle % (Coolant)", 0.0, 3.0, 1.0, 0.1,
            help="Nanoparticle enhancement ratio (boosts thermal capacity up to 15%)"
        )

        st.markdown(f'<div class="section-header">⚙️ Simulation Parameters</div>', unsafe_allow_html=True)
        n_steps    = st.slider("Total Time Steps (s)", 20, 250, 100, 10)
        T_init     = st.slider("Initial Temperature (°C)", 15.0, 42.0, 25.0, 0.5)
        fixed_flow = st.slider("Fixed Baseline Flow (L/min)", 0.5, 10.0, 5.0, 0.5)
        speed_ms   = st.slider("Simulation Throttle (ms)", 0, 300, 40, 10,
                               help="Delay between frames for smooth visualization")

        col1, col2 = st.columns(2)
        with col1:
            start    = st.button("▶ Run Agent", use_container_width=True, key="btn_start")
        with col2:
            stop_btn = st.button("↺ Reset All", use_container_width=True, key="btn_reset")

        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.06); border:1px solid {BORDER}; border-radius:12px;
                    padding:12px; margin-top:16px; font-size:0.75rem; color:{MUTED}; line-height:1.6;">
            <b style="color:{TEXT};">Physics Principles:</b><br>
            • <code style="color:{CYAN};">dT/dt = (Q_in - Q_out) / C_eff</code><br>
            • Nanofluid thermal boost active<br>
            • Parasitic pump power calculation
        </div>
        """, unsafe_allow_html=True)

    return n_steps, T_init, fixed_flow, speed_ms, start, stop_btn, nano_pct, profile


# ── Main Application ──────────────────────────────────────────────
def main():
    init_session()
    s = st.session_state

    n_steps, T_init, fixed_flow, speed_ms, start, stop_btn, nano_pct, profile = render_sidebar()

    if start:
        reset_simulation(n_steps, fixed_flow, T_init, nano_pct)
        st.session_state.profile = profile
        st.session_state.running = True

    if stop_btn:
        st.session_state.running = False
        reset_simulation(n_steps, fixed_flow, T_init, nano_pct)

    step_now    = s.step
    total_steps = s.get("n_steps", n_steps)
    T_now       = s.agent_temps[-1] if s.agent_temps else s.get("T_init", T_init)
    flow_now    = s.agent_flows[-1] if s.agent_flows else 3.0
    hl_now      = get_heat_load(float(step_now), s.get("profile", profile))
    pct         = (step_now / total_steps * 100) if total_steps > 0 else 0

    # Cumulative parasitic pump energy (Joules = Watt-seconds)
    energy_agent_kj = (sum(s.pump_power_agent) / 1000.0) if s.pump_power_agent else 0.0
    energy_fixed_kj = (sum(s.pump_power_fixed) / 1000.0) if s.pump_power_fixed else 0.0
    energy_saved_pct = (
        ((energy_fixed_kj - energy_agent_kj) / energy_fixed_kj * 100)
        if energy_fixed_kj > 0.01 else 0.0
    )

    if s.running:
        badge = f'<span class="badge badge-run">● Agent Active</span>'
    elif step_now > 0 and not s.running:
        badge = f'<span class="badge badge-idle">✓ Simulation Finished</span>'
    else:
        badge = f'<span class="badge badge-idle">Ready</span>'

    t_color = GREEN if T_MIN <= T_now <= T_MAX - 3 else (AMBER if T_now <= T_MAX else RED)
    v_color = GREEN if s.violations_a == 0 else RED

    # ── Top Hero Header ───────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:16px 0 10px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
                    <h1 style="font-size:2.2rem; font-weight:800; margin:0;
                               background: linear-gradient(135deg, #ffffff 0%, {CYAN} 50%, {ACCENT} 100%);
                               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                               letter-spacing: -0.5px;">
                        🔋 AI Thermal Mission Control
                    </h1>
                    {badge}
                </div>
                <div style="color:{MUTED}; font-size:0.85rem;">
                    Autonomous predictive battery thermal management using <b>LangGraph + Physics Surrogate AI</b>
                </div>
            </div>
            <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:8px 16px; font-size:0.8rem;">
                <span style="color:{MUTED};">Scenario:</span> <b style="color:{TEXT};">{s.get('profile', profile)}</b>
                &nbsp;·&nbsp;
                <span style="color:{MUTED};">Nanofluid:</span> <b style="color:{CYAN};">{s.get('nano_pct', nano_pct):.1f}%</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(metric_card("🌡️", f"{T_now:.1f}°C", "Battery Temp",
                                t_color, "Limit 45.0°C"), unsafe_allow_html=True)
    with k2:
        st.markdown(metric_card("💧", f"{flow_now:.2f}", "Flow Rate",
                                CYAN, "Liters / min"), unsafe_allow_html=True)
    with k3:
        st.markdown(metric_card("⚡", f"{hl_now:.0f}W", "Heat Load",
                                AMBER if hl_now > 300 else GREEN, "Current dissipation"),
                    unsafe_allow_html=True)
    with k4:
        st.markdown(metric_card("🛡️", str(s.violations_a), "AI Violations",
                                v_color, "Steps > 45°C"), unsafe_allow_html=True)
    with k5:
        st.markdown(metric_card("🔋", f"{energy_agent_kj:.1f} kJ", "Pump Energy",
                                GREEN if energy_saved_pct >= 0 else RED,
                                f"{energy_saved_pct:+.1f}% vs fixed"), unsafe_allow_html=True)
    with k6:
        st.markdown(metric_card("⏱️", f"{step_now}/{total_steps}", "Cycle Progress",
                                ACCENT, f"{pct:.0f}% finished"), unsafe_allow_html=True)

    if s.running or step_now > 0:
        st.markdown(f"""
        <div class="prog-container">
            <div class="prog-fill" style="width:{pct:.1f}%;"></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs Navigation ───────────────────────────────────────────
    tab_live, tab_telemetry, tab_analytics, tab_memory, tab_export = st.tabs([
        "📡 Live Dashboard",
        "⚡ Energy & Telemetry",
        "📊 Comparative Benchmark",
        "🧠 Agent Memory & Decisions",
        "💾 Data Export & Reports"
    ])

    # ── Tab 1: Live Dashboard ─────────────────────────────────────
    with tab_live:
        if s.agent_temps:
            col_chart, col_gauges = st.columns([3, 1])

            with col_chart:
                fig_t = make_temp_chart()
                if fig_t: st.pyplot(fig_t, use_container_width=True); plt.close(fig_t)

                fig_f = make_flow_chart()
                if fig_f: st.pyplot(fig_f, use_container_width=True); plt.close(fig_f)

            with col_gauges:
                st.markdown(f'<div class="section-header">🎛️ Real-Time Instruments</div>', unsafe_allow_html=True)

                # Thermal Gauge
                temp_pct = min(100.0, max(0.0, (T_now - 15.0) / (50.0 - 15.0) * 100))
                g_col = GREEN if T_now < 42.0 else (AMBER if T_now <= 45.0 else RED)

                st.markdown(f"""
                <div class="gauge-box">
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:6px;">
                        <span style="color:{MUTED}; font-weight:700;">THERMAL BAND</span>
                        <span style="color:{g_col}; font-weight:800;">{T_now:.1f}°C</span>
                    </div>
                    <div class="gauge-bar-bg">
                        <div class="gauge-bar-val" style="width:{temp_pct:.1f}%; background:{g_col};"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:{MUTED}; margin-top:4px;">
                        <span>15°C (Min)</span>
                        <span>45°C (Crit)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Flow Demand Gauge
                flow_pct = min(100.0, max(0.0, (flow_now - FLOW_MIN) / (FLOW_MAX - FLOW_MIN) * 100))
                st.markdown(f"""
                <div class="gauge-box">
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:6px;">
                        <span style="color:{MUTED}; font-weight:700;">PUMP MODULATION</span>
                        <span style="color:{CYAN}; font-weight:800;">{flow_now:.2f} L/m</span>
                    </div>
                    <div class="gauge-bar-bg">
                        <div class="gauge-bar-val" style="width:{flow_pct:.1f}%; background:{CYAN};"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:{MUTED}; margin-top:4px;">
                        <span>0.5 L/m</span>
                        <span>10.0 L/m</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Heat Extraction Power
                q_now = s.q_cool_agent[-1] if s.q_cool_agent else 0.0
                st.markdown(f"""
                <div class="gauge-box">
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:6px;">
                        <span style="color:{MUTED}; font-weight:700;">COOLING POWER</span>
                        <span style="color:{GREEN}; font-weight:800;">{q_now:.0f} W</span>
                    </div>
                    <div style="font-size:0.72rem; color:{MUTED}; margin-top:6px;">
                        Active removal rate via liquid cold-plate
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:18px;
                        padding:70px 20px; text-align:center; margin:20px 0;">
                <div style="font-size:3.5rem; margin-bottom:12px;">⚡</div>
                <div style="font-size:1.3rem; font-weight:800; color:{TEXT}; margin-bottom:6px;">
                    Simulator Awaiting Signal
                </div>
                <div style="font-size:0.85rem; color:{MUTED}; max-width:440px; margin:0 auto 20px auto;">
                    Select your thermal profile and parameters in the sidebar, then click
                    <b style="color:{CYAN};">▶ Run Agent</b> to initiate real-time telemetry.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2: Energy & Telemetry ─────────────────────────────────
    with tab_telemetry:
        if s.agent_temps:
            st.markdown(f'<div class="section-header">⚡ Parasitic Power & Cooling Efficiency</div>', unsafe_allow_html=True)
            col_en1, col_en2 = st.columns([2, 1])

            with col_en1:
                fig_en = make_energy_chart()
                if fig_en: st.pyplot(fig_en, use_container_width=True); plt.close(fig_en)

            with col_en2:
                st.markdown(f"""
                <div class="report-card">
                    <div class="report-title">💡 Parasitic Energy Savings</div>
                    <div style="font-size:2rem; font-weight:800; color:{GREEN}; font-family:'JetBrains Mono'; margin-bottom:6px;">
                        {energy_saved_pct:+.1f}%
                    </div>
                    <div style="font-size:0.8rem; color:{MUTED}; line-height:1.6;">
                        Fixed Flow ({s.fixed_flow} L/min): <b style="color:{TEXT};">{energy_fixed_kj:.2f} kJ</b><br>
                        AI Adaptive Flow: <b style="color:{GREEN};">{energy_agent_kj:.2f} kJ</b><br>
                        Net Saved: <b style="color:{CYAN};">{max(0.0, energy_fixed_kj - energy_agent_kj):.2f} kJ</b>
                    </div>
                    <div style="margin-top:14px; font-size:0.75rem; color:{MUTED}; border-top:1px solid {BORDER}; padding-top:10px;">
                        The AI agent throttles pump speeds during low-load intervals, eliminating wasteful over-cooling while preventing thermal spikes.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info("Run simulation to analyze energy and telemetry metrics.", icon="⚡")

    # ── Tab 3: Comparative Benchmark ──────────────────────────────
    with tab_analytics:
        if s.agent_temps:
            col_bar, col_sc = st.columns(2)
            with col_bar:
                fig_b = make_violation_bar()
                if fig_b: st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)
            with col_sc:
                fig_sc = make_scatter()
                if fig_sc: st.pyplot(fig_sc, use_container_width=True); plt.close(fig_sc)

            st.markdown(f'<div class="section-header">📋 Strategy Performance Breakdown</div>', unsafe_allow_html=True)

            avg_a = np.mean(s.agent_temps)
            avg_n = np.mean(s.none_temps)
            avg_f = np.mean(s.fixed_temps)
            max_a = np.max(s.agent_temps)
            max_n = np.max(s.none_temps)
            max_f = np.max(s.fixed_temps)
            avg_flow = np.mean(s.agent_flows)

            def _table(rows):
                html = "<table style='width:100%; font-size:0.83rem; border-collapse:collapse;'>"
                for label, val, clr in rows:
                    html += (f"<tr style='border-bottom:1px solid rgba(255,255,255,0.04);'>"
                             f"<td style='color:{MUTED}; padding:6px 0;'>{label}</td>"
                             f"<td style='color:{clr}; font-weight:700; text-align:right;'>{val}</td></tr>")
                return html + "</table>"

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown(f"""
                <div class="report-card">
                    <div class="report-title">🤖 AI Agent (LangGraph)</div>
                    {_table([
                        ("Average Temp", f"{avg_a:.2f}°C", GREEN),
                        ("Peak Temp", f"{max_a:.2f}°C", GREEN if max_a <= 45 else RED),
                        ("Violations (>45°C)", str(s.violations_a), GREEN if s.violations_a == 0 else RED),
                        ("Average Coolant Flow", f"{avg_flow:.2f} L/min", CYAN),
                        ("Total Pump Energy", f"{energy_agent_kj:.2f} kJ", GREEN),
                    ])}
                </div>""", unsafe_allow_html=True)

            with rc2:
                st.markdown(f"""
                <div class="report-card">
                    <div class="report-title">📊 Fixed Flow ({s.fixed_flow} L/min)</div>
                    {_table([
                        ("Average Temp", f"{avg_f:.2f}°C", AMBER),
                        ("Peak Temp", f"{max_f:.2f}°C", AMBER),
                        ("Violations (>45°C)", str(s.violations_f), AMBER),
                        ("Average Coolant Flow", f"{s.fixed_flow:.2f} L/min", MUTED),
                        ("Total Pump Energy", f"{energy_fixed_kj:.2f} kJ", AMBER),
                    ])}
                </div>""", unsafe_allow_html=True)

            with rc3:
                st.markdown(f"""
                <div class="report-card">
                    <div class="report-title">🔥 Passive (No Active Cooling)</div>
                    {_table([
                        ("Average Temp", f"{avg_n:.2f}°C", RED),
                        ("Peak Temp", f"{max_n:.2f}°C", RED),
                        ("Violations (>45°C)", str(s.violations_n), RED),
                        ("Average Coolant Flow", f"{FLOW_MIN:.2f} L/min", MUTED),
                        ("Total Pump Energy", "0.00 kJ", MUTED),
                    ])}
                </div>""", unsafe_allow_html=True)

        else:
            st.info("Run simulation to view benchmark analytics.", icon="📊")

    # ── Tab 4: Agent Memory & Decisions ───────────────────────────
    with tab_memory:
        if s.memory:
            st.markdown(f'<div class="section-header">🧠 Real-Time Rolling Memory Stream (Last 30 Decisions)</div>',
                        unsafe_allow_html=True)

            col_l, col_r = st.columns([2.5, 1])
            with col_r:
                st.markdown(f"""
                <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:14px;
                            padding:16px; font-size:0.8rem; color:{MUTED}; line-height:1.8;">
                    <b style="color:{TEXT};">State Classification</b><br>
                    🟢 <b>Safe Band:</b> 15°C – 42°C<br>
                    🟡 <b>Warning Band:</b> 42°C – 45°C<br>
                    🔴 <b>Violation Band:</b> &gt; 45°C<br><br>
                    <b style="color:{TEXT};">Reasoning Pipeline:</b><br>
                    1. Read pack telemetry & ambient<br>
                    2. Query ML surrogate model<br>
                    3. Calculate urgency coefficient<br>
                    4. Command proportional PWM pump
                </div>
                """, unsafe_allow_html=True)

            with col_l:
                for entry in reversed(s.memory):
                    T_e = entry["T"]
                    if T_e > T_MAX:
                        cls, icon, tc = "danger", "🔴 VIOLATION", RED
                    elif T_e > T_MAX - 3:
                        cls, icon, tc = "warn", "🟡 WARNING", AMBER
                    else:
                        cls, icon, tc = "", "🟢 NORMAL", GREEN

                    st.markdown(f"""
                    <div class="memory-entry {cls}">
                        <div>
                            <b>t={entry['t']:3d}s</b> &nbsp; {icon} &nbsp;
                            T=<span style="color:{tc}; font-weight:700;">{T_e:5.1f}°C</span>
                            (Pred: {entry.get('pred_T', T_e):.1f}°C)
                        </div>
                        <div>
                            Flow: <span style="color:{CYAN}; font-weight:700;">{entry['flow']:.2f} L/m</span> &nbsp;
                            HL: <span style="color:{AMBER};">{entry['hl']:.0f}W</span> &nbsp;
                            Power: <span style="color:{MUTED};">{entry.get('power', 0):.1f}W</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:14px;
                        padding:48px; text-align:center; color:{MUTED}; font-size:0.88rem;">
                🧠 The agent reasoning loop logs will stream here during execution.
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 5: Data Export & Reports ──────────────────────────────
    with tab_export:
        st.markdown(f'<div class="section-header">💾 Export Simulation Telemetry</div>', unsafe_allow_html=True)
        if s.times:
            df = pd.DataFrame({
                "time_s": s.times,
                "heat_load_w": s.heat_loads,
                "temp_agent_c": s.agent_temps,
                "flow_agent_lpm": s.agent_flows,
                "q_cool_agent_w": s.q_cool_agent,
                "pump_power_w": s.pump_power_agent,
                "temp_fixed_c": s.fixed_temps,
                "temp_none_c": s.none_temps,
            })

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Telemetry Dataset (.CSV)",
                data=csv,
                file_name=f"cooling_agent_telemetry_{int(time.time())}.csv",
                mime="text/csv",
            )
            st.dataframe(df.head(20), use_container_width=True)
        else:
            st.info("Execute a simulation to export dataset.", icon="💾")

    # ── Autonomous Simulation Execution Loop ─────────────────────
    if s.running and step_now < total_steps:
        advance_one_step(profile=s.get("profile", profile))
        if speed_ms > 0:
            time.sleep(speed_ms / 1000.0)
        if s.step < total_steps:
            st.rerun()
        else:
            s.running = False
            st.rerun()
    elif step_now >= total_steps and total_steps > 0 and not s.running and s.agent_temps:
        st.balloons()
        st.success(
            f"🎯 Simulation Cycle Completed! "
            f"AI Violations: **{s.violations_a}** | "
            f"Fixed Violations: **{s.violations_f}** | "
            f"Parasitic Energy Saved: **{energy_saved_pct:+.1f}%**"
        )


if __name__ == "__main__":
    main()
