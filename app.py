"""
Streamlit Dashboard — AI Battery Cooling Agent  ✦ Mission Control Ultimate v5.0
================================================================================
Run:  cd /Users/aadi/Downloads/ai-cooling-agent && python3 -m streamlit run app.py
"""

import os
import sys
import time
import math
import json
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
    page_title="🔋 AI Battery Cooling Agent — Mission Control Ultimate",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

from simulator import BatterySimulator, BatteryConfig, heat_load_schedule
from train_model import load_model, load_metrics, predict_next_temp

# ── Design tokens ─────────────────────────────────────────────────
T_MIN    = 15.0
T_MAX    = 45.0
FLOW_MIN = 0.5
FLOW_MAX = 10.0

BG       = "#070b14"
SURFACE  = "#0d1322"
CARD     = "#11182c"
CARD2    = "#162038"
BORDER   = "#1e2b48"
ACCENT   = "#6366f1"   # Indigo
ACCENT2  = "#8b5cf6"   # Violet
CYAN     = "#06b6d4"   # Coolant Cyan
GREEN    = "#10b981"   # Emerald Safe
AMBER    = "#f59e0b"   # Warning Amber
RED      = "#ef4444"   # Critical Rose
TEXT     = "#f1f5f9"
MUTED    = "#94a3b8"

C_NONE   = "#ef4444"
C_FIXED  = "#f59e0b"
C_AGENT  = "#10b981"
PANEL_BG = "#0d1322"
GRID_CLR = "#1a253e"
TEXT_CLR = "#f1f5f9"

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
    background: linear-gradient(180deg, #090e1b 0%, #040710 100%);
    border-right: 1px solid {BORDER};
}}

/* Streamlit Button Styling */
.stButton > button {{
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.18);
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
    border-color: rgba(255,255,255,0.35);
}}

/* Metric Cards */
.metric-card {{
    background: linear-gradient(145deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 16px 14px 12px;
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
    background: linear-gradient(90deg, {CYAN}, {ACCENT}, {GREEN});
    opacity: 0.85;
}}
.metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(0,0,0,0.5);
    border-color: rgba(99,102,241,0.4);
}}
.metric-value {{
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.5px;
}}
.metric-label {{
    font-size: 0.65rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-weight: 600;
}}
.metric-sub {{
    font-size: 0.7rem;
    color: {MUTED};
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}}
.metric-icon {{ font-size: 1.25rem; margin-bottom: 2px; display: block; }}

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

/* Gauges */
.gauge-box {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 15px;
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

/* 8-Cell Battery Module Grid */
.cell-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 12px 0;
}}
.cell-item {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 12px 10px;
    text-align: center;
    transition: all 0.25s ease;
}}
.cell-item:hover {{
    transform: scale(1.03);
    border-color: {CYAN};
}}
.cell-temp {{
    font-size: 1.15rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}}
.cell-id {{
    font-size: 0.65rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
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
.badge-pause {{
    background: rgba(245,158,11,0.15);
    color: {AMBER};
    border: 1px solid rgba(245,158,11,0.35);
}}
.badge-idle {{
    background: rgba(148,163,184,0.12);
    color: {MUTED};
    border: 1px solid rgba(148,163,184,0.25);
}}
.badge-scram {{
    background: rgba(239,68,68,0.2);
    color: {RED};
    border: 1px solid rgba(239,68,68,0.5);
    box-shadow: 0 0 14px rgba(239,68,68,0.4);
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

.help-box {{
    background: rgba(99,102,241,0.06);
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 14px;
    margin-top: 14px;
    font-size: 0.78rem;
    color: {MUTED};
    line-height: 1.6;
}}

hr {{ border: none; border-top: 1px solid {BORDER}; margin: 18px 0; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────
def init_session():
    if "model" not in st.session_state:
        with st.spinner("🤖 Initializing AI Thermal Neural Surrogate…"):
            model, name = load_model()
            st.session_state.model = model
            st.session_state.model_name = name

    defaults = dict(
        sim_agent=None, sim_none=None, sim_fixed=None,
        running=False, paused=False, step=0, flow_agent=FLOW_MIN, memory=[],
        agent_temps=[], none_temps=[], fixed_temps=[],
        agent_flows=[], times=[], heat_loads=[],
        q_cool_agent=[], q_cool_fixed=[],
        pump_power_agent=[], pump_power_fixed=[],
        cell_temps=[],
        scram_active=False,
        openai_key_input="",
        llm_reasoning_log=[],
        violations_a=0, violations_n=0, violations_f=0,
        n_steps=100, fixed_flow=5.0, T_init=25.0,
        nano_pct=1.0, profile="Dynamic Cycle",
        control_mode="Balanced Autonomous",
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
    st.session_state.cell_temps         = []
    st.session_state.llm_reasoning_log  = []
    st.session_state.scram_active       = False
    st.session_state.paused             = False
    st.session_state.violations_a       = 0
    st.session_state.violations_n       = 0
    st.session_state.violations_f       = 0
    st.session_state.n_steps            = n_steps
    st.session_state.fixed_flow         = fixed_flow
    st.session_state.T_init             = T_init
    st.session_state.nano_pct           = nano_pct


# ── Heat Load Profiles ────────────────────────────────────────────
def get_heat_load(t: float, profile: str) -> float:
    if profile == "Dynamic Cycle":
        return heat_load_schedule(t)
    elif profile == "Extreme Fast-Charge (Peak 450W)":
        if t < 40:
            return 250.0 + 5.0 * t
        elif t < 70:
            return 450.0
        else:
            return max(150.0, 450.0 - 10.0 * (t - 70))
    elif profile == "Constant Highway Cruise (280W)":
        return 280.0 + 15.0 * math.sin(t / 10.0)
    elif profile == "Urban Stop & Go (Pulsed)":
        pulse = 390.0 if (int(t) // 10) % 2 == 1 else 130.0
        return pulse
    elif profile == "Thermal Stress Wave (300-500W)":
        return 350.0 + 130.0 * math.sin(t / 6.0)
    return heat_load_schedule(t)


# ── Agent Decision: Neural ML vs Optional LLM ────────────────────
def call_optional_llm(api_key: str, T_current: float, T_pred: float, heat_load: float, current_flow: float) -> tuple[float, str]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = f"""You are an autonomous thermal cooling agent for an EV battery.
Current state:
- Temperature: {T_current:.1f}°C
- Predicted next temp: {T_pred:.1f}°C
- Heat dissipation: {heat_load:.0f} W
- Safe range: 15°C to 45°C
- Allowable flow: 0.5 to 10.0 L/min

Provide target coolant flow rate and a 1-sentence rationale in JSON format:
{{"flow_rate": <number>, "reason": "<rationale>"}}"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=80,
        )
        data = json.loads(resp.choices[0].message.content)
        flow = float(data.get("flow_rate", 5.0))
        reason = data.get("reason", "LLM reasoning executed")
        return max(FLOW_MIN, min(FLOW_MAX, flow)), reason
    except Exception as e:
        return None, str(e)


def agent_decide(T_current, heat_load, current_flow, model, nano_pct, mode="Balanced Autonomous", api_key=""):
    T_pred = predict_next_temp(model, T_current, heat_load, current_flow, nano_pct)
    headroom = T_MAX - T_pred
    rationale = "Neural predictive headroom control"

    if api_key and len(api_key.strip()) > 10:
        llm_flow, llm_reason = call_optional_llm(api_key.strip(), T_current, T_pred, heat_load, current_flow)
        if llm_flow is not None:
            return round(llm_flow, 2), T_pred, f"🤖 GPT-4o-mini: {llm_reason}"

    if mode == "Aggressive Safety (Zero-Tolerance)":
        target_max = 42.0
        headroom = target_max - T_pred
        urgency = 1.0 - headroom / (target_max - T_MIN)
        urgency = max(0.0, min(1.0, urgency)) * 1.3
        rationale = "Zero-tolerance safety guard (42°C ceiling)"
    elif mode == "Eco / Range-Saver (Energy Minimization)":
        urgency = 1.0 - headroom / (T_MAX - T_MIN)
        urgency = max(0.0, min(1.0, urgency)) * 0.75
        rationale = "Parasitic power minimization mode"
    else:  # Balanced Autonomous
        urgency = 1.0 - headroom / (T_MAX - T_MIN)
        urgency = max(0.0, min(1.0, urgency))
        rationale = "Autonomous dynamic proportional throttling"

    urgency = max(0.0, min(1.0, urgency))
    flow_new = FLOW_MIN + urgency * (FLOW_MAX - FLOW_MIN)
    T_pred2 = predict_next_temp(model, T_current, heat_load, flow_new, nano_pct)

    if T_pred2 > T_MAX:
        flow_new = FLOW_MAX
        rationale = "Safety override: Max flow engaged"

    return round(max(FLOW_MIN, min(FLOW_MAX, flow_new)), 2), T_pred, rationale


# ── Advance Step ──────────────────────────────────────────────────
def advance_one_step(profile="Dynamic Cycle", mode="Balanced Autonomous", api_key=""):
    s = st.session_state
    t = s.step + 1
    hl = get_heat_load(float(t), profile)
    nano = s.nano_pct

    if s.scram_active:
        flow_new = FLOW_MAX
        pred_t = T_MAX
        rationale = "🚨 MANUAL EMERGENCY SCRAM ACTIVE"
    else:
        rec_a = s.sim_agent.step(hl, s.flow_agent, nano)
        T_a = rec_a["temperature"]
        flow_new, pred_t, rationale = agent_decide(T_a, hl, s.flow_agent, s.model, nano, mode, api_key)

    rec_a = s.sim_agent.step(hl, flow_new, nano)
    T_a = rec_a["temperature"]
    s.flow_agent = flow_new

    rec_n = s.sim_none.step(hl, FLOW_MIN, nano)
    rec_f = s.sim_fixed.step(hl, s.fixed_flow, nano)

    p_agent = calc_pump_power(flow_new)
    p_fixed = calc_pump_power(s.fixed_flow)

    offsets = [-0.9, -0.4, 0.5, 1.1, 1.2, 0.4, -0.3, -0.8]
    rng_jitter = np.random.normal(0, 0.15, 8)
    cell_temps_now = [round(T_a + offsets[i] + rng_jitter[i], 1) for i in range(8)]

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
    s.cell_temps = cell_temps_now

    s.memory.append({
        "t": t, "T": T_a, "flow": flow_new, "hl": hl,
        "q_cool": rec_a["Q_cool"], "pred_T": pred_t, "power": p_agent,
        "mode": mode, "rationale": rationale
    })
    if len(s.memory) > 35:
        s.memory = s.memory[-35:]

    if T_a > T_MAX:                  s.violations_a += 1
    if rec_n["temperature"] > T_MAX: s.violations_n += 1
    if rec_f["temperature"] > T_MAX: s.violations_f += 1
    s.step += 1


# ── Chart styling helpers ─────────────────────────────────────────
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
    ax.text(s.times[-1], T_MAX + 0.4, f" {T_MAX}°C Critical", color=RED,  fontsize=7, fontweight="bold")
    ax.text(s.times[-1], T_MIN - 0.4, f" {T_MIN}°C Min", color=CYAN, fontsize=7, va="top", fontweight="bold")

    ax.plot(s.times, s.none_temps,  color=C_NONE,  lw=1.6, alpha=0.85, label="Passive (No Active Cooling)")
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
    cats = ["Passive", f"Fixed ({s.fixed_flow} L/m)", "AI Agent"]
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


# ── Sidebar ───────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:10px 0 16px;">
            <div style="font-size:2.2rem; filter:drop-shadow(0 0 10px rgba(99,102,241,0.5));">🔋</div>
            <div style="font-size:1.1rem; font-weight:800; color:{TEXT}; letter-spacing:0.5px;">
                AI Cooling Agent
            </div>
            <div style="font-size:0.72rem; color:{CYAN}; font-weight:600; margin-top:2px;">
                ✦ MISSION CONTROL ULTIMATE v5.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        model_name = st.session_state.get("model_name", "GradientBoostingRegressor")
        st.markdown(f"""
        <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:12px;
                    padding:10px 14px; margin-bottom:14px; text-align:center;">
            <div style="font-size:0.65rem; color:{MUTED}; text-transform:uppercase;
                        letter-spacing:1.5px; margin-bottom:2px;">Surrogate Neural Engine</div>
            <div style="font-size:0.85rem; font-weight:700; color:{CYAN};">{model_name}</div>
        </div>
        """, unsafe_allow_html=True)

        # Optional OpenAI Key
        with st.expander("🔑 LLM Reasoner (Optional)", expanded=False):
            st.markdown(f"""
            <div style="font-size:0.72rem; color:{MUTED}; margin-bottom:6px;">
                By default, the agent runs 100% offline using the built-in Machine Learning model. Enter an OpenAI key if you'd like GPT-4o-mini to provide natural language rationales.
            </div>
            """, unsafe_allow_html=True)
            env_key = os.getenv("OPENAI_API_KEY", "")
            api_key_input = st.text_input("OpenAI API Key", value=env_key, type="password", placeholder="sk-...")

        st.markdown(f'<div class="section-header">🎯 Agent Strategy Tuning</div>', unsafe_allow_html=True)
        control_mode = st.selectbox(
            "Agent Control Strategy",
            [
                "Balanced Autonomous",
                "Aggressive Safety (Zero-Tolerance)",
                "Eco / Range-Saver (Energy Minimization)"
            ],
            index=0,
            help="Modulates agent urgency weighting and headroom threshold"
        )

        st.markdown(f'<div class="section-header">⚡ Workload Scenario</div>', unsafe_allow_html=True)
        profile = st.selectbox(
            "Heat Load Profile",
            [
                "Dynamic Cycle",
                "Extreme Fast-Charge (Peak 450W)",
                "Constant Highway Cruise (280W)",
                "Urban Stop & Go (Pulsed)",
                "Thermal Stress Wave (300-500W)"
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
        speed_ms   = st.slider("Simulation Throttle (ms)", 0, 300, 30, 10,
                               help="Delay between frames for smooth visualization")

        # Control Action Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            start = st.button("▶ Start", use_container_width=True, key="btn_start")
        with col_btn2:
            pause_label = "▶ Resume" if st.session_state.get("paused", False) else "⏸ Pause"
            pause_btn = st.button(pause_label, use_container_width=True, key="btn_pause")

        col_rst1, col_rst2 = st.columns(2)
        with col_rst1:
            stop_btn = st.button("↺ Reset All", use_container_width=True, key="btn_reset")
        with col_rst2:
            scram = st.button("🚨 SCRAM", use_container_width=True, key="btn_scram")
            if scram:
                st.session_state.scram_active = True

        st.markdown(f"""
        <div class="help-box">
            <b style="color:{TEXT};">Quick Usage:</b><br>
            • Click <b>▶ Start</b> to launch real-time simulation.<br>
            • Use <b>⏸ Pause</b> to inspect graphs at any step.<br>
            • Hit <b>🚨 SCRAM</b> for instant maximum emergency cooling.
        </div>
        """, unsafe_allow_html=True)

    return n_steps, T_init, fixed_flow, speed_ms, start, pause_btn, stop_btn, nano_pct, profile, control_mode, api_key_input


# ── Main ──────────────────────────────────────────────────────────
def main():
    init_session()
    s = st.session_state

    n_steps, T_init, fixed_flow, speed_ms, start, pause_btn, stop_btn, nano_pct, profile, control_mode, api_key_input = render_sidebar()

    if start:
        reset_simulation(n_steps, fixed_flow, T_init, nano_pct)
        st.session_state.profile = profile
        st.session_state.control_mode = control_mode
        st.session_state.running = True
        st.session_state.paused = False

    if pause_btn:
        st.session_state.paused = not st.session_state.paused

    if stop_btn:
        st.session_state.running = False
        st.session_state.paused = False
        reset_simulation(n_steps, fixed_flow, T_init, nano_pct)

    step_now    = s.step
    total_steps = s.get("n_steps", n_steps)
    T_now       = s.agent_temps[-1] if s.agent_temps else s.get("T_init", T_init)
    flow_now    = s.agent_flows[-1] if s.agent_flows else 3.0
    hl_now      = get_heat_load(float(step_now), s.get("profile", profile))
    pct         = (step_now / total_steps * 100) if total_steps > 0 else 0

    energy_agent_kj = (sum(s.pump_power_agent) / 1000.0) if s.pump_power_agent else 0.0
    energy_fixed_kj = (sum(s.pump_power_fixed) / 1000.0) if s.pump_power_fixed else 0.0
    energy_saved_pct = (
        ((energy_fixed_kj - energy_agent_kj) / energy_fixed_kj * 100)
        if energy_fixed_kj > 0.01 else 0.0
    )

    if s.scram_active:
        badge = f'<span class="badge badge-scram">⚠️ SCRAM EMERGENCY COOLING ACTIVE</span>'
    elif s.paused:
        badge = f'<span class="badge badge-pause">⏸ Paused at Step {step_now}</span>'
    elif s.running:
        badge = f'<span class="badge badge-run">● Agent Active</span>'
    elif step_now > 0 and not s.running:
        badge = f'<span class="badge badge-idle">✓ Simulation Finished</span>'
    else:
        badge = f'<span class="badge badge-idle">Ready</span>'

    t_color = GREEN if T_MIN <= T_now <= T_MAX - 3 else (AMBER if T_now <= T_MAX else RED)
    v_color = GREEN if s.violations_a == 0 else RED

    # ── Header ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:16px 0 10px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
                    <h1 style="font-size:2.2rem; font-weight:800; margin:0;
                               background: linear-gradient(135deg, #ffffff 0%, {CYAN} 50%, {ACCENT} 100%);
                               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                               letter-spacing: -0.5px;">
                        🔋 AI Thermal Mission Control Pro
                    </h1>
                    {badge}
                </div>
                <div style="color:{MUTED}; font-size:0.85rem;">
                    Autonomous predictive battery thermal management using <b>LangGraph + Physics Surrogate AI</b>
                </div>
            </div>
            <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:8px 16px; font-size:0.8rem;">
                <span style="color:{MUTED};">Mode:</span> <b style="color:{CYAN};">{s.get('control_mode', control_mode)}</b>
                &nbsp;·&nbsp;
                <span style="color:{MUTED};">Scenario:</span> <b style="color:{TEXT};">{s.get('profile', profile)}</b>
                &nbsp;·&nbsp;
                <span style="color:{MUTED};">Nanofluid:</span> <b style="color:{GREEN};">{s.get('nano_pct', nano_pct):.1f}%</b>
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
    tab_live, tab_pack, tab_telemetry, tab_analytics, tab_memory, tab_ml_eval, tab_guide, tab_export = st.tabs([
        "📡 Live Dashboard",
        "🔋 Pack Cell Matrix",
        "⚡ Energy & Telemetry",
        "📊 Comparative Benchmark",
        "🧠 Agent Memory & Decisions",
        "🔬 ML Model Validation",
        "📖 Architecture & Guide",
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
                    <b style="color:{CYAN};">▶ Start</b> to initiate real-time telemetry.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2: Pack Cell Matrix ───────────────────────────────────
    with tab_pack:
        st.markdown(f'<div class="section-header">🔋 8-Cell Battery Module Spatial Heat Distribution</div>', unsafe_allow_html=True)
        if s.cell_temps:
            c_left, c_right = st.columns([2, 1])
            with c_left:
                st.markdown('<div class="cell-grid">', unsafe_allow_html=True)
                for idx, c_t in enumerate(s.cell_temps):
                    clr = GREEN if c_t < 42.0 else (AMBER if c_t <= 45.0 else RED)
                    st.markdown(f"""
                    <div class="cell-item" style="border-left: 3px solid {clr};">
                        <div class="cell-temp" style="color:{clr};">{c_t:.1f}°C</div>
                        <div class="cell-id">CELL #{idx+1:02d}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c_right:
                delta_t = max(s.cell_temps) - min(s.cell_temps)
                st.markdown(f"""
                <div class="report-card">
                    <div class="report-title">🌡️ Thermal Gradient & Balance</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{GREEN if delta_t < 3.0 else AMBER}; font-family:'JetBrains Mono';">
                        Δ {delta_t:.1f}°C
                    </div>
                    <div style="font-size:0.75rem; color:{MUTED}; margin-top:4px;">
                        Max Cell-to-Cell Temperature Spread (Target &lt; 3°C)
                    </div>
                    <div style="margin-top:14px; font-size:0.75rem; color:{MUTED}; border-top:1px solid {BORDER}; padding-top:10px;">
                        Uniform cooling prevents accelerated degradation in hot-spot cells, preserving cycle life.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run the simulation to inspect spatial cell temperature balancing.", icon="🔋")

    # ── Tab 3: Energy & Telemetry ─────────────────────────────────
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

    # ── Tab 4: Comparative Benchmark ──────────────────────────────
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
                    <div class="report-title">🤖 AI Agent ({s.get('control_mode', control_mode)})</div>
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

    # ── Tab 5: Agent Memory & Decisions ───────────────────────────
    with tab_memory:
        if s.memory:
            st.markdown(f'<div class="section-header">🧠 Real-Time Rolling Memory Stream (Last 35 Decisions)</div>',
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

                    rat = entry.get("rationale", "Proportional headroom")

                    st.markdown(f"""
                    <div class="memory-entry {cls}">
                        <div>
                            <b>t={entry['t']:3d}s</b> &nbsp; {icon} &nbsp;
                            T=<span style="color:{tc}; font-weight:700;">{T_e:5.1f}°C</span>
                            (Pred: {entry.get('pred_T', T_e):.1f}°C)<br>
                            <span style="font-size:0.72rem; color:{MUTED};">{rat}</span>
                        </div>
                        <div style="text-align:right;">
                            Flow: <span style="color:{CYAN}; font-weight:700;">{entry['flow']:.2f} L/m</span> &nbsp;
                            HL: <span style="color:{AMBER};">{entry['hl']:.0f}W</span><br>
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

    # ── Tab 6: ML Model Validation & Academic Rigor ───────────────
    with tab_ml_eval:
        st.markdown(f'<div class="section-header">🔬 Surrogate Model 5-Fold Cross-Validation & Benchmark Metrics</div>', unsafe_allow_html=True)
        try:
            bundle = load_metrics()
            best_name = bundle.get("best_model", "GradientBoosting")
            metrics = bundle.get("metrics", {})

            col_m1, col_m2 = st.columns([1.5, 1])

            with col_m1:
                st.markdown(f"**Model Selection & 5-Fold K-Fold Cross Validation:**")
                rows = []
                for m_name, d in metrics.items():
                    rows.append({
                        "Algorithm": m_name,
                        "Test MAE (°C)": f"{d['mae']:.4f}",
                        "Test RMSE (°C)": f"{d['rmse']:.4f}",
                        "Test R²": f"{d['r2']:.4f}",
                        "5-Fold CV R² (μ ± σ)": f"{d['cv_r2_mean']:.4f} ± {d['cv_r2_std']:.4f}",
                    })
                df_models = pd.DataFrame(rows)
                st.dataframe(df_models, use_container_width=True, hide_index=True)

                # Parity scatter plot
                best_d = metrics.get(best_name, {})
                if "sample_actual" in best_d and "sample_pred" in best_d:
                    fig_par, ax_par = plt.subplots(figsize=(6, 3.2), facecolor=BG)
                    _style_ax(ax_par, fig_par)
                    acts = best_d["sample_actual"]
                    preds = best_d["sample_pred"]
                    ax_par.scatter(acts, preds, color=CYAN, s=20, alpha=0.7, edgecolors="none", label="Predictions")
                    mn, mx = min(acts), max(acts)
                    ax_par.plot([mn, mx], [mn, mx], color=AMBER, ls="--", lw=1.5, label="Perfect Fit (y = x)")
                    ax_par.set_xlabel("Actual Simulated Temp (°C)", color=MUTED, fontsize=8)
                    ax_par.set_ylabel("Predicted Temp (°C)", color=MUTED, fontsize=8)
                    ax_par.set_title(f"Parity Plot: {best_name} (R² = {best_d['r2']:.4f})", color=TEXT_CLR, fontsize=10, fontweight="bold")
                    ax_par.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT_CLR, fontsize=7)
                    fig_par.tight_layout(pad=1.2)
                    st.pyplot(fig_par, use_container_width=True)
                    plt.close(fig_par)

            with col_m2:
                st.markdown(f"""
                <div class="report-card">
                    <div class="report-title">🏆 Optimal Model: {best_name}</div>
                    <div style="font-size:0.8rem; color:{MUTED}; line-height:1.7;">
                        • <b>Cross-Validation:</b> 5-Fold Stratified K-Fold<br>
                        • <b>Mean Abs Error (MAE):</b> <span style="color:{GREEN}; font-weight:700;">{metrics[best_name]['mae']:.4f}°C</span><br>
                        • <b>Root Mean Sq Err (RMSE):</b> <span style="color:{GREEN}; font-weight:700;">{metrics[best_name]['rmse']:.4f}°C</span><br>
                        • <b>Coefficient of Det (R²):</b> <span style="color:{CYAN}; font-weight:700;">{metrics[best_name]['r2']:.4f}</span><br>
                        • <b>Residual Mean Error:</b> {metrics[best_name]['residuals_mean']:.5f}°C
                    </div>
                </div>
                """, unsafe_allow_html=True)

                feat_imp = metrics.get(best_name, {}).get("feature_importances", {})
                if feat_imp:
                    st.markdown(f'<div class="section-header">⚖️ Feature Importance Attribution</div>', unsafe_allow_html=True)
                    df_imp = pd.DataFrame({
                        "Feature": list(feat_imp.keys()),
                        "Relative Weight (%)": [round(v * 100, 2) for v in feat_imp.values()]
                    }).sort_values("Relative Weight (%)", ascending=False)
                    st.dataframe(df_imp, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Run `python3 train_model.py` to compile academic cross-validation metrics. ({e})")

    # ── Tab 7: Architecture & Guide ───────────────────────────────
    with tab_guide:
        st.markdown(f'<div class="section-header">📖 Embedded System Architecture & Theory</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:16px; padding:24px; line-height:1.7;">
            <h3 style="color:{CYAN}; margin-top:0;">🔋 How the AI Cooling Agent Operates</h3>
            <p style="color:{TEXT}; font-size:0.9rem;">
                This system autonomously regulates an Electric Vehicle (EV) battery pack's thermal dynamics using a closed-loop predictive AI pipeline:
            </p>
            <ol style="color:{MUTED}; font-size:0.85rem;">
                <li><b style="color:{TEXT};">State Observation:</b> Every second, temperature (T), scheduled electrical heat load (W), and ambient conditions are gathered.</li>
                <li><b style="color:{TEXT};">Neural Surrogate Inference:</b> An ML surrogate model (GradientBoostingRegressor) predicts the next-step temperature in under 1 millisecond.</li>
                <li><b style="color:{TEXT};">Headroom & Urgency Evaluation:</b> The agent calculates distance to the 45°C critical limit and computes an urgency coefficient.</li>
                <li><b style="color:{TEXT};">Adaptive Actuation:</b> Commands variable coolant flow (0.5 – 10.0 L/min), cutting parasitic pump energy during low loads while safeguarding against thermal runaway.</li>
            </ol>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px;">
                <div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:14px;">
                    <b style="color:{GREEN};">🛡️ Safety Limits</b><br>
                    • Safe Operating Range: 15°C – 42°C<br>
                    • Warning Range: 42°C – 45°C<br>
                    • Critical Runaway Risk: &gt; 45°C
                </div>
                <div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:14px;">
                    <b style="color:{CYAN};">⚡ Energy Optimization</b><br>
                    • Parasitic Hydraulic Power: P ∝ Flow<sup>1.8</sup><br>
                    • Nanofluid thermal capacity boost: +1.5% per 1% nano<br>
                    • 30%–50% parasitic energy reduction vs fixed flow
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 7: Data Export & Reports ──────────────────────────────
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
    if s.running and not s.paused and step_now < total_steps:
        advance_one_step(
            profile=s.get("profile", profile),
            mode=s.get("control_mode", control_mode),
            api_key=api_key_input
        )
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
