"""
Streamlit Dashboard — AI Battery Cooling Agent  ✦ Premium Edition
=================================================================
Run:  streamlit run app.py
"""

import os
import sys
import time
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
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔋 AI Battery Cooling Agent",
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
NANO_PCT = 1.0

BG      = "#080b14"
SURFACE = "#0e1220"
CARD    = "#131729"
CARD2   = "#181c2e"
BORDER  = "#252a42"
ACCENT  = "#7c83fd"
ACCENT2 = "#5c6cfa"
GREEN   = "#4ade80"
AMBER   = "#fbbf24"
RED     = "#f87171"
CYAN    = "#22d3ee"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"

C_NONE  = "#f87171"
C_FIXED = "#fbbf24"
C_AGENT = "#4ade80"
PANEL_BG = "#0e1220"
GRID_CLR = "#1e2438"
TEXT_CLR = "#e2e8f0"

# ── CSS ───────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {BG};
    color: {TEXT};
}}
.stApp {{ background-color: {BG}; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {SURFACE}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {SURFACE} 0%, #0a0d1a 100%);
    border-right: 1px solid {BORDER};
}}

.stButton > button {{
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 10px 20px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(124,131,253,0.25);
    width: 100%;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124,131,253,0.45);
}}

.metric-card {{
    background: linear-gradient(135deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 18px 16px 14px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, {ACCENT}, transparent);
    opacity: 0.5;
}}
.metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}}
.metric-value {{
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}}
.metric-label {{
    font-size: 0.7rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}}
.metric-sub {{
    font-size: 0.72rem;
    color: {MUTED};
    margin-top: 4px;
}}
.metric-icon {{ font-size: 1.4rem; margin-bottom: 6px; display: block; }}

.status-safe   {{ color: {GREEN}; }}
.status-warn   {{ color: {AMBER}; }}
.status-danger {{ color: {RED}; }}

.section-header {{
    font-size: 0.72rem;
    font-weight: 700;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 8px;
    margin-bottom: 16px;
}}

.memory-entry {{
    background: {CARD};
    border-left: 3px solid {ACCENT};
    padding: 7px 12px;
    margin: 3px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: {TEXT};
    transition: background 0.2s;
}}
.memory-entry:hover {{ background: {CARD2}; }}
.memory-entry.danger {{ border-left-color: {RED}; }}

.prog-container {{
    background: {CARD};
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin: 8px 0;
    border: 1px solid {BORDER};
}}
.prog-fill {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, {ACCENT2}, {ACCENT}, {CYAN});
    transition: width 0.4s ease;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDER};
    gap: 4px;
    padding: 0 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {MUTED};
    font-weight: 600;
    font-size: 0.85rem;
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    border: none;
}}
.stTabs [aria-selected="true"] {{
    background: {CARD} !important;
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}

.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
.badge-run  {{ background: rgba(124,131,253,0.12); color: {ACCENT}; border: 1px solid rgba(124,131,253,0.3); }}
.badge-idle {{ background: rgba(100,116,139,0.12); color: {MUTED}; border: 1px solid rgba(100,116,139,0.3); }}

.report-card {{
    background: linear-gradient(135deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
}}
.report-title {{
    font-size: 1rem;
    font-weight: 700;
    color: {TEXT};
    margin-bottom: 16px;
}}

hr {{ border: none; border-top: 1px solid {BORDER}; margin: 16px 0; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────
def init_session():
    if "model" not in st.session_state:
        with st.spinner("🤖 Loading / training prediction model…"):
            model, name = load_model()
            st.session_state.model = model
            st.session_state.model_name = name

    defaults = dict(
        sim_agent=None, sim_none=None, sim_fixed=None,
        running=False, step=0, flow_agent=FLOW_MIN, memory=[],
        agent_temps=[], none_temps=[], fixed_temps=[],
        agent_flows=[], times=[], heat_loads=[],
        violations_a=0, violations_n=0, violations_f=0,
        n_steps=100, fixed_flow=5.0, T_init=25.0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_simulation(n_steps, fixed_flow, T_init):
    cfg = BatteryConfig()
    st.session_state.sim_agent = BatterySimulator(T_init=T_init, config=cfg, seed=99)
    st.session_state.sim_none  = BatterySimulator(T_init=T_init, config=cfg, seed=99)
    st.session_state.sim_fixed = BatterySimulator(T_init=T_init, config=cfg, seed=99)
    st.session_state.step         = 0
    st.session_state.flow_agent   = 3.0
    st.session_state.memory       = []
    st.session_state.agent_temps  = []
    st.session_state.none_temps   = []
    st.session_state.fixed_temps  = []
    st.session_state.agent_flows  = []
    st.session_state.times        = []
    st.session_state.heat_loads   = []
    st.session_state.violations_a = 0
    st.session_state.violations_n = 0
    st.session_state.violations_f = 0
    st.session_state.n_steps      = n_steps
    st.session_state.fixed_flow   = fixed_flow
    st.session_state.T_init       = T_init


# ── Agent logic ───────────────────────────────────────────────────
def agent_decide(T_current, heat_load, current_flow, model, memory):
    T_pred   = predict_next_temp(model, T_current, heat_load, current_flow, NANO_PCT)
    headroom = T_MAX - T_pred
    urgency  = 1.0 - headroom / (T_MAX - T_MIN)
    urgency  = max(0.0, min(1.0, urgency))
    flow_new = FLOW_MIN + urgency * (FLOW_MAX - FLOW_MIN)
    T_pred2  = predict_next_temp(model, T_current, heat_load, flow_new, NANO_PCT)
    if T_pred2 > T_MAX:
        flow_new = FLOW_MAX
    return round(max(FLOW_MIN, min(FLOW_MAX, flow_new)), 2), T_pred


def advance_one_step():
    s  = st.session_state
    t  = s.step + 1
    hl = heat_load_schedule(float(t))

    rec_a = s.sim_agent.step(hl, s.flow_agent, NANO_PCT)
    T_a   = rec_a["temperature"]
    flow_new, _ = agent_decide(T_a, hl, s.flow_agent, s.model, s.memory)
    s.flow_agent = flow_new

    rec_n = s.sim_none.step(hl, FLOW_MIN,     NANO_PCT)
    rec_f = s.sim_fixed.step(hl, s.fixed_flow, NANO_PCT)

    s.times.append(t)
    s.heat_loads.append(hl)
    s.agent_temps.append(T_a)
    s.none_temps.append(rec_n["temperature"])
    s.fixed_temps.append(rec_f["temperature"])
    s.agent_flows.append(flow_new)
    s.memory.append({"t": t, "T": T_a, "flow": flow_new, "hl": hl})
    if len(s.memory) > 20:
        s.memory = s.memory[-20:]

    if T_a > T_MAX:                      s.violations_a += 1
    if rec_n["temperature"] > T_MAX:     s.violations_n += 1
    if rec_f["temperature"] > T_MAX:     s.violations_f += 1
    s.step += 1


# ── Chart helpers ─────────────────────────────────────────────────
def _style_ax(ax, fig=None):
    ax.set_facecolor(PANEL_BG)
    if fig:
        fig.patch.set_facecolor(BG)
    ax.grid(color=GRID_CLR, lw=0.7, linestyle="--", alpha=0.8)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)


def make_temp_chart():
    s = st.session_state
    if not s.times:
        return None
    fig, ax = plt.subplots(figsize=(11, 4), facecolor=BG)
    _style_ax(ax, fig)
    ax.fill_between(s.times, T_MIN, T_MAX, color=ACCENT, alpha=0.05)
    ax.axhline(T_MAX, color=RED,  ls="--", lw=1.0, alpha=0.6)
    ax.axhline(T_MIN, color=CYAN, ls="--", lw=1.0, alpha=0.6)
    ax.text(s.times[-1], T_MAX + 0.5, f" {T_MAX}°C limit", color=RED,  fontsize=7)
    ax.text(s.times[-1], T_MIN - 0.5, f" {T_MIN}°C limit", color=CYAN, fontsize=7, va="top")
    ax.plot(s.times, s.none_temps,  color=C_NONE,  lw=1.6, alpha=0.85, label="No Cooling")
    ax.plot(s.times, s.fixed_temps, color=C_FIXED, lw=1.6, alpha=0.85,
            label=f"Fixed ({s.fixed_flow} L/min)")
    ax.fill_between(s.times, s.agent_temps, alpha=0.12, color=C_AGENT)
    ax.plot(s.times, s.agent_temps, color=C_AGENT, lw=2.5, label="AI Agent", zorder=5)
    ax.set_xlabel("Time (s)", color=MUTED, fontsize=8)
    ax.set_ylabel("Temperature (°C)", color=MUTED, fontsize=8)
    ax.set_title("Battery Temperature — Live Comparison", color=TEXT_CLR, fontsize=11,
                 fontweight="bold", pad=12)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT_CLR, fontsize=8,
              loc="upper left", framealpha=0.9)
    fig.tight_layout(pad=1.5)
    return fig


def make_flow_chart():
    s = st.session_state
    if not s.times:
        return None
    fig, ax = plt.subplots(figsize=(11, 3), facecolor=BG)
    _style_ax(ax, fig)
    ax.fill_between(s.times, s.agent_flows, FLOW_MIN, alpha=0.18, color=C_AGENT)
    ax.plot(s.times, s.agent_flows, color=C_AGENT, lw=2.2, label="AI Agent flow")
    ax.axhline(s.fixed_flow, color=C_FIXED, ls="--", lw=1.2, alpha=0.7,
               label=f"Fixed ({s.fixed_flow} L/min)")
    ax.set_xlabel("Time (s)", color=MUTED, fontsize=8)
    ax.set_ylabel("Flow Rate (L/min)", color=MUTED, fontsize=8)
    ax.set_title("AI Agent — Adaptive Coolant Flow", color=TEXT_CLR, fontsize=11,
                 fontweight="bold", pad=12)
    ax.set_ylim(-0.2, FLOW_MAX + 1.5)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT_CLR, fontsize=8, framealpha=0.9)
    fig.tight_layout(pad=1.5)
    return fig


def make_heat_load_chart():
    s = st.session_state
    if not s.times:
        return None
    fig, ax = plt.subplots(figsize=(11, 2.5), facecolor=BG)
    _style_ax(ax, fig)
    ax.fill_between(s.times, s.heat_loads, alpha=0.2, color=AMBER)
    ax.plot(s.times, s.heat_loads, color=AMBER, lw=1.8, label="Heat Load (W)")
    ax.set_xlabel("Time (s)", color=MUTED, fontsize=8)
    ax.set_ylabel("Watts", color=MUTED, fontsize=8)
    ax.set_title("Heat Load Schedule", color=TEXT_CLR, fontsize=11, fontweight="bold", pad=12)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT_CLR, fontsize=8, framealpha=0.9)
    fig.tight_layout(pad=1.5)
    return fig


def make_violation_bar():
    s = st.session_state
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=BG)
    _style_ax(ax, fig)
    cats = ["No Cooling", f"Fixed\n{s.fixed_flow} L/min", "AI Agent"]
    vals = [s.violations_n, s.violations_f, s.violations_a]
    bars = ax.bar(cats, vals, color=[C_NONE, C_FIXED, C_AGENT], edgecolor=BORDER,
                  linewidth=0.8, width=0.5, zorder=3)
    ymax = max(vals) if max(vals) > 0 else 1
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.03,
                str(v), ha="center", color=TEXT_CLR, fontsize=13, fontweight="bold")
    ax.set_ylabel("Steps > 45°C", color=MUTED, fontsize=8)
    ax.set_title("Safety Violations Comparison", color=TEXT_CLR, fontsize=11,
                 fontweight="bold", pad=12)
    ax.set_ylim(0, ymax * 1.3 + 1)
    fig.tight_layout(pad=1.5)
    return fig


def make_phase_heatmap():
    s = st.session_state
    if not s.times:
        return None
    arr     = np.array(s.agent_temps).reshape(1, -1)
    encoded = np.where(arr > T_MAX, 2, np.where(arr > T_MAX - 3, 1, 0))
    cmap    = mcolors.ListedColormap([GREEN, AMBER, RED])
    norm    = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(11, 1.2), facecolor=BG)
    _style_ax(ax, fig)
    ax.imshow(encoded, aspect="auto", cmap=cmap, norm=norm,
              extent=[s.times[0], s.times[-1], 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", color=MUTED, fontsize=8)
    ax.set_title("Temperature Phase Timeline  (🟢 Safe  🟡 Warning  🔴 Danger)",
                 color=TEXT_CLR, fontsize=9, fontweight="bold", pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.8)
    return fig


def make_scatter():
    s = st.session_state
    if len(s.times) < 5:
        return None
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    _style_ax(ax, fig)
    sc = ax.scatter(s.agent_flows, s.agent_temps, c=s.heat_loads,
                    cmap="plasma", s=20, alpha=0.7, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
    cbar.set_label("Heat Load (W)", color=MUTED, fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)
    ax.axhline(T_MAX, color=RED, ls="--", lw=1.0, alpha=0.7, label=f"Danger > {T_MAX}°C")
    ax.set_xlabel("Flow Rate (L/min)", color=MUTED, fontsize=8)
    ax.set_ylabel("Battery Temp (°C)", color=MUTED, fontsize=8)
    ax.set_title("Flow vs Temperature  (color = Heat Load)", color=TEXT_CLR,
                 fontsize=10, fontweight="bold", pad=10)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT_CLR, fontsize=8)
    fig.tight_layout(pad=1.5)
    return fig


# ── HTML helpers ──────────────────────────────────────────────────
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


def progress_bar_html(pct):
    return f"""
    <div class="prog-container">
        <div class="prog-fill" style="width:{pct:.1f}%;"></div>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:16px 0 20px;">
            <div style="font-size:2.4rem;">🔋</div>
            <div style="font-size:1rem; font-weight:800; color:{TEXT}; letter-spacing:0.5px;">
                AI Cooling Agent
            </div>
            <div style="font-size:0.72rem; color:{MUTED}; margin-top:4px;">
                LangGraph · scikit-learn · Adaptive Control
            </div>
        </div>
        """, unsafe_allow_html=True)

        model_name = st.session_state.get("model_name", "—")
        st.markdown(f"""
        <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:10px;
                    padding:10px 14px; margin-bottom:16px; text-align:center;">
            <div style="font-size:0.68rem; color:{MUTED}; text-transform:uppercase;
                        letter-spacing:1.5px; margin-bottom:4px;">Active Model</div>
            <div style="font-size:0.9rem; font-weight:700; color:{ACCENT};">{model_name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f'<div style="font-size:0.7rem; font-weight:700; color:{MUTED}; '
                    f'text-transform:uppercase; letter-spacing:2px; margin-bottom:10px;">'
                    f'⚙️ Simulation Controls</div>', unsafe_allow_html=True)
        n_steps    = st.slider("Steps", 20, 200, 100, 10,
                               help="Total simulation time steps")
        T_init     = st.slider("Initial Temp (°C)", 15.0, 40.0, 25.0, 0.5,
                               help="Starting battery temperature")
        fixed_flow = st.slider("Fixed Flow (L/min)", 0.5, 10.0, 5.0, 0.5,
                               help="Baseline fixed coolant flow for comparison")
        speed_ms   = st.slider("Step Delay (ms)", 0, 300, 50, 10,
                               help="Slow down to watch simulation step-by-step")

        st.markdown(f"""
        <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:10px;
                    padding:12px 14px; margin:12px 0;">
            <div style="font-size:0.68rem; color:{MUTED}; text-transform:uppercase;
                        letter-spacing:1.5px; margin-bottom:8px;">Safe Operating Range</div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                <span style="color:{CYAN};">🌡 {T_MIN}°C min</span>
                <span style="color:{RED};">🔥 {T_MAX}°C max</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
                <span style="color:{GREEN};">💧 {FLOW_MIN} L/min min</span>
                <span style="color:{ACCENT};">💧 {FLOW_MAX} L/min max</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            start    = st.button("▶ Start",  use_container_width=True, key="btn_start")
        with col2:
            stop_btn = st.button("↺ Reset",  use_container_width=True, key="btn_reset")

        st.markdown(f"""
        <div style="margin-top:24px; padding-top:16px; border-top:1px solid {BORDER};
                    font-size:0.68rem; color:{MUTED}; text-align:center; line-height:1.8;">
            Nanoparticle conc.: <b style="color:{TEXT};">{NANO_PCT}%</b><br>
            Physics: <b style="color:{TEXT};">dT/dt = (Q_heat − Q_cool) / C</b>
        </div>
        """, unsafe_allow_html=True)

    return n_steps, T_init, fixed_flow, speed_ms, start, stop_btn


# ── Main ──────────────────────────────────────────────────────────
def main():
    init_session()
    s = st.session_state

    n_steps, T_init, fixed_flow, speed_ms, start, stop_btn = render_sidebar()

    if start:
        reset_simulation(n_steps, fixed_flow, T_init)
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
        reset_simulation(n_steps, fixed_flow, T_init)

    # KPI state
    step_now    = s.step
    total_steps = s.get("n_steps", n_steps)
    T_now       = s.agent_temps[-1] if s.agent_temps else s.get("T_init", T_init)
    flow_now    = s.agent_flows[-1] if s.agent_flows else 3.0
    hl_now      = heat_load_schedule(float(step_now))
    pct         = (step_now / total_steps * 100) if total_steps > 0 else 0

    if s.running:
        badge = f'<span class="badge badge-run">● Running</span>'
    elif step_now > 0 and not s.running:
        badge = f'<span class="badge badge-idle">✓ Complete</span>'
    else:
        badge = f'<span class="badge badge-idle">Idle</span>'

    t_color = GREEN if T_MIN <= T_now <= T_MAX - 3 else (AMBER if T_now <= T_MAX else RED)
    v_color = GREEN if s.violations_a == 0 else RED

    # ── Header ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:24px 0 12px 0;">
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:4px; flex-wrap:wrap;">
            <h1 style="font-size:2rem; font-weight:900; margin:0;
                       background: linear-gradient(135deg, {ACCENT}, {GREEN}, {CYAN});
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🔋 AI Battery Cooling Agent
            </h1>
            {badge}
        </div>
        <p style="color:{MUTED}; font-size:0.85rem; margin:0;">
            Adaptive thermal management via LangGraph agent — real-time battery safety monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(metric_card("🌡️", f"{T_now:.1f}°C", "Battery Temp",
                                t_color, "Agent-controlled"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("💧", f"{flow_now:.1f}", "Flow L/min",
                                GREEN, f"Range {FLOW_MIN}–{FLOW_MAX}"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("⚡", f"{hl_now:.0f}W", "Heat Load",
                                AMBER if hl_now > 280 else GREEN, "Scheduled"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("⚠️", str(s.violations_a), "AI Violations",
                                v_color, "Steps > 45°C"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card("⏱", f"{step_now}/{total_steps}", "Progress",
                                ACCENT, f"{pct:.0f}% done"), unsafe_allow_html=True)

    if s.running or step_now > 0:
        st.markdown(progress_bar_html(pct), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────
    tab_live, tab_analytics, tab_log = st.tabs(
        ["📡  Live Monitor", "📊  Analytics & Comparison", "🧠  Agent Memory Log"]
    )

    # Tab 1: Live Monitor
    with tab_live:
        if s.agent_temps:
            fig_t = make_temp_chart()
            if fig_t:
                st.pyplot(fig_t, use_container_width=True); plt.close(fig_t)
            fig_f = make_flow_chart()
            if fig_f:
                st.pyplot(fig_f, use_container_width=True); plt.close(fig_f)
            fig_ph = make_phase_heatmap()
            if fig_ph:
                st.pyplot(fig_ph, use_container_width=True); plt.close(fig_ph)
        else:
            st.markdown(f"""
            <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:16px;
                        padding:64px; text-align:center; margin:16px 0;">
                <div style="font-size:3rem; margin-bottom:16px;">🔋</div>
                <div style="font-size:1.1rem; font-weight:700; color:{TEXT}; margin-bottom:8px;">
                    Ready to Simulate
                </div>
                <div style="font-size:0.85rem; color:{MUTED};">
                    Configure parameters in the sidebar and click
                    <b style="color:{ACCENT};">▶ Start</b> to begin.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Tab 2: Analytics
    with tab_analytics:
        if s.agent_temps:
            col_bar, col_sc = st.columns(2)
            with col_bar:
                fig_b = make_violation_bar()
                if fig_b:
                    st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)
            with col_sc:
                fig_sc = make_scatter()
                if fig_sc:
                    st.pyplot(fig_sc, use_container_width=True); plt.close(fig_sc)

            st.markdown("<hr>", unsafe_allow_html=True)
            fig_hl = make_heat_load_chart()
            if fig_hl:
                st.pyplot(fig_hl, use_container_width=True); plt.close(fig_hl)

            # Post-run report
            if not s.running and step_now >= total_steps and total_steps > 0:
                st.markdown(f'<div class="section-header">📋 Simulation Report Card</div>',
                            unsafe_allow_html=True)
                avg_a = np.mean(s.agent_temps)
                avg_n = np.mean(s.none_temps)
                avg_f = np.mean(s.fixed_temps)
                max_a = np.max(s.agent_temps)
                max_n = np.max(s.none_temps)
                max_f = np.max(s.fixed_temps)
                avg_flow = np.mean(s.agent_flows)
                viol_vs_none  = s.violations_n - s.violations_a
                viol_vs_fixed = s.violations_f - s.violations_a

                rc1, rc2, rc3 = st.columns(3)

                def _table(rows):
                    html = "<table style='width:100%; font-size:0.83rem; border-collapse:collapse;'>"
                    for label, val, clr in rows:
                        html += (f"<tr><td style='color:{MUTED}; padding:4px 0;'>{label}</td>"
                                 f"<td style='color:{clr}; font-weight:700; text-align:right;'>{val}</td></tr>")
                    return html + "</table>"

                with rc1:
                    st.markdown(f"""
                    <div class="report-card">
                        <div class="report-title">🤖 AI Agent</div>
                        {_table([("Avg Temp", f"{avg_a:.2f}°C", GREEN),
                                 ("Max Temp", f"{max_a:.2f}°C", GREEN),
                                 ("Violations", str(s.violations_a), GREEN),
                                 ("Avg Flow", f"{avg_flow:.2f} L/min", GREEN)])}
                    </div>""", unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"""
                    <div class="report-card">
                        <div class="report-title">🔥 No Cooling</div>
                        {_table([("Avg Temp", f"{avg_n:.2f}°C", RED),
                                 ("Max Temp", f"{max_n:.2f}°C", RED),
                                 ("Violations", str(s.violations_n), RED),
                                 ("Flow", f"{FLOW_MIN} L/min", MUTED)])}
                    </div>""", unsafe_allow_html=True)
                with rc3:
                    st.markdown(f"""
                    <div class="report-card">
                        <div class="report-title">📊 Fixed ({s.fixed_flow} L/min)</div>
                        {_table([("Avg Temp", f"{avg_f:.2f}°C", AMBER),
                                 ("Max Temp", f"{max_f:.2f}°C", AMBER),
                                 ("Violations", str(s.violations_f), AMBER),
                                 ("Flow", f"{s.fixed_flow} L/min", MUTED)])}
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background: linear-gradient(135deg,
                            rgba(74,222,128,0.08), rgba(124,131,253,0.08));
                            border: 1px solid rgba(74,222,128,0.25);
                            border-radius:16px; padding:20px; margin-top:12px; text-align:center;">
                    <div style="font-size:1rem; font-weight:700; color:{GREEN}; margin-bottom:8px;">
                        🏆 AI Agent Performance Advantage
                    </div>
                    <div style="font-size:0.85rem; color:{TEXT};">
                        Prevented <b style="color:{GREEN};">{viol_vs_none}</b> more violations vs No Cooling
                        &nbsp;·&nbsp;
                        <b style="color:{GREEN};">{viol_vs_fixed}</b> more violations vs Fixed Flow
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run the simulation first to see analytics.", icon="📊")

    # Tab 3: Memory Log
    with tab_log:
        if s.memory:
            st.markdown(f'<div class="section-header">🧠 Agent Decision Memory (last 20 steps)</div>',
                        unsafe_allow_html=True)
            col_l, col_r = st.columns([2, 1])
            with col_r:
                st.markdown(f"""
                <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:12px;
                            padding:14px; font-size:0.78rem; color:{MUTED};">
                    <b style="color:{TEXT};">Legend</b><br><br>
                    🟢 Temperature safe<br>
                    🟡 Within 3°C of limit<br>
                    🔴 Exceeded 45°C limit
                </div>
                """, unsafe_allow_html=True)
            with col_l:
                for entry in reversed(s.memory[-20:]):
                    T_e = entry["T"]
                    if T_e > T_MAX:
                        icon, cls, tc = "🔴", "danger", RED
                    elif T_e > T_MAX - 3:
                        icon, cls, tc = "🟡", "", AMBER
                    else:
                        icon, cls, tc = "🟢", "", GREEN
                    st.markdown(
                        f'<div class="memory-entry {cls}">'
                        f'{icon} &nbsp;<b>t={entry["t"]:3d}s</b>'
                        f'&nbsp;&nbsp; T=<span style="color:{tc};">{T_e:5.1f}°C</span>'
                        f'&nbsp;&nbsp; flow=<span style="color:{ACCENT};">{entry["flow"]:.2f} L/min</span>'
                        f'&nbsp;&nbsp; HL=<span style="color:{AMBER};">{entry["hl"]:.0f}W</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(f"""
            <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:16px;
                        padding:48px; text-align:center;">
                <div style="font-size:2rem; margin-bottom:12px;">🧠</div>
                <div style="color:{MUTED}; font-size:0.9rem;">
                    Agent memory will appear here once the simulation starts.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Simulation loop ───────────────────────────────────────────
    if s.running and step_now < total_steps:
        advance_one_step()
        if speed_ms > 0:
            time.sleep(speed_ms / 1000.0)
        if s.step < total_steps:
            st.rerun()
        else:
            s.running = False
            st.rerun()
    elif step_now >= total_steps and total_steps > 0 and not s.running and s.agent_temps:
        st.success(
            f"✅ Simulation complete!  "
            f"AI Agent violations: **{s.violations_a}**  |  "
            f"No Cooling: **{s.violations_n}**  |  "
            f"Fixed Flow: **{s.violations_f}**"
        )


if __name__ == "__main__":
    main()
