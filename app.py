"""
Streamlit Dashboard — AI Battery Cooling Agent
===============================================
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

# ──────────────────────────────────────────────────────────────────
# Imports after page config
# ──────────────────────────────────────────────────────────────────
from simulator import BatterySimulator, BatteryConfig, heat_load_schedule
from train_model import load_model, predict_next_temp

# ──────────────────────────────────────────────────────────────────
# Custom CSS — premium dark theme
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e8eaf6;
}

.metric-card {
    background: linear-gradient(135deg, #1a1d27 0%, #22253a 100%);
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.8rem; color: #9095b0; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }

.status-safe   { color: #69f0ae; }
.status-warn   { color: #ffb300; }
.status-danger { color: #ff5252; }

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #7c83fd;
    border-bottom: 1px solid #2a2d3a;
    padding-bottom: 6px;
    margin-bottom: 16px;
}

.stButton > button {
    background: linear-gradient(135deg, #7c83fd, #5c6cfa);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(124,131,253,0.4);
}

.memory-entry {
    background: #1a1d27;
    border-left: 3px solid #7c83fd;
    padding: 6px 12px;
    margin: 4px 0;
    border-radius: 4px;
    font-size: 0.82rem;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

T_MIN = 15.0
T_MAX = 45.0
FLOW_MIN = 0.5
FLOW_MAX = 10.0
NANO_PCT = 1.0


# ──────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────
def init_session():
    if "model" not in st.session_state:
        with st.spinner("🤖 Loading / training prediction model…"):
            model, name = load_model()
            st.session_state.model = model
            st.session_state.model_name = name

    defaults = dict(
        sim_agent  = None,
        sim_none   = None,
        sim_fixed  = None,
        running    = False,
        step       = 0,
        flow_agent = FLOW_MIN,
        memory     = [],
        agent_temps  = [],
        none_temps   = [],
        fixed_temps  = [],
        agent_flows  = [],
        times        = [],
        violations_a = 0,
        violations_n = 0,
        violations_f = 0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_simulation(n_steps, fixed_flow, T_init):
    cfg = BatteryConfig()
    st.session_state.sim_agent = BatterySimulator(T_init=T_init, config=cfg, seed=99)
    st.session_state.sim_none  = BatterySimulator(T_init=T_init, config=cfg, seed=99)
    st.session_state.sim_fixed = BatterySimulator(T_init=T_init, config=cfg, seed=99)
    st.session_state.step        = 0
    st.session_state.flow_agent  = 3.0
    st.session_state.memory      = []
    st.session_state.agent_temps = []
    st.session_state.none_temps  = []
    st.session_state.fixed_temps = []
    st.session_state.agent_flows = []
    st.session_state.times       = []
    st.session_state.violations_a = 0
    st.session_state.violations_n = 0
    st.session_state.violations_f = 0
    st.session_state.n_steps      = n_steps
    st.session_state.fixed_flow   = fixed_flow


# ──────────────────────────────────────────────────────────────────
# Agent decision logic (inline, no LangGraph overhead for live UI)
# ──────────────────────────────────────────────────────────────────
def agent_decide(T_current, heat_load, current_flow, model, memory):
    T_pred  = predict_next_temp(model, T_current, heat_load, current_flow, NANO_PCT)
    headroom = T_MAX - T_pred
    urgency  = 1.0 - headroom / (T_MAX - T_MIN)
    urgency  = max(0.0, min(1.0, urgency))
    flow_new = FLOW_MIN + urgency * (FLOW_MAX - FLOW_MIN)

    # safety correction
    T_pred2 = predict_next_temp(model, T_current, heat_load, flow_new, NANO_PCT)
    if T_pred2 > T_MAX:
        flow_new = FLOW_MAX

    flow_new = round(max(FLOW_MIN, min(FLOW_MAX, flow_new)), 2)
    return flow_new, T_pred


def advance_one_step():
    s    = st.session_state
    t    = s.step + 1
    hl   = heat_load_schedule(float(t))

    # Agent simulation
    rec_a = s.sim_agent.step(hl, s.flow_agent, NANO_PCT)
    T_a   = rec_a["temperature"]
    flow_new, T_pred = agent_decide(T_a, hl, s.flow_agent, s.model, s.memory)
    s.flow_agent = flow_new

    # Baselines
    rec_n = s.sim_none.step(hl,  FLOW_MIN,       NANO_PCT)
    rec_f = s.sim_fixed.step(hl, s.fixed_flow, NANO_PCT)

    # Record
    s.times.append(t)
    s.agent_temps.append(T_a)
    s.none_temps.append(rec_n["temperature"])
    s.fixed_temps.append(rec_f["temperature"])
    s.agent_flows.append(flow_new)
    s.memory.append({"t": t, "T": T_a, "flow": flow_new, "hl": hl})
    if len(s.memory) > 20:
        s.memory = s.memory[-20:]

    if T_a > T_MAX:   s.violations_a += 1
    if rec_n["temperature"] > T_MAX: s.violations_n += 1
    if rec_f["temperature"] > T_MAX: s.violations_f += 1

    s.step += 1


# ──────────────────────────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────────────────────────
PANEL_BG = "#1a1d27"
GRID_CLR = "#2a2d3a"
TEXT_CLR = "#e8eaf6"
C_NONE   = "#ff5252"
C_FIXED  = "#ffb300"
C_AGENT  = "#69f0ae"
ACCENT   = "#7c83fd"


def make_temp_chart():
    s = st.session_state
    if not s.times:
        return None

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f1117")
    ax.set_facecolor(PANEL_BG)
    ax.fill_between(s.times, T_MIN, T_MAX, color=ACCENT, alpha=0.1)
    ax.axhline(T_MAX, color=ACCENT, ls="--", lw=0.8, alpha=0.5)
    ax.axhline(T_MIN, color=ACCENT, ls="--", lw=0.8, alpha=0.5)

    ax.plot(s.times, s.none_temps,  color=C_NONE,  lw=1.8, label="No Cooling")
    ax.plot(s.times, s.fixed_temps, color=C_FIXED, lw=1.8, label=f"Fixed ({s.fixed_flow} L/min)")
    ax.plot(s.times, s.agent_temps, color=C_AGENT, lw=2.3, label="AI Agent")

    ax.set_xlabel("Time (s)", color=TEXT_CLR, fontsize=9)
    ax.set_ylabel("Temperature (°C)", color=TEXT_CLR, fontsize=9)
    ax.set_title("Battery Temperature Comparison", color=TEXT_CLR, fontsize=11, fontweight="bold")
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.grid(color=GRID_CLR, lw=0.6)
    for sp in ax.spines.values(): sp.set_color(GRID_CLR)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8)

    fig.tight_layout()
    return fig


def make_flow_chart():
    s = st.session_state
    if not s.times:
        return None

    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0f1117")
    ax.set_facecolor(PANEL_BG)
    ax.fill_between(s.times, s.agent_flows, FLOW_MIN, alpha=0.2, color=C_AGENT)
    ax.plot(s.times, s.agent_flows, color=C_AGENT, lw=2, label="Agent flow rate")
    ax.axhline(s.fixed_flow, color=C_FIXED, ls="--", lw=1.2, alpha=0.7, label=f"Fixed ({s.fixed_flow} L/min)")

    ax.set_xlabel("Time (s)", color=TEXT_CLR, fontsize=9)
    ax.set_ylabel("Flow Rate (L/min)", color=TEXT_CLR, fontsize=9)
    ax.set_title("AI Agent — Adaptive Coolant Flow", color=TEXT_CLR, fontsize=11, fontweight="bold")
    ax.set_ylim(0, FLOW_MAX + 1)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.grid(color=GRID_CLR, lw=0.6)
    for sp in ax.spines.values(): sp.set_color(GRID_CLR)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────
# Main app layout
# ──────────────────────────────────────────────────────────────────
def main():
    init_session()

    # ── Header ────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <h1 style="font-size:2.2rem; font-weight:800; 
                   background: linear-gradient(135deg, #7c83fd, #69f0ae);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🔋 AI Battery Cooling Agent
        </h1>
        <p style="color:#9095b0; font-size:0.95rem; margin-top:-10px;">
            LangGraph • scikit-learn • Adaptive Thermal Control
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar — controls ─────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Simulation Controls</div>', unsafe_allow_html=True)
        n_steps    = st.slider("Number of steps",    20, 200, 100, 10)
        T_init     = st.slider("Initial temperature (°C)", 15.0, 40.0, 25.0, 0.5)
        fixed_flow = st.slider("Fixed flow rate (L/min)",  0.5, 10.0, 5.0, 0.5)
        speed_ms   = st.slider("Step delay (ms)", 0, 200, 50, 10)

        st.markdown("---")
        st.markdown('<div class="section-header">📊 Model Info</div>', unsafe_allow_html=True)
        st.info(f"**Model:** {st.session_state.model_name}")
        st.info(f"**Safe range:** {T_MIN}–{T_MAX}°C")
        st.info(f"**Flow range:** {FLOW_MIN}–{FLOW_MAX} L/min")

        col1, col2 = st.columns(2)
        with col1:
            start = st.button("▶ Start", use_container_width=True)
        with col2:
            stop_btn = st.button("⏹ Reset", use_container_width=True)

    if start:
        reset_simulation(n_steps, fixed_flow, T_init)
        st.session_state.running = True

    if stop_btn:
        st.session_state.running = False
        reset_simulation(n_steps, fixed_flow, T_init)

    # ── Main dashboard ────────────────────────────────────────────
    s = st.session_state

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    step_now = s.step
    T_now    = s.agent_temps[-1]  if s.agent_temps  else T_init
    flow_now = s.agent_flows[-1]  if s.agent_flows  else 3.0
    hl_now   = heat_load_schedule(float(step_now))

    t_class = "status-safe" if T_MIN <= T_now <= T_MAX else "status-danger"

    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value {t_class}">{T_now:.1f}°C</div>
            <div class="metric-label">Battery Temp</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#69f0ae;">{flow_now:.1f}</div>
            <div class="metric-label">Flow (L/min)</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ff7043;">{hl_now:.0f}W</div>
            <div class="metric-label">Heat Load</div></div>""", unsafe_allow_html=True)
    with c4:
        v_cls = "status-safe" if s.violations_a == 0 else "status-danger"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value {v_cls}">{s.violations_a}</div>
            <div class="metric-label">Agent Violations</div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#9095b0;">{step_now}/{s.get('n_steps', n_steps)}</div>
            <div class="metric-label">Step</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    chart_temp = st.empty()
    chart_flow = st.empty()

    # Memory log
    col_l, col_r = st.columns([2, 1])
    with col_r:
        st.markdown('<div class="section-header">🧠 Agent Memory (last 10)</div>', unsafe_allow_html=True)
        memory_box = st.empty()

    with col_l:
        st.markdown('<div class="section-header">📈 Violation Comparison</div>', unsafe_allow_html=True)
        bar_box = st.empty()

    # ── Simulation loop ───────────────────────────────────────────
    if s.running and step_now < s.get("n_steps", n_steps):
        advance_one_step()

        fig_t = make_temp_chart()
        if fig_t: chart_temp.pyplot(fig_t, use_container_width=True); plt.close(fig_t)

        fig_f = make_flow_chart()
        if fig_f: chart_flow.pyplot(fig_f, use_container_width=True); plt.close(fig_f)

        # Memory log
        with memory_box:
            for entry in reversed(s.memory[-10:]):
                T_c   = "🟢" if T_MIN <= entry["T"] <= T_MAX else "🔴"
                st.markdown(
                    f'<div class="memory-entry">'
                    f'{T_c} t={entry["t"]:3d}s  T={entry["T"]:5.1f}°C  '
                    f'flow={entry["flow"]:.2f} L/min  HL={entry["hl"]:.0f}W'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Bar chart violations
        with bar_box:
            fig_b, ax_b = plt.subplots(figsize=(5, 3), facecolor="#0f1117")
            ax_b.set_facecolor(PANEL_BG)
            cats = ["No Cooling", f"Fixed\n{s.fixed_flow} L/min", "AI Agent"]
            vals = [s.violations_n, s.violations_f, s.violations_a]
            bars = ax_b.bar(cats, vals, color=[C_NONE, C_FIXED, C_AGENT], edgecolor=GRID_CLR)
            for bar, v in zip(bars, vals):
                ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                          str(v), ha="center", color=TEXT_CLR, fontsize=11, fontweight="bold")
            ax_b.set_ylabel("Steps > 45°C", color=TEXT_CLR, fontsize=8)
            ax_b.tick_params(colors=TEXT_CLR, labelsize=7)
            ax_b.grid(axis="y", color=GRID_CLR, lw=0.6)
            for sp in ax_b.spines.values(): sp.set_color(GRID_CLR)
            fig_b.tight_layout()
            bar_box.pyplot(fig_b, use_container_width=True)
            plt.close(fig_b)

        if speed_ms > 0:
            time.sleep(speed_ms / 1000.0)

        if s.step < s.get("n_steps", n_steps):
            st.rerun()
        else:
            s.running = False
            st.success(f"✅ Simulation complete! AI Agent violations: {s.violations_a}")
    else:
        # Static display if not running
        fig_t = make_temp_chart()
        if fig_t: chart_temp.pyplot(fig_t, use_container_width=True); plt.close(fig_t)
        fig_f = make_flow_chart()
        if fig_f: chart_flow.pyplot(fig_f, use_container_width=True); plt.close(fig_f)

        if not s.agent_temps:
            st.info("👆 Configure the parameters in the sidebar and click **▶ Start** to begin the simulation.")


if __name__ == "__main__":
    main()
