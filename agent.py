# -*- coding: utf-8 -*-
"""
AI Cooling Agent — LangGraph Implementation
============================================
Graph topology
──────────────
         ┌──────────────┐
    ┌───▶│  read_state  │
    │    └──────┬───────┘
    │           ▼
    │    ┌──────────────┐
    │    │    predict   │
    │    └──────┬───────┘
    │           ▼
    │    ┌──────────────┐
    │    │    decide    │◀─────────┐
    │    └──────┬───────┘          │
    │           ▼                  │
    │    ┌──────────────────────┐  │
    │    │   safety_check       │──┘  (loop if unsafe)
    │    └──────┬───────────────┘
    │           │ safe
    └───────────┘  (next time step)

Memory
------
The agent stores the last N readings in AgentState and maintains
a rolling summary string for the LLM reasoning node.

LLM
---
Uses the OPENAI_API_KEY env var if present; otherwise falls back to a
deterministic rule-based reasoner (no API key needed for demo).
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import TypedDict, Annotated, List, Optional
import operator

import numpy as np

# ── LangGraph ──────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

# ── Local modules ──────────────────────────────────────────────────
from simulator import BatterySimulator, BatteryConfig, heat_load_schedule
from train_model import load_model, predict_next_temp

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
T_MIN          = 15.0     # °C  lower safe limit
T_MAX          = 45.0     # °C  upper safe limit
FLOW_MIN       = 0.5      # L/min minimum pump speed
FLOW_MAX       = 10.0     # L/min maximum pump speed
FLOW_NOMINAL   = 3.0      # L/min default / rest flow
NANO_PCT       = 1.0      # % nanoparticle concentration (fixed)
MAX_DECIDE_LOOPS = 5      # max re-try loops per time step


# ──────────────────────────────────────────────────────────────────
# Agent State
# ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # ── Simulation context ────────────────────────────────────────
    current_temp:    float
    heat_load:       float
    time_step:       int
    # ── Prediction ────────────────────────────────────────────────
    predicted_temp:  Optional[float]
    # ── Decision ─────────────────────────────────────────────────
    chosen_flow:     float
    reasoning:       str
    # ── Safety ────────────────────────────────────────────────────
    safety_ok:       bool
    decide_loops:    int
    # ── Memory: list of past records (append-only) ─────────────────
    memory:          Annotated[List[dict], operator.add]


# ──────────────────────────────────────────────────────────────────
# LLM / Reasoner
# ──────────────────────────────────────────────────────────────────
def _rule_based_reasoner(state: AgentState) -> tuple[float, str]:
    """
    Deterministic fallback reasoner (no LLM required).
    Returns (recommended_flow_rate, explanation).
    """
    T      = state["current_temp"]
    T_pred = state["predicted_temp"] or T
    HL     = state["heat_load"]
    loops  = state["decide_loops"]

    # Proportional control with a safety bias
    headroom = T_MAX - T_pred       # how much room we have before overheating
    urgency  = 1.0 - headroom / (T_MAX - T_MIN)   # 0 (cool) → 1 (hot)
    urgency  = max(0.0, min(1.0, urgency))

    flow = FLOW_MIN + urgency * (FLOW_MAX - FLOW_MIN)

    # If we're looping due to a safety failure, be more aggressive
    if loops > 0:
        flow = min(FLOW_MAX, flow * (1.0 + 0.3 * loops))

    # Clamp
    flow = max(FLOW_MIN, min(FLOW_MAX, flow))

    reason = (
        f"T_current={T:.1f}°C  T_predicted={T_pred:.1f}°C  "
        f"heat_load={HL:.0f}W  urgency={urgency:.2f} → "
        f"flow_rate={flow:.2f} L/min"
    )
    return round(flow, 2), reason


def _llm_reasoner(state: AgentState) -> tuple[float, str]:
    """
    Uses OpenAI chat completion to decide flow rate.
    Returns (recommended_flow_rate, explanation).
    Falls back to rule-based if the API is unavailable.
    """
    try:
        from openai import OpenAI
        client = OpenAI()   # reads OPENAI_API_KEY from env

        history_lines = "\n".join(
            f"  t={r['time']:.0f}s: T={r['temperature']:.1f}°C  flow={r['flow_rate']:.2f} L/min"
            for r in state["memory"][-5:]
        ) or "  (no history yet)"

        prompt = textwrap.dedent(f"""
            You are an AI battery thermal management agent.
            
            Current status:
              - Battery temperature  : {state['current_temp']:.2f}°C
              - Predicted temperature: {state['predicted_temp']:.2f}°C
              - Heat load            : {state['heat_load']:.1f} W
              - Time step            : {state['time_step']}
              - Safe range           : {T_MIN}–{T_MAX}°C
              - Flow rate limits     : {FLOW_MIN}–{FLOW_MAX} L/min
            
            Recent history (last 5 readings):
            {history_lines}
            
            Decide the optimal coolant flow rate (between {FLOW_MIN} and {FLOW_MAX} L/min).
            Reply with JSON only:
            {{"flow_rate": <number>, "reason": "<one sentence>"}}
        """)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=150,
        )
        import json
        data = json.loads(response.choices[0].message.content)
        flow = float(data.get("flow_rate", FLOW_NOMINAL))
        flow = max(FLOW_MIN, min(FLOW_MAX, flow))
        return round(flow, 2), data.get("reason", "LLM decision")

    except Exception as exc:
        print(f"[agent] LLM unavailable ({exc}); using rule-based fallback.")
        return _rule_based_reasoner(state)


USE_LLM = bool(os.getenv("OPENAI_API_KEY"))
reasoner = _llm_reasoner if USE_LLM else _rule_based_reasoner


# ──────────────────────────────────────────────────────────────────
# Node functions
# ──────────────────────────────────────────────────────────────────

# We inject model & simulator via closure when building the graph.

def make_read_node(sim: BatterySimulator, heat_fn):
    """Returns the 'read_state' node bound to a simulator instance."""
    def read_state(state: AgentState) -> dict:
        t   = state["time_step"]
        HL  = heat_fn(float(t))
        rec = sim.step(HL, state["chosen_flow"], NANO_PCT)
        T   = rec["temperature"]
        mem_entry = {
            "time":        rec["time"],
            "temperature": T,
            "heat_load":   HL,
            "flow_rate":   state["chosen_flow"],
        }
        print(f"  [read]    t={t:3d}  T={T:6.2f}°C  HL={HL:5.0f}W  flow={state['chosen_flow']:.2f}")
        return {
            "current_temp": T,
            "heat_load":    HL,
            "time_step":    t + 1,
            "memory":       [mem_entry],
            "decide_loops": 0,
            "safety_ok":    False,
        }
    return read_state


def make_predict_node(model):
    """Returns the 'predict' node bound to an sklearn model."""
    def predict(state: AgentState) -> dict:
        T_pred = predict_next_temp(
            model,
            T_current  = state["current_temp"],
            heat_load  = state["heat_load"],
            flow_rate  = state["chosen_flow"],
            nano_pct   = NANO_PCT,
        )
        print(f"  [predict] T_next_est={T_pred:.2f}°C")
        return {"predicted_temp": T_pred}
    return predict


def decide(state: AgentState) -> dict:
    """Choose coolant flow rate using LLM or rule-based reasoner."""
    flow, reason = reasoner(state)
    print(f"  [decide]  flow={flow:.2f} L/min  | {reason}")
    return {
        "chosen_flow": flow,
        "reasoning":   reason,
        "decide_loops": state["decide_loops"] + 1,
    }


def make_predict_with_new_flow(model):
    """Re-predict using the newly chosen flow before safety check."""
    def predict_new(state: AgentState) -> dict:
        T_pred = predict_next_temp(
            model,
            T_current  = state["current_temp"],
            heat_load  = state["heat_load"],
            flow_rate  = state["chosen_flow"],
            nano_pct   = NANO_PCT,
        )
        return {"predicted_temp": T_pred}
    return predict_new


def safety_check(state: AgentState) -> dict:
    """
    Validate that:
      1. chosen_flow is within [FLOW_MIN, FLOW_MAX]
      2. predicted temperature stays within [T_MIN, T_MAX]
    If not, mark unsafe so the graph loops back to 'decide'.
    """
    flow   = state["chosen_flow"]
    T_pred = state["predicted_temp"] or state["current_temp"]

    flow_ok   = FLOW_MIN <= flow <= FLOW_MAX
    temp_ok   = T_MIN <= T_pred <= T_MAX
    safe      = flow_ok and temp_ok

    if not safe:
        reason = []
        if not flow_ok:  reason.append(f"flow {flow:.2f} out of [{FLOW_MIN},{FLOW_MAX}]")
        if not temp_ok:  reason.append(f"T_pred {T_pred:.2f}°C out of [{T_MIN},{T_MAX}]°C")
        print(f"  [safety]  ⚠️  UNSAFE – {' | '.join(reason)}")
    else:
        print(f"  [safety]  ✅ SAFE   (T_pred={T_pred:.2f}°C  flow={flow:.2f})")

    return {"safety_ok": safe}


# ──────────────────────────────────────────────────────────────────
# Routing edges
# ──────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """After safety_check: loop back to decide if unsafe (up to limit)."""
    if state["safety_ok"]:
        return "read_state"           # advance to next time step
    if state["decide_loops"] >= MAX_DECIDE_LOOPS:
        print(f"  [routing] Max loops reached; forcing safe flow.")
        return "read_state"
    return "decide"


# ──────────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────────

def build_agent(sim: BatterySimulator, model, heat_fn=heat_load_schedule) -> object:
    g = StateGraph(AgentState)

    read_node     = make_read_node(sim, heat_fn)
    predict_node  = make_predict_node(model)
    predict_new   = make_predict_with_new_flow(model)

    g.add_node("read_state",    read_node)
    g.add_node("predict",       predict_node)
    g.add_node("decide",        decide)
    g.add_node("predict_new",   predict_new)
    g.add_node("safety_check",  safety_check)

    g.set_entry_point("read_state")
    g.add_edge("read_state",  "predict")
    g.add_edge("predict",     "decide")
    g.add_edge("decide",      "predict_new")
    g.add_edge("predict_new", "safety_check")

    g.add_conditional_edges(
        "safety_check",
        should_continue,
        {
            "read_state": "read_state",
            "decide":     "decide",
        },
    )

    return g.compile()


# ──────────────────────────────────────────────────────────────────
# Public runner
# ──────────────────────────────────────────────────────────────────

def run_agent(n_steps: int = 100, T_init: float = 25.0, verbose: bool = True) -> list[dict]:
    """
    Run the LangGraph agent for n_steps time steps.
    Returns list of per-step records.
    """
    model, model_name = load_model()
    cfg = BatteryConfig()
    sim = BatterySimulator(T_init=T_init, config=cfg, seed=99)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  AI COOLING AGENT  |  model: {model_name}  |  LLM: {USE_LLM}")
        print(f"  Steps: {n_steps}  T_init: {T_init}°C  Safe: [{T_MIN},{T_MAX}]°C")
        print(f"{'='*60}\n")

    app = build_agent(sim, model, heat_load_schedule)

    state: AgentState = {
        "current_temp":   T_init,
        "heat_load":      heat_load_schedule(0.0),
        "time_step":      0,
        "predicted_temp": None,
        "chosen_flow":    FLOW_NOMINAL,
        "reasoning":      "",
        "safety_ok":      False,
        "decide_loops":   0,
        "memory":         [],
    }

    # Collect per-step results from simulator history
    step_count = 0
    for event in app.stream(state, {"recursion_limit": n_steps * (MAX_DECIDE_LOOPS + 3) + 10}):
        node_name = list(event.keys())[0]
        if node_name == "read_state":
            step_count += 1
        if step_count >= n_steps:
            break

    return sim.get_dataframe().to_dict("records")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    records = run_agent(n_steps=20, verbose=True)
    print(f"\nFinal temperature: {records[-1]['temperature']:.2f}°C")
