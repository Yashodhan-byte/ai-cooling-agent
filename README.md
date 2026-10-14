# 🔋 AI Battery Cooling Agent

> **Syllabus concepts covered:** Agent planning · Reasoning & memory · LangGraph · scikit-learn · Safety checks · Responsible AI

---

## Project Overview

A fully software-only simulation of an AI thermal management agent for a battery pack. No hardware or CFD data needed — the physics model is embedded in Python.

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent                          │
│                                                             │
│  ┌──────────┐   ┌─────────┐   ┌────────┐   ┌──────────┐   │
│  │read_state│──▶│ predict │──▶│ decide │──▶│  safety  │   │
│  │(simulator│   │ (RF     │   │(LLM or │   │  check   │   │
│  │  step)   │   │  model) │   │ rules) │   │  node    │   │
│  └──────────┘   └─────────┘   └────────┘   └────┬─────┘   │
│       ▲                                          │         │
│       └─────────── safe ─────────────────────────┘         │
│                 (unsafe → loop back to decide)              │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
ai_cooling_agent/
├── simulator.py       ← Step 1: Battery thermal simulator (physics + noise)
├── train_model.py     ← Step 2: ML model training (RF, GB, Linear, Ridge)
├── agent.py           ← Step 3: LangGraph agent with 4 nodes + memory
├── demo.py            ← Step 4: 100-step demo + comparison plots
├── app.py             ← Bonus: Live Streamlit dashboard
├── requirements.txt
├── models/            ← Auto-created: saved model + training data cache
└── plots/             ← Auto-created: saved PNG plots
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the static demo (generates comparison plots)
```bash
python demo.py
```
Output: `plots/demo_comparison.png`

### 3. Run the interactive Streamlit dashboard
```bash
streamlit run app.py
```

### 4. (Optional) Use OpenAI LLM for the decide node
```bash
set OPENAI_API_KEY=sk-...
python demo.py
```

### 5. Retrain the model from scratch
```bash
python train_model.py --retrain --samples 8000
```

---

## Step-by-Step Implementation

### Step 1 — Simulator (`simulator.py`)

Physics model (discrete time, dt = 1 s):

```
dT = (Q_heat - Q_cool) / C_thermal * dt + noise

Q_heat  = heat_load  (electrical dissipation, W)
Q_cool  = c_fluid × ρ × flow_rate × (T − T_ambient)
C_thermal boosted by nanoparticle %
```

Heat-load profile: charging ramp → hold → discharge taper → rest.

### Step 2 — Prediction Model (`train_model.py`)

| Model | Features | Target |
|-------|----------|--------|
| Random Forest ✅ | heat\_load, flow\_rate, nano\_pct, T\_current | T\_next |
| Gradient Boosting | same | same |
| Linear Regression | same | same |
| Ridge Regression | same | same |

Best model selected by lowest MAE on 20 % hold-out test set.

### Step 3 — LangGraph Agent (`agent.py`)

**Nodes:**

| Node | Role |
|------|------|
| `read_state` | Advance simulator one tick; record to memory |
| `predict` | ML model estimates T at next step |
| `decide` | LLM (or rule-based fallback) picks flow rate |
| `predict_new` | Re-predict with chosen flow |
| `safety_check` | Validates flow ∈ [0.5, 10] L/min & T\_pred ∈ [15, 45]°C |

**Routing:** `safety_check` → loops to `decide` if unsafe (max 5 retries), then advances.

**Memory:** rolling list of past readings appended to `AgentState` (LangGraph's `Annotated[List, operator.add]` pattern).

### Step 4 — Demo (`demo.py`)

Three scenarios run for 100 time steps:

| Scenario | Flow Rate |
|----------|-----------|
| No Cooling | 0.5 L/min (min pump) |
| Fixed Cooling | 5.0 L/min constant |
| **AI Agent** | Adaptive 0.5–10 L/min |

Output: 5-panel dark-themed matplotlib figure showing temperature comparison, heat load profile, agent flow rate, violations bar chart, and statistics table.

---

## Safety & Responsible AI

The `safety_check` node enforces hard constraints:
- Flow rate must stay within hardware limits `[0.5, 10]` L/min
- Predicted temperature must stay within safe band `[15, 45]`°C
- If violated, the agent loops (up to 5 times) before forcing maximum safe action

This demonstrates **agent safety patterns** — the agent cannot apply a decision until it passes a constraint check.

---

## Extending the Project

- **Real CFD data:** Replace `generate_training_data()` with your own CSV loader in `train_model.py`
- **Real LLM:** Set `OPENAI_API_KEY` for GPT-4o-mini reasoning
- **Hugging Face:** Swap OpenAI client for `transformers` pipeline in `agent.py`
- **More nodes:** Add an `alert` node that sends an email when T > 42°C
- **Real sensors:** Replace `sim.step()` in `read_state` with an MQTT/serial read
