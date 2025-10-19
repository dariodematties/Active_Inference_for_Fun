# Active Inference for Fun

Tiny, transparent experiments in **Active Inference** using [pymdp](https://github.com/infer-actively/pymdp?utm_source=chatgpt.com) and [gymnasium](https://gymnasium.farama.org/).

Start from the **simplest possible** environments and agents, then add complexity **step-by-step**—so you can reason about each design choice and its cognitive implications.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Goal: a **parsimonious, scientifically minded** playground to study cognition, building up difficulty “evolutionarily” (fully observable → partially observable; single factor → multi-factor; single modality → multi-modal; deterministic → noisy; etc.).

## Contents

- Minimal `N×M` **GridWorld** (deterministic walls, reward cell, punish cell)
- Text / **graphical renderer** (single persistent window, dynamic updates)
- **Active Inference** agent factory `(A, B, C, D)` tailored to GridWorld
- Batch **experiments & plots** (random vs AIF; sequential & parallel)
- Live demo: watch random episodes then AIF episodes in one window
- Roadmap: POMDP variants, multi-factor control, multi-modal outcomes, learning

## Repository Structure

```
Active_Inference_for_Fun/Environments/
├─ gridworld_env.py                 # Gymnasium env: N×M grid, reward & punish tiles
├─ ai_agent_factory.py              # build_gridworld_agent(): constructs A,B,C,D & Agent
├─ run_gridworld_stats.py           # random baseline: many episodes, plots results
├─ run_gridworld_aif_vs_random.py   # AIF vs Random, sequential or parallel (processes)
├─ run_gridworld_live_demo.py       # live dynamic render: random then AIF episodes
└─ README.md
```

## Installation

Python ≥ 3.9 recommended.

```bash
# (optional) create a fresh environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# core deps
pip install --upgrade pip
pip install numpy matplotlib gymnasium

# pymdp (pick one)
pip install pymdp             # if available in your index
# or from source:
# pip install git+https://github.com/infer-actively/pymdp.git
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If `pymdp` API differs across versions, the code includes small compatibility shims (e.g., scalar vs list observations, `sample_action()` return types).

## Quick Start

1. ### To run a minimal environment smoke test run the following command `python run_gridworld_human.py` ![](run_gridworld_human.mp4)
2. ### 


