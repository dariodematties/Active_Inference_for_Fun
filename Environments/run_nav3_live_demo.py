# run_nav3_live_demo.py
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from oriented_trimodal_grid import OrientedTriModalGrid, ORIENTS
from ai_agent_factory_nav3 import build_trimodal_nav_agent

# -------------- compat helpers ----------------
def infer_states_compat(agent, obs_triplet):
    """Pass 3-modality observation to pymdp agent (handles API variants)."""
    try:
        agent.infer_states(list(obs_triplet))
    except Exception:
        agent.infer_states([obs_triplet[0], obs_triplet[1], obs_triplet[2]])

def sample_action_compat(agent) -> int:
    a = agent.sample_action()
    return int(a[0] if isinstance(a, (list, tuple, np.ndarray)) else a)

def infer_policies_compat(agent, controls, sophisticated: bool):
    if sophisticated and controls and callable(controls.get("infer_policies", None)):
        return controls["infer_policies"]()
    # fallback
    try:
        return agent.infer_policies(mode="sophisticated" if sophisticated else "classic")
    except TypeError:
        try:
            return agent.infer_policies(method="sophisticated" if sophisticated else "classic")
        except TypeError:
            return agent.infer_policies()

# -------------- action decoding ----------------
ACTIONS = ["forward", "turn_left", "turn_right"]

# -------------- live figure ----------------
def init_figure():
    fig = plt.figure(figsize=(10, 6))
    ax_grid = fig.add_subplot(1, 2, 1)
    ax_grid.set_axis_off()
    ax_hud  = fig.add_subplot(1, 2, 2)
    ax_hud.set_axis_off()
    txt = ax_hud.text(
        0.02, 0.98,
        "HUD",
        va="top", ha="left", fontsize=12, family="monospace"
    )
    plt.ion(); plt.show(block=False)
    return fig, ax_grid, ax_hud, txt

def update_grid(ax_grid, frame):
    if not hasattr(update_grid, "_im") or update_grid._im is None:
        update_grid._im = ax_grid.imshow(frame, interpolation="nearest")
    else:
        update_grid._im.set_data(frame)

def update_hud(txt_obj, step, cum_return, orient_idx, obs, last_action):
    m1, m2, m3 = obs
    # map M2 class ids to strings
    m2_map = {0: "EDGE", 1: "RED", 2: "GREEN"}
    m3_map = {0: "EMPTY", 1: "EDGE", 2: "RED", 3: "GREEN"}
    content = (
        f"Step         : {step}\n"
        f"Return (Σr)  : {cum_return:.3f}\n"
        f"Orientation  : {ORIENTS[orient_idx]}\n"
        f"Obs M1 dist  : {m1}\n"
        f"Obs M2 class : {m2_map.get(int(m2), str(m2))}\n"
        f"Obs M3 here  : {m3_map.get(int(m3), str(m3))}\n"
        f"Last action  : {last_action}"
    )
    txt_obj.set_text(content)

# -------------- main episode loop ----------------
def run_live_episode(env, agent, controls, fps: float, sophisticated: bool, hud_txt, ax_grid):
    obs, _ = env.reset()
    total_r, steps = 0.0, 0
    done = False
    last_action_name = "—"
    # Draw first frame
    frame = env.render()  # rgb_array
    update_grid(ax_grid, frame)
    update_hud(hud_txt, steps, total_r, env.ori, obs, last_action_name)
    plt.gcf().canvas.draw_idle(); plt.gcf().canvas.flush_events()

    while not done:
        # 1) infer states from tri-modal obs
        infer_states_compat(agent, obs)
        # 2) plan policies (sophisticated if available)
        infer_policies_compat(agent, controls, sophisticated)
        # 3) sample and step
        action = sample_action_compat(agent)
        obs, r, terminated, truncated, _ = env.step(action)
        total_r += r; steps += 1
        done = terminated or truncated
        last_action_name = ACTIONS[action]

        # 4) render + HUD
        frame = env.render()
        update_grid(ax_grid, frame)
        update_hud(hud_txt, steps, total_r, env.ori, obs, last_action_name)
        plt.gcf().canvas.draw_idle(); plt.gcf().canvas.flush_events()
        time.sleep(max(1e-3, 1.0 / fps))

    return total_r, steps

# -------------- CLI ----------------
def parse_pos(s: str):
    s = s.strip().lower()
    if s == "random":
        return None
    r, c = s.split(",")
    return (int(r), int(c))

def main():
    ap = argparse.ArgumentParser(description="Live demo for the tri-modal oriented agent.")
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--reward-pos", type=str, default="4,6")
    ap.add_argument("--punish-pos", type=str, default="0,6")
    ap.add_argument("--start-pos", type=str, default="0,0")
    ap.add_argument("--start-ori", type=str, default="E", choices=["N","E","S","W"])
    ap.add_argument("--step-cost", type=float, default=0.0)
    ap.add_argument("--reward", type=float, default=1.0)
    ap.add_argument("--punish", type=float, default=-1.0)
    ap.add_argument("--max-steps", type=int, default=200)

    # agent settings
    ap.add_argument("--policy-len", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=16.0)
    ap.add_argument("--act-sel", type=str, default="stochastic", choices=["stochastic","deterministic"])
    ap.add_argument("--c-green", type=float, default=3.0)
    ap.add_argument("--c-red", type=float, default=-3.0)
    ap.add_argument("--sophisticated", action="store_true")

    # model noise (inside agent)
    ap.add_argument("--a-noise", type=float, default=0.0, help="obs noise for A in factory (0..1)")
    ap.add_argument("--b-noise", type=float, default=0.0, help="dynamics model noise for B in factory (0..1)")

    # display
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reward_pos = parse_pos(args.reward_pos)
    punish_pos = parse_pos(args.punish_pos)
    start_pos  = parse_pos(args.start_pos)

    # --- Environment (rgb_array for fast blitting into our figure) ---
    env = OrientedTriModalGrid(
        n_rows=args.rows, n_cols=args.cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos, start_ori=args.start_ori,
        step_cost=args.step_cost, reward=args.reward, punish=args.punish,
        max_steps=args.max_steps, render_mode="rgb_array"
    )
    env.reset(seed=args.seed)

    # --- Agent (build with noise in-factory) ---
    agent, model, controls = build_trimodal_nav_agent(
        n_rows=args.rows, n_cols=args.cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos, start_ori=args.start_ori,
        a_obs_noise=args.a_noise, b_model_noise=args.b_noise,
        policy_len=args.policy_len, gamma=args.gamma,
        action_selection=args.act_sel,
        c_green=args.c_green, c_red=args.c_red,
        sophisticated=args.sophisticated,
    )

    # --- Figure ---
    fig, ax_grid, ax_hud, hud_txt = init_figure()

    try:
        for ep in range(1, args.episodes + 1):
            # reset agent beliefs between episodes if available
            if hasattr(agent, "reset") and callable(agent.reset):
                try: agent.reset()
                except Exception: pass

            total_r, steps = run_live_episode(
                env, agent, controls, args.fps, args.sophisticated, hud_txt, ax_grid
            )
            print(f"[Episode {ep}] return={total_r:.2f}, steps={steps}")
            time.sleep(0.4)
    finally:
        # keep window open until user closes it
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)
        env.close()

if __name__ == "__main__":
    main()



