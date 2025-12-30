# run_nav3_live_metrics.py
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from oriented_trimodal_grid import OrientedTriModalGrid, ORIENTS
from ai_agent_factory_nav3 import build_trimodal_nav_agent

# ---------------- small math utils ----------------
def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    z = ex.sum()
    return ex / (z if z > 0 else 1.0)

def kl_div(p, q, eps=1e-16):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0); p /= p.sum()
    q = np.clip(q, eps, 1.0); q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))

def normalize(p, eps=1e-16):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    s = p.sum()
    return p / (s if s > 0 else 1.0)

# ---------------- pymdp compatibility ----------------
def infer_states_compat(agent, obs_triplet):
    """Pass tri-modality observation to pymdp across versions."""
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
    # fallbacks for different pymdp signatures
    try:
        return agent.infer_policies(mode="sophisticated" if sophisticated else "classic")
    except TypeError:
        try:
            return agent.infer_policies(method="sophisticated" if sophisticated else "classic")
        except TypeError:
            return agent.infer_policies()

def get_qs_compat(agent, expected_S: int | None = None):
    """Return a 1-D posterior over the single (pos×ori) factor."""
    def coerce_1d(x):
        try:
            a = np.asarray(x, dtype=np.float64)
            if a.ndim > 1: a = np.squeeze(a)
            if a.ndim == 1 and a.size > 0: return a
        except Exception:
            pass
        return None

    candidates = []
    qs_attr = getattr(agent, "qs", None)
    if qs_attr is not None:
        if isinstance(qs_attr, (list, tuple)):
            for item in qs_attr:
                v = coerce_1d(item);  candidates.append(v) if v is not None else None
        elif isinstance(qs_attr, np.ndarray) and qs_attr.dtype == object:
            for item in qs_attr.ravel():
                v = coerce_1d(item);  candidates.append(v) if v is not None else None
        else:
            v = coerce_1d(qs_attr);   candidates.append(v) if v is not None else None
    if not candidates:
        for name in ("q_s", "qs_current", "qs_prev"):
            v = coerce_1d(getattr(agent, name, None))
            if v is not None: candidates.append(v)
    if not candidates: return None
    if expected_S is not None:
        for v in candidates:
            if v.shape[0] == expected_S: return normalize(v)
    return normalize(max(candidates, key=lambda a: a.shape[0]))

def get_models(agent, model_dict):
    """Extract object-arrays A (len=3), B (len=1), C (len=3), D (len=1) as plain Python lists/arrays."""
    A = model_dict.get("A", getattr(agent, "A", None))
    B = model_dict.get("B", getattr(agent, "B", None))
    C = model_dict.get("C", getattr(agent, "C", None))
    D = model_dict.get("D", getattr(agent, "D", None))

    def to_list(obj):
        if isinstance(obj, (list, tuple)): return list(obj)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            return [obj[i] for i in range(obj.size)]
        return [obj]

    A_list = [np.asarray(a, dtype=np.float64) for a in to_list(A)]
    B_list = [np.asarray(b, dtype=np.float64) for b in to_list(B)]
    C_list = [np.asarray(c, dtype=np.float64) for c in to_list(C)]
    D_list = [np.asarray(d, dtype=np.float64) for d in to_list(D)]
    return A_list, B_list, C_list, D_list

# ---------------- FE / EFE metrics (tri-modality) ----------------
def step_metrics_multi(qs, prior_s, A_list, C_list, obs_triplet):
    """
    Returns per-step: (complexity, accuracy_sum, extrinsic_sum, epistemic_sum)

    - Complexity: KL(q(s) || prior(s))
    - Accuracy: sum_m  -E_q ln p(o_m | s)
    - Extrinsic: sum_m E_{q(o_m)}[-ln p(o_m|C_m)]
    - Epistemic: sum_m E_{q(o_m)} [ KL(q(s|o_m) || q(s)) ]
    """
    qs = normalize(qs)
    prior_s = normalize(prior_s)

    # Complexity
    complexity = kl_div(qs, prior_s)

    accuracy_sum = 0.0
    extrinsic_sum = 0.0
    epistemic_sum = 0.0

    for m, (A_m, C_m, o_m) in enumerate(zip(A_list, C_list, obs_triplet)):
        # Accuracy (likelihood surprise at observed o_m)
        lik_col = np.clip(A_m[int(o_m), :], 1e-16, 1.0)
        accuracy_sum += - float(np.sum(qs * np.log(lik_col)))

        # Predictive over outcomes for modality m
        q_o_m = normalize(A_m @ qs)

        # Extrinsic risk w.r.t. C_m
        pC_m = softmax(C_m)  # convert utilities to probs
        extrinsic_sum += - float(np.sum(q_o_m * np.log(np.clip(pC_m, 1e-16, 1.0))))

        # Epistemic (state info gain) for modality m
        epi_m = 0.0
        for o_idx in range(A_m.shape[0]):
            post_o = normalize(A_m[o_idx, :] * qs)
            epi_m += q_o_m[o_idx] * kl_div(post_o, qs)
        epistemic_sum += float(epi_m)

    return float(complexity), float(accuracy_sum), float(extrinsic_sum), float(epistemic_sum)

# ---------------- rendering (grid + bars) ----------------
def init_figure():
    fig = plt.figure(figsize=(11, 6))
    ax_grid = fig.add_subplot(1, 2, 1)
    ax_grid.set_axis_off()
    ax_bars = fig.add_subplot(1, 2, 2)
    bars = ax_bars.bar(["Complexity", "Accuracy", "Extrinsic", "Epistemic"], [0, 0, 0, 0])
    ax_bars.set_ylim(0, 5)
    ax_bars.set_ylabel("Value")
    ax_bars.set_title("Free Energy Terms (step-by-step)")
    ax_bars.grid(True, axis="y", alpha=0.3)
    plt.ion(); plt.show(block=False)
    fig.tight_layout()
    return fig, ax_grid, ax_bars, bars

def update_grid(ax_grid, frame):
    if not hasattr(update_grid, "_im") or update_grid._im is None:
        update_grid._im = ax_grid.imshow(frame, interpolation="nearest")
    else:
        update_grid._im.set_data(frame)

def update_bars(ax_bars, bars, values):
    vmax = max(1e-6, float(np.max(values)))
    target_ylim = max(1.0, vmax * 1.2)
    _, ymax = ax_bars.get_ylim()
    if target_ylim > ymax or target_ylim < 0.5 * ymax:
        ax_bars.set_ylim(0, target_ylim)
    for b, v in zip(bars, values):
        b.set_height(float(v))

# ---------------- per-episode line plots (optional) ----------------
def plot_episode_metrics(history, ep_idx, note=""):
    t = np.arange(1, len(history["complexity"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(t, history["complexity"], label="Complexity", linewidth=2)
    plt.plot(t, history["accuracy"],   label="Accuracy",   linewidth=2)
    plt.plot(t, history["extrinsic"],  label="Extrinsic",  linewidth=2)
    plt.plot(t, history["epistemic"],  label="Epistemic",  linewidth=2)
    plt.xlabel("Step"); plt.ylabel("Value")
    plt.title(f"Per-step FE/EFE — Episode {ep_idx}{(' — '+note) if note else ''}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.show(block=False)

# ---------------- one live episode ----------------
def run_live_episode_once(env, agent, A_list, B_list, C_list, D_list,
                          fig, ax_grid, ax_bars, bars,
                          fps=12.0, sophisticated=False, controls=None):
    # Single factor => take D[0], B[0]
    D0 = normalize(np.asarray(D_list[0], dtype=np.float64))
    B = np.asarray(B_list[0], dtype=np.float64)   # (S,S,U)

    obs, _ = env.reset()
    if hasattr(agent, "reset") and callable(agent.reset):
        try: agent.reset()
        except Exception: pass

    prior_s = D0.copy()
    total_r, steps, done = 0.0, 0, False

    hist = {"complexity": [], "accuracy": [], "extrinsic": [], "epistemic": []}

    # draw first frame
    frame = env.render()
    update_grid(ax_grid, frame)
    fig.canvas.draw_idle(); fig.canvas.flush_events()

    while not done:
        # (1) update beliefs with tri-modal obs
        infer_states_compat(agent, obs)
        qs = get_qs_compat(agent, expected_S=A_list[0].shape[1])
        if qs is None:
            S = A_list[0].shape[1]
            qs = np.zeros(S);  # degenerate fallback (shouldn't happen)
            # place mass on any state consistent with M3? leaving as zero-guard:
            qs[:] = 1.0 / S

        # (2) compute metrics (sum across modalities)
        comp, acc, extr, epi = step_metrics_multi(qs, prior_s, A_list, C_list, obs)
        hist["complexity"].append(comp)
        hist["accuracy"].append(acc)
        hist["extrinsic"].append(extr)
        hist["epistemic"].append(epi)

        # (3) live UI
        update_bars(ax_bars, bars, [comp, acc, extr, epi])
        ax_bars.set_title(f"FE Terms — step {steps+1} | ori={ORIENTS[env.ori]}")
        fig.canvas.draw_idle(); fig.canvas.flush_events()
        time.sleep(max(1e-3, 1.0 / fps))

        # (4) plan + act
        infer_policies_compat(agent, controls, sophisticated)
        action = sample_action_compat(agent)
        obs, r, terminated, truncated, _ = env.step(action)
        total_r += r; steps += 1
        done = terminated or truncated

        # (5) predictive prior for next step using agent's B
        prior_s = normalize(B[:, :, action] @ qs)

        # (6) render latest grid frame
        frame = env.render()
        update_grid(ax_grid, frame)

    # pack histories
    for k in hist: hist[k] = np.asarray(hist[k], dtype=float)
    return total_r, steps, hist

# ---------------- CLI ----------------
def parse_pos(s: str):
    s = s.strip().lower()
    if s == "random":
        return None
    r, c = s.split(",")
    return (int(r), int(c))

def main():
    ap = argparse.ArgumentParser(description="Live Grid + dynamic FE/EFE bars for tri-modal oriented agent.")
    # env
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
    # agent
    ap.add_argument("--policy-len", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=16.0)
    ap.add_argument("--act-sel", type=str, default="stochastic", choices=["stochastic","deterministic"])
    ap.add_argument("--c-green", type=float, default=3.0)
    ap.add_argument("--c-red", type=float, default=-3.0)
    ap.add_argument("--sophisticated", action="store_true")
    ap.add_argument("--a-noise", type=float, default=0.0, help="A noise in factory (0..1)")
    ap.add_argument("--b-noise", type=float, default=0.0, help="B model noise in factory (0..1)")
    # run/display
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--pause-after-ep", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-episode-curves", action="store_true")
    args = ap.parse_args()

    reward_pos = parse_pos(args.reward_pos)
    punish_pos = parse_pos(args.punish_pos)
    start_pos  = parse_pos(args.start_pos)

    # persistent figure (reused across episodes)
    fig, ax_grid, ax_bars, bars = init_figure()

    # environment (rgb_array for fast blitting)
    env = OrientedTriModalGrid(
        n_rows=args.rows, n_cols=args.cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos, start_ori=args.start_ori,
        step_cost=args.step_cost, reward=args.reward, punish=args.punish,
        max_steps=args.max_steps, render_mode="rgb_array"
    )
    env.reset(seed=args.seed)

    try:
        totals = []
        for ep in range(1, args.episodes + 1):
            # -------- rebuild a FRESH AGENT each episode --------
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
            # unpack model arrays
            A_list, B_list, C_list, D_list = get_models(agent, model)

            # reset bars at episode start
            update_bars(ax_bars, bars, [0, 0, 0, 0])
            ax_bars.set_title(f"FE Terms — episode {ep}/{args.episodes}")
            fig.canvas.draw_idle(); fig.canvas.flush_events()

            total_r, steps, hist = run_live_episode_once(
                env, agent, A_list, B_list, C_list, D_list,
                fig, ax_grid, ax_bars, bars,
                fps=args.fps, sophisticated=args.sophisticated, controls=controls
            )
            totals.append((total_r, steps))
            print(f"[Episode {ep}] return={total_r:.2f}, steps={steps}")

            if args.plot_episode_curves:
                note = f"A-noise={args.a_noise}, B-noise={args.b_noise}, soph={args.sophisticated}"
                plot_episode_metrics(hist, ep, note)

            time.sleep(args.pause_after_ep)

        print("All episodes complete:", totals)

        # keep window open
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()




