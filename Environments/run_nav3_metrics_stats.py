# run_nav3_metrics_stats.py
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from oriented_trimodal_grid import OrientedTriModalGrid
from ai_agent_factory_nav3 import build_trimodal_nav_agent
from typing import Tuple, Dict, Any, List

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

def moving_average(x, w):
    if w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[w:] - c[:-w]) / float(w)
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, ma])

def smart_bins(x, max_bins=40):
    """Safe histogram bin edges (handles constant arrays and caps bin count)."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1
    xmin, xmax = np.min(x), np.max(x)
    if np.isclose(xmin, xmax):
        width = 1.0 if xmax == 0 else abs(xmax) * 0.1
        if width == 0:
            width = 1.0
        return np.array([xmax - width, xmax + width])
    edges = np.histogram_bin_edges(x, bins='auto')
    if edges.size - 1 > max_bins:
        edges = np.histogram_bin_edges(x, bins=max_bins)
    return edges

# ---------------- pymdp compatibility ----------------
def infer_states_compat(agent, obs_triplet):
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
    try:
        return agent.infer_policies(mode="sophisticated" if sophisticated else "classic")
    except TypeError:
        try:
            return agent.infer_policies(method="sophisticated" if sophisticated else "classic")
        except TypeError:
            return agent.infer_policies()

def get_qs_compat(agent, expected_S: int | None = None):
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
    """Extract object-arrays A (len=3), B (len=1), C (len=3), D (len=1) as plain lists/arrays."""
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
    Returns: (complexity, accuracy_sum, extrinsic_sum, epistemic_sum)
    - Complexity: KL(q(s) || prior(s))
    - Accuracy: sum_m  -E_q ln p(o_m | s)
    - Extrinsic: sum_m E_{q(o_m)}[-ln p(o_m|C_m)]
    - Epistemic: sum_m E_{q(o_m)} [ KL(q(s|o_m) || q(s)) ]
    """
    qs = normalize(qs); prior_s = normalize(prior_s)
    complexity = kl_div(qs, prior_s)

    accuracy_sum = 0.0
    extrinsic_sum = 0.0
    epistemic_sum = 0.0

    for A_m, C_m, o_m in zip(A_list, C_list, obs_triplet):
        # Accuracy (surprise at observed outcome)
        lik_col = np.clip(A_m[int(o_m), :], 1e-16, 1.0)
        accuracy_sum += - float(np.sum(qs * np.log(lik_col)))

        # Predictive outcomes for modality m
        q_o_m = normalize(A_m @ qs)

        # Extrinsic risk
        pC_m = softmax(C_m)
        extrinsic_sum += - float(np.sum(q_o_m * np.log(np.clip(pC_m, 1e-16, 1.0))))

        # Epistemic (state info gain)
        epi_m = 0.0
        for o_idx in range(A_m.shape[0]):
            post_o = normalize(A_m[o_idx, :] * qs)
            epi_m += q_o_m[o_idx] * kl_div(post_o, qs)
        epistemic_sum += float(epi_m)

    return float(complexity), float(accuracy_sum), float(extrinsic_sum), float(epistemic_sum)

# ---------------- single episode runner ----------------
def run_episode_once(env, agent, A_list, B_list, C_list, D_list, sophisticated: bool, controls):
    D0 = normalize(np.asarray(D_list[0], dtype=np.float64))
    B = np.asarray(B_list[0], dtype=np.float64)  # (S,S,U)

    obs, _ = env.reset()
    if hasattr(agent, "reset") and callable(agent.reset):
        try: agent.reset()
        except Exception: pass

    prior_s = D0.copy()
    total_r, steps, done = 0.0, 0, False

    # per-step metric histories
    hist_c, hist_a, hist_x, hist_e = [], [], [], []

    while not done:
        # infer states
        infer_states_compat(agent, obs)
        qs = get_qs_compat(agent, expected_S=A_list[0].shape[1])
        if qs is None:
            S = A_list[0].shape[1]
            qs = np.full(S, 1.0 / S, dtype=np.float64)

        # metrics
        comp, acc, extr, epi = step_metrics_multi(qs, prior_s, A_list, C_list, obs)
        hist_c.append(comp); hist_a.append(acc); hist_x.append(extr); hist_e.append(epi)

        # act
        infer_policies_compat(agent, controls, sophisticated)
        action = sample_action_compat(agent)
        obs, r, terminated, truncated, _ = env.step(action)
        total_r += r; steps += 1
        done = terminated or truncated

        # predictive prior
        prior_s = normalize(B[:, :, action] @ qs)

    # outcome label
    outcome = "timeout"
    if done:
        if env.pos == env.reward_pos:
            outcome = "reward"
        elif env.pos == env.punish_pos:
            outcome = "punish"

    # per-episode means
    mC = float(np.mean(hist_c)) if hist_c else 0.0
    mA = float(np.mean(hist_a)) if hist_a else 0.0
    mX = float(np.mean(hist_x)) if hist_x else 0.0
    mE = float(np.mean(hist_e)) if hist_e else 0.0

    return total_r, steps, outcome, mC, mA, mX, mE

# ---------------- worker ----------------
def _worker_chunk(
    worker_seed: int,
    episodes: int,
    env_kwargs: Dict[str, Any],
    agent_kwargs: Dict[str, Any],
    sophisticated: bool
):
    rng = np.random.default_rng(worker_seed)

    env = OrientedTriModalGrid(**env_kwargs, render_mode=None)
    # results
    returns = np.zeros(episodes, dtype=np.float64)
    steps   = np.zeros(episodes, dtype=np.int32)
    outcomes: List[str] = []
    mC = np.zeros(episodes, dtype=np.float64)
    mA = np.zeros(episodes, dtype=np.float64)
    mX = np.zeros(episodes, dtype=np.float64)
    mE = np.zeros(episodes, dtype=np.float64)

    for ep in range(episodes):
        # Re-seed env each episode for diversity in 'random' starts
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))

        # Fresh agent each episode
        agent, model, controls = build_trimodal_nav_agent(**agent_kwargs, sophisticated=sophisticated)
        A_list, B_list, C_list, D_list = get_models(agent, model)

        R, S, outcome, mc, ma, mx, me = run_episode_once(
            env, agent, A_list, B_list, C_list, D_list,
            sophisticated=sophisticated, controls=controls
        )
        returns[ep] = R
        steps[ep]   = S
        outcomes.append(outcome)
        mC[ep], mA[ep], mX[ep], mE[ep] = mc, ma, mx, me

    env.close()
    return returns, steps, outcomes, mC, mA, mX, mE

# ---------------- CLI / main ----------------
def parse_pos(s: str):
    s = s.strip().lower()
    if s == "random":
        return None
    r, c = s.split(",")
    return (int(r), int(c))

def main():
    ap = argparse.ArgumentParser(description="Metrics & stats for Tri-Modal Navigation Agent (parallel).")
    # env
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--reward-pos", type=str, default="4,6")
    ap.add_argument("--punish-pos", type=str, default="0,6")
    ap.add_argument("--start-pos", type=str, default="0,0", help="'random' for random starts")
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
    ap.add_argument("--a-noise", type=float, default=0.0, help="A noise in factory (0..1)")
    ap.add_argument("--b-noise", type=float, default=0.0, help="B model noise in factory (0..1)")
    ap.add_argument("--sophisticated", action="store_true")
    # run
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ma-window", type=int, default=50)
    ap.add_argument("--max-bins", type=int, default=40)
    ap.add_argument("--savefig", type=str, default="")
    args = ap.parse_args()

    reward_pos = parse_pos(args.reward_pos)
    punish_pos = parse_pos(args.punish_pos)
    start_pos  = parse_pos(args.start_pos)

    env_kwargs = dict(
        n_rows=args.rows, n_cols=args.cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos, start_ori=args.start_ori,
        step_cost=args.step_cost, reward=args.reward, punish=args.punish,
        max_steps=args.max_steps
    )
    agent_kwargs = dict(
        n_rows=args.rows, n_cols=args.cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos, start_ori=args.start_ori,
        a_obs_noise=args.a_noise, b_model_noise=args.b_noise,
        policy_len=args.policy_len, gamma=args.gamma,
        action_selection=args.act_sel,
        c_green=args.c_green, c_red=args.c_red,
    )

    # -------------- run episodes (maybe parallel) --------------
    if args.workers <= 1:
        results = _worker_chunk(
            worker_seed=args.seed,
            episodes=args.episodes,
            env_kwargs=env_kwargs,
            agent_kwargs=agent_kwargs,
            sophisticated=args.sophisticated
        )
        returns, steps, outcomes, mC, mA, mX, mE = results
    else:
        base = args.episodes // args.workers
        rem  = args.episodes %  args.workers
        sizes = [base + (1 if i < rem else 0) for i in range(args.workers)]
        seeds = [int(args.seed + i * 1000003) for i in range(args.workers)]

        returns = np.zeros(args.episodes, dtype=np.float64)
        steps   = np.zeros(args.episodes, dtype=np.int32)
        outcomes_all: List[str] = []
        mC = np.zeros(args.episodes, dtype=np.float64)
        mA = np.zeros(args.episodes, dtype=np.float64)
        mX = np.zeros(args.episodes, dtype=np.float64)
        mE = np.zeros(args.episodes, dtype=np.float64)

        idx = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = []
            for n_ep, s in zip(sizes, seeds):
                futures.append((idx, n_ep, ex.submit(
                    _worker_chunk, s, n_ep, env_kwargs, agent_kwargs, args.sophisticated
                )))
                idx += n_ep
            for start, n_ep, fut in futures:
                rR, sS, oO, cC, aA, xX, eE = fut.result()
                end = start + n_ep
                returns[start:end] = rR
                steps[start:end]   = sS
                mC[start:end]      = cC
                mA[start:end]      = aA
                mX[start:end]      = xX
                mE[start:end]      = eE
                outcomes_all.extend(oO)
        outcomes = outcomes_all

    # -------------- summarize --------------
    def summarize(returns, steps, outcomes):
        cnt = Counter(outcomes)
        n = max(1, len(outcomes))
        return {
            "success_rate": cnt.get("reward", 0) / n,
            "punish_rate":  cnt.get("punish", 0) / n,
            "timeout_rate": cnt.get("timeout", 0) / n,
            "avg_return":   float(np.mean(returns)) if len(returns) else 0.0,
            "avg_steps":    float(np.mean(steps)) if len(steps) else 0.0,
            "counts":       cnt,
        }

    summary = summarize(returns, steps, outcomes)
    print("\n=== Summary ===")
    for k, v in summary.items():
        if k != "counts": print(f"{k:>13}: {v}")
    print("counts:", dict(summary["counts"]))
    print()
    print(f"Per-episode means (overall): Complexity={np.mean(mC):.3f}, Accuracy={np.mean(mA):.3f}, Extrinsic={np.mean(mX):.3f}, Epistemic={np.mean(mE):.3f}")

    # -------------- plots --------------
    fig = plt.figure(figsize=(12, 10))

    # 1) Returns with MA
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(returns, alpha=0.4, lw=0.8, label="Return")
    ax1.plot(moving_average(returns, args.ma_window), lw=2, label=f"MA({args.ma_window})")
    ax1.set_title("Episode Returns")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Return"); ax1.grid(True, alpha=0.3); ax1.legend()

    # 2) Steps MA
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(moving_average(steps.astype(float), args.ma_window), lw=2, label="Steps (MA)")
    ax2.set_title("Steps per Episode (MA)")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Steps"); ax2.grid(True, alpha=0.3); ax2.legend()

    # 3) Outcomes bars
    ax3 = plt.subplot(3, 2, 3)
    labels = ["reward", "punish", "timeout"]
    counts = np.array([summary["counts"].get(k, 0) for k in labels])
    x = np.arange(len(labels))
    ax3.bar(x, counts, width=0.6)
    ax3.set_xticks(x); ax3.set_xticklabels(labels)
    ax3.set_title("Outcome Counts"); ax3.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax3.text(i, v, str(int(v)), ha="center", va="bottom")

    # 4) Complexity mean/ep
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(mC, bins=smart_bins(mC, max_bins=args.max_bins), alpha=0.8)
    ax4.set_title("Complexity (mean/ep)")
    ax4.set_xlabel("Value"); ax4.set_ylabel("Episodes")

    # 5) Accuracy mean/ep
    ax5 = plt.subplot(3, 2, 5)
    ax5.hist(mA, bins=smart_bins(mA, max_bins=args.max_bins), alpha=0.8)
    ax5.set_title("Accuracy (mean/ep)")
    ax5.set_xlabel("Value"); ax5.set_ylabel("Episodes")

    # 6) Extrinsic & Epistemic mean/ep
    ax6 = plt.subplot(3, 2, 6)
    ax6.hist(mX, bins=smart_bins(mX, max_bins=args.max_bins), alpha=0.6, label="Extrinsic")
    ax6.hist(mE, bins=smart_bins(mE, max_bins=args.max_bins), alpha=0.6, label="Epistemic")
    ax6.set_title("Extrinsic vs Epistemic (mean/ep)")
    ax6.set_xlabel("Value"); ax6.set_ylabel("Episodes"); ax6.legend()

    fig.suptitle(
        f"Nav3 Stats — {args.rows}x{args.cols}, episodes={args.episodes}, workers={args.workers}\n"
        f"A-noise={args.a_noise}, B-noise={args.b_noise}, sophisticated={args.sophisticated}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])

    if args.savefig:
        plt.savefig(args.savefig, dpi=160)
        print(f"Saved figure to: {args.savefig}")

    plt.show()

if __name__ == "__main__":
    main()
