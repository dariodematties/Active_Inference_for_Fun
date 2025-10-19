# run_gridworld_aif_vs_random.py
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from gridworld_env import GridWorld
from ai_agent_factory import build_gridworld_agent
from concurrent.futures import ProcessPoolExecutor




def _worker_run_chunk(
    seed: int,
    episodes: int,
    rows: int, cols: int,
    reward_pos: tuple[int,int], punish_pos: tuple[int,int], start_pos: tuple[int,int] | None,
    step_cost: float, reward: float, punish: float, max_steps: int, slip_prob: float,
    policy_len: int, gamma: float, act_sel: str, c_reward: float, c_punish: float,
):
    """Runs a chunk of episodes for BOTH Random and AIF in an isolated process."""
    import numpy as np
    from gridworld_env import GridWorld
    from ai_agent_factory import build_gridworld_agent

    rng = np.random.default_rng(seed)

    env = GridWorld(
        n_rows=rows, n_cols=cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos,
        step_cost=step_cost, reward=reward, punish=punish,
        max_steps=max_steps, slip_prob=slip_prob,
        render_mode=None
    )
    agent, _ = build_gridworld_agent(
        n_rows=rows, n_cols=cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos,
        c_reward=c_reward, c_punish=c_punish,
        policy_len=policy_len, gamma=gamma, action_selection=act_sel,
    )

    # Local copies of the single-episode runners to avoid cross-proc imports
    def infer_states_compat(agent, obs):
        try: agent.infer_states(obs)
        except Exception: agent.infer_states([obs])
    def sample_action_compat(agent) -> int:
        a = agent.sample_action()
        return int(a[0] if isinstance(a, (list, tuple, np.ndarray)) else a)
    def reset_agent_compat(agent):
        if hasattr(agent, "reset") and callable(agent.reset):
            try: agent.reset()
            except Exception: pass

    def run_episode_random():
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total_r, steps, outcome = 0.0, 0, "timeout"
        while True:
            action = int(rng.integers(0, env.action_space.n))
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r; steps += 1
            if terminated:
                outcome = "reward" if env.pos == env.reward_pos else ("punish" if env.pos == env.punish_pos else "terminal")
                break
            if truncated: break
        return total_r, steps, outcome

    def run_episode_aif():
        reset_agent_compat(agent)
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total_r, steps, outcome = 0.0, 0, "timeout"
        while True:
            infer_states_compat(agent, obs)
            agent.infer_policies()
            action = sample_action_compat(agent)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r; steps += 1
            if terminated:
                outcome = "reward" if env.pos == env.reward_pos else ("punish" if env.pos == env.punish_pos else "terminal")
                break
            if truncated: break
        return total_r, steps, outcome

    returns_rand = np.zeros(episodes); steps_rand = np.zeros(episodes, dtype=int); outcomes_rand = []
    returns_aif  = np.zeros(episodes); steps_aif  = np.zeros(episodes, dtype=int); outcomes_aif  = []

    for i in range(episodes):
        tr, st, oc = run_episode_random()
        returns_rand[i] = tr
        steps_rand[i] = st
        outcomes_rand.append(oc)

        tr, st, oc = run_episode_aif()
        returns_aif[i] = tr
        steps_aif[i] = st
        outcomes_aif.append(oc)

    # Return plain Python/NumPy data (pickle-friendly)
    return (
        returns_rand, steps_rand, outcomes_rand,
        returns_aif,  steps_aif,  outcomes_aif,
    )





# ---------- Compatibility helpers for different pymdp versions ----------
def infer_states_compat(agent, obs):
    """Call agent.infer_states with scalar or [scalar], depending on API."""
    try:
        agent.infer_states(obs)      # some versions accept scalar
    except Exception:
        agent.infer_states([obs])    # others expect list per modality

def sample_action_compat(agent) -> int:
    """Return an int action for 1 control factor."""
    a = agent.sample_action()
    # normalize to int
    if isinstance(a, (list, tuple, np.ndarray)):
        return int(a[0])
    return int(a)

def reset_agent_compat(agent):
    """Soft-reset beliefs if available."""
    if hasattr(agent, "reset") and callable(agent.reset):
        try:
            agent.reset()
        except Exception:
            pass


# ---------- Episode runners ----------
def run_episode_random(env: GridWorld, rng: np.random.Generator):
    obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    total_r, steps = 0.0, 0
    outcome = "timeout"
    while True:
        action = int(rng.integers(0, env.action_space.n))
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r
        steps += 1
        if terminated:
            outcome = "reward" if env.pos == env.reward_pos else ("punish" if env.pos == env.punish_pos else "terminal")
            break
        if truncated:
            outcome = "timeout"
            break
    return {"return": total_r, "steps": steps, "outcome": outcome}


def run_episode_aif(env: GridWorld, agent, rng: np.random.Generator):
    # optional per-episode reset of agent beliefs
    reset_agent_compat(agent)

    obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    total_r, steps = 0.0, 0
    outcome = "timeout"
    while True:
        # (1) update beliefs given last observation
        infer_states_compat(agent, obs)
        # (2) evaluate policies
        agent.infer_policies()
        # (3) sample an action
        action = sample_action_compat(agent)

        obs, r, terminated, truncated, info = env.step(action)
        total_r += r
        steps += 1
        if terminated:
            outcome = "reward" if env.pos == env.reward_pos else ("punish" if env.pos == env.punish_pos else "terminal")
            break
        if truncated:
            outcome = "timeout"
            break
    return {"return": total_r, "steps": steps, "outcome": outcome}


# ---------- Utilities ----------
def moving_average(x, w):
    if w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[w:] - c[:-w]) / float(w)
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, ma])


# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(description="Active Inference vs Random in GridWorld (no rendering)")
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--reward-pos", type=str, default="4,6")
    p.add_argument("--punish-pos", type=str, default="0,6")
    p.add_argument("--start-pos", type=str, default="0,0", help="'random' for random starts")
    p.add_argument("--step-cost", type=float, default=0.0)
    p.add_argument("--reward", type=float, default=1.0)
    p.add_argument("--punish", type=float, default=-1.0)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--slip-prob", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ma-window", type=int, default=50)
    # Agent hyperparams
    p.add_argument("--policy-len", type=int, default=4)
    p.add_argument("--gamma", type=float, default=16.0)
    p.add_argument("--act-sel", type=str, default="stochastic", choices=["stochastic", "deterministic"])
    p.add_argument("--c-reward", type=float, default=3.0)
    p.add_argument("--c-punish", type=float, default=-3.0)
    p.add_argument("--savefig", type=str, default="")
    p.add_argument("--workers", type=int, default=1, help="process workers; 1 = no parallelism")
    args = p.parse_args()

    def parse_pos(s):
        if s.strip().lower() == "random":
            return None
        r, c = s.split(",")
        return (int(r), int(c))

    reward_pos = parse_pos(args.reward_pos)
    punish_pos = parse_pos(args.punish_pos)
    start_pos = parse_pos(args.start_pos)

    # after parsing & building env/agent args, replace the sequential loop with:
    if args.workers <= 1:
        # keep your existing sequential loop here (unchanged)

        rng = np.random.default_rng(args.seed)

        # Env (no rendering)
        env = GridWorld(
            n_rows=args.rows, n_cols=args.cols,
            reward_pos=reward_pos, punish_pos=punish_pos,
            start_pos=start_pos,
            step_cost=args.step_cost, reward=args.reward, punish=args.punish,
            max_steps=args.max_steps, slip_prob=args.slip_prob,
            render_mode=None
        )

        # Agent
        agent, model = build_gridworld_agent(
            n_rows=args.rows, n_cols=args.cols,
            reward_pos=reward_pos, punish_pos=punish_pos,
            start_pos=start_pos,
            c_reward=args.c_reward, c_punish=args.c_punish,
            policy_len=args.policy_len, gamma=args.gamma,
            action_selection=args.act_sel,
        )

        # Run experiments
        returns_rand = np.zeros(args.episodes)
        steps_rand   = np.zeros(args.episodes, dtype=int)
        outcomes_rand = []

        returns_aif = np.zeros(args.episodes)
        steps_aif   = np.zeros(args.episodes, dtype=int)
        outcomes_aif = []

        for ep in range(args.episodes):
            s_r = run_episode_random(env, rng)
            returns_rand[ep], steps_rand[ep], outcomes_rand_ep = s_r["return"], s_r["steps"], s_r["outcome"]
            outcomes_rand.append(outcomes_rand_ep)

            s_a = run_episode_aif(env, agent, rng)
            returns_aif[ep], steps_aif[ep], outcomes_aif_ep = s_a["return"], s_a["steps"], s_a["outcome"]
            outcomes_aif.append(outcomes_aif_ep)
    else:
        # --- Parallel path ---
        # split episodes roughly equally across workers
        chunks = []
        base = args.episodes // args.workers
        rem  = args.episodes %  args.workers
        sizes = [base + (1 if i < rem else 0) for i in range(args.workers)]
        seeds = [int(args.seed + i * 1000003) for i in range(args.workers)]  # spaced seeds

        returns_rand = np.zeros(args.episodes)
        steps_rand   = np.zeros(args.episodes, dtype=int)
        outcomes_rand = []
        returns_aif  = np.zeros(args.episodes)
        steps_aif    = np.zeros(args.episodes, dtype=int)
        outcomes_aif = []

        # gather into contiguous blocks for simple assignment
        idx = 0
        futures = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for n_eps, s in zip(sizes, seeds):
                fut = ex.submit(
                    _worker_run_chunk,
                    s, n_eps,
                    args.rows, args.cols,
                    reward_pos, punish_pos, start_pos,
                    args.step_cost, args.reward, args.punish, args.max_steps, args.slip_prob,
                    args.policy_len, args.gamma, args.act_sel, args.c_reward, args.c_punish,
                )
                futures.append((idx, n_eps, fut))
                idx += n_eps

            # collect
            for start, n_eps, fut in futures:
                (rR, sR, oR, rA, sA, oA) = fut.result()
                end = start + n_eps
                returns_rand[start:end] = rR
                steps_rand[start:end]   = sR
                returns_aif[start:end]  = rA
                steps_aif[start:end]    = sA
                outcomes_rand.extend(oR)
                outcomes_aif.extend(oA)

    # Aggregate
    def summarize(returns, steps, outcomes):
        cnt = Counter(outcomes)
        return {
            "success_rate": cnt.get("reward", 0) / len(outcomes),
            "punish_rate":  cnt.get("punish", 0) / len(outcomes),
            "timeout_rate": cnt.get("timeout", 0) / len(outcomes),
            "avg_return":   float(np.mean(returns)),
            "avg_steps":    float(np.mean(steps)),
            "counts":       cnt,
        }

    sum_r = summarize(returns_rand, steps_rand, outcomes_rand)
    sum_a = summarize(returns_aif, steps_aif, outcomes_aif)

    print("\n=== Summary (Random) ===")
    for k, v in sum_r.items():
        if k != "counts": print(f"{k:>13}: {v}")
    print("counts:", dict(sum_r["counts"]))

    print("\n=== Summary (AIF) ===")
    for k, v in sum_a.items():
        if k != "counts": print(f"{k:>13}: {v}")
    print("counts:", dict(sum_a["counts"]))
    print()

    # ---------- Plots ----------
    fig = plt.figure(figsize=(12, 8))

    # 1) Returns with MA
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(returns_rand, alpha=0.35, lw=0.8, label="Random")
    ax1.plot(returns_aif,  alpha=0.35, lw=0.8, label="AIF")
    ax1.plot(moving_average(returns_rand, args.ma_window), lw=2, label=f"Random (MA={args.ma_window})")
    ax1.plot(moving_average(returns_aif,  args.ma_window), lw=2, label=f"AIF (MA={args.ma_window})")
    ax1.set_title("Episode Returns")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Return"); ax1.grid(True, alpha=0.3); ax1.legend()

    # 2) Steps per episode (MA for readability)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(moving_average(steps_rand.astype(float), args.ma_window), lw=2, label="Random (MA)")
    ax2.plot(moving_average(steps_aif.astype(float),  args.ma_window), lw=2, label="AIF (MA)")
    ax2.set_title("Steps per Episode (MA)")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Steps"); ax2.grid(True, alpha=0.3); ax2.legend()

    # 3) Outcomes (side-by-side bars)
    ax3 = plt.subplot(2, 2, 3)
    labels = ["reward", "punish", "timeout"]
    r_counts = np.array([sum_r["counts"].get(k, 0) for k in labels])
    a_counts = np.array([sum_a["counts"].get(k, 0) for k in labels])
    x = np.arange(len(labels))
    w = 0.4
    ax3.bar(x - w/2, r_counts, width=w, label="Random")
    ax3.bar(x + w/2, a_counts, width=w, label="AIF")
    ax3.set_xticks(x); ax3.set_xticklabels(labels)
    ax3.set_title("Outcome Counts")
    ax3.set_ylabel("Count")
    for i, (rv, av) in enumerate(zip(r_counts, a_counts)):
        ax3.text(i - w/2, rv, str(int(rv)), ha="center", va="bottom")
        ax3.text(i + w/2, av, str(int(av)), ha="center", va="bottom")
    ax3.legend()

    # 4) Return histograms
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(returns_rand, bins=30, alpha=0.6, label="Random")
    ax4.hist(returns_aif,  bins=30, alpha=0.6, label="AIF")
    ax4.set_title("Return Distribution"); ax4.set_xlabel("Return"); ax4.set_ylabel("Frequency")
    ax4.legend()

    fig.suptitle(
        f"GridWorld {args.rows}x{args.cols} — Episodes={args.episodes}, slip={args.slip_prob}\n"
        f"Random: success={sum_r['success_rate']:.3f}, avgR={sum_r['avg_return']:.3f}, avgSteps={sum_r['avg_steps']:.1f} | "
        f"AIF: success={sum_a['success_rate']:.3f}, avgR={sum_a['avg_return']:.3f}, avgSteps={sum_a['avg_steps']:.1f}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])

    if args.savefig:
        plt.savefig(args.savefig, dpi=150)
        print(f"Saved figure to: {args.savefig}")

    plt.show()


if __name__ == "__main__":
    main()
