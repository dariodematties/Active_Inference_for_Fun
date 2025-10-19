# run_gridworld_stats.py
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from gridworld_env import GridWorld  # ensure gridworld_env.py is importable

def run_episode(env: GridWorld, rng: np.random.Generator):
    """Run one episode with a random policy. Returns a dict of episode stats."""
    obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    total_r = 0.0
    steps = 0
    outcome = "timeout"  # default unless we terminate on reward/punish
    while True:
        action = int(rng.integers(0, env.action_space.n))
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r
        steps += 1
        if terminated:
            # Determine whether we hit reward or punishment
            if env.pos == env.reward_pos:
                outcome = "reward"
            elif env.pos == env.punish_pos:
                outcome = "punish"
            else:
                outcome = "terminal"
            break
        if truncated:
            outcome = "timeout"
            break
    return {
        "return": total_r,
        "steps": steps,
        "outcome": outcome,
    }

def moving_average(x, w):
    if w <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[w:] - cumsum[:-w]) / float(w)
    # pad to original length for nicer plotting
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, ma])

def main():
    parser = argparse.ArgumentParser(description="Run GridWorld stats (no rendering) and plot results.")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--reward-pos", type=str, default="4,6", help="row,col of reward")
    parser.add_argument("--punish-pos", type=str, default="0,6", help="row,col of punish")
    parser.add_argument("--start-pos", type=str, default="0,0", help="row,col start (or 'random' for random starts)")
    parser.add_argument("--step-cost", type=float, default=0.0)
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--punish", type=float, default=-1.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--slip-prob", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ma-window", type=int, default=50, help="moving-average window for plots")
    parser.add_argument("--savefig", type=str, default="", help="optional path to save the figure (PNG)")
    args = parser.parse_args()

    def parse_pos(s):
        if s.strip().lower() == "random":
            return None
        r, c = s.split(",")
        return (int(r), int(c))

    reward_pos = parse_pos(args.reward_pos)
    punish_pos = parse_pos(args.punish_pos)
    start_pos = parse_pos(args.start_pos)

    rng = np.random.default_rng(args.seed)

    env = GridWorld(
        n_rows=args.rows,
        n_cols=args.cols,
        reward_pos=reward_pos,
        punish_pos=punish_pos,
        start_pos=start_pos,
        step_cost=args.step_cost,
        reward=args.reward,
        punish=args.punish,
        max_steps=args.max_steps,
        slip_prob=args.slip_prob,
        render_mode=None,   # <-- IMPORTANT: no rendering
    )

    returns = np.zeros(args.episodes, dtype=float)
    steps = np.zeros(args.episodes, dtype=int)
    outcomes = []

    for ep in range(args.episodes):
        stats = run_episode(env, rng)
        returns[ep] = stats["return"]
        steps[ep] = stats["steps"]
        outcomes.append(stats["outcome"])

    # Aggregate
    outcome_counts = Counter(outcomes)
    success_rate = outcome_counts.get("reward", 0) / args.episodes
    punish_rate = outcome_counts.get("punish", 0) / args.episodes
    timeout_rate = outcome_counts.get("timeout", 0) / args.episodes
    avg_return = np.mean(returns)
    avg_steps = np.mean(steps)

    print(f"Episodes:        {args.episodes}")
    print(f"Success rate:    {success_rate:.3f}")
    print(f"Punish rate:     {punish_rate:.3f}")
    print(f"Timeout rate:    {timeout_rate:.3f}")
    print(f"Avg return:      {avg_return:.3f}")
    print(f"Avg steps:       {avg_steps:.2f}")

    # ---- Plotting ----
    fig = plt.figure(figsize=(12, 7))

    # 1) Returns with moving average
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(returns, lw=0.8, alpha=0.5)
    ma = moving_average(returns, args.ma_window)
    ax1.plot(ma, lw=2)
    ax1.set_title(f"Episode Returns (MA window={args.ma_window})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.grid(True, alpha=0.3)

    # 2) Steps per episode
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(steps, lw=0.8, alpha=0.7)
    ax2.set_title("Steps per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.grid(True, alpha=0.3)

    # 3) Outcome distribution (bar)
    ax3 = plt.subplot(2, 2, 3)
    labels = ["reward", "punish", "timeout"]
    counts = [outcome_counts.get(k, 0) for k in labels]
    ax3.bar(labels, counts)
    ax3.set_title("Episode Outcomes")
    ax3.set_ylabel("Count")
    for i, c in enumerate(counts):
        ax3.text(i, c, str(c), ha="center", va="bottom")

    # 4) Histogram of returns
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(returns, bins=30, alpha=0.9)
    ax4.set_title("Return Distribution")
    ax4.set_xlabel("Return")
    ax4.set_ylabel("Frequency")

    fig.suptitle(
        f"GridWorld Stats — {args.rows}x{args.cols}, "
        f"reward={args.reward}, punish={args.punish}, step_cost={args.step_cost}, "
        f"max_steps={args.max_steps}, slip_prob={args.slip_prob}\n"
        f"Success={success_rate:.3f}, Punish={punish_rate:.3f}, Timeout={timeout_rate:.3f}, "
        f"AvgReturn={avg_return:.3f}, AvgSteps={avg_steps:.2f}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    if args.savefig:
        plt.savefig(args.savefig, dpi=150)
        print(f"Saved figure to: {args.savefig}")

    plt.show()

if __name__ == "__main__":
    main()

