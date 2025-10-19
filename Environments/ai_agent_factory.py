# ai_agent_factory.py
import numpy as np
from pymdp.agent import Agent

def build_gridworld_agent(
    n_rows: int,
    n_cols: int,
    reward_pos: tuple[int, int],
    punish_pos: tuple[int, int],
    start_pos: tuple[int, int] | None = None,
    c_reward: float = 3.0,
    c_punish: float = -3.0,
    policy_len: int = 4,
    gamma: float = 16.0,
    action_selection: str = "stochastic",
):
    """
    Minimal Active Inference agent for an N×M fully-observable GridWorld.
    Returns:
        agent: pymdp.agent.Agent instance
        model: dict with A, B, C, D and small helpers
    """
    S = n_rows * n_cols                     # hidden states (positions)
    O = S                                   # observations (positions)
    U = 4                                   # actions: 0=up,1=right,2=down,3=left

    def pos_to_idx(r: int, c: int) -> int:
        return r * n_cols + c

    def clip_move(r: int, c: int, u: int) -> tuple[int, int]:
        if u == 0: r = max(0, r - 1)                 # up
        elif u == 1: c = min(n_cols - 1, c + 1)      # right
        elif u == 2: r = min(n_rows - 1, r + 1)      # down
        elif u == 3: c = max(0, c - 1)               # left
        return r, c

    # --- A: observation model p(o|s) (O×S) — fully observable → identity ---
    A = np.eye(O, S, dtype=np.float64)

    # --- B: transition model p(s'|s,u) (S×S×U), column-stochastic per (s,u) ---
    B = np.zeros((S, S, U), dtype=np.float64)
    for s_prev in range(S):
        r, c = divmod(s_prev, n_cols)
        for u in range(U):
            r2, c2 = clip_move(r, c, u)
            s_next = pos_to_idx(r2, c2)
            B[s_next, s_prev, u] = 1.0
    # sanity: columns should sum to 1 for each action
    # B.sum(axis=0) -> shape (S, U), all ones

    # --- C: prior preferences over outcomes (length O) ---
    C = np.zeros(O, dtype=np.float64)
    reward_idx = pos_to_idx(*reward_pos)
    punish_idx = pos_to_idx(*punish_pos)
    C[reward_idx] = c_reward
    C[punish_idx] = c_punish
    # (Interpreted as utilities or log-preferences depending on pymdp version.)

    # --- D: prior over initial hidden state (length S) ---
    if start_pos is not None:
        start_idx = pos_to_idx(*start_pos)
        D = np.zeros(S, dtype=np.float64); D[start_idx] = 1.0
    else:
        D = np.ones(S, dtype=np.float64)
        D[[reward_idx, punish_idx]] = 0.0
        D /= D.sum()

    # --- Build Agent (handle slight API differences across pymdp versions) ---
    try:
        # newer style with kwargs and action_selection/gamma
        agent = Agent(
            A=A, B=B, C=C, D=D,
            policy_len=policy_len,
            gamma=gamma,
            action_selection=action_selection,
        )
    except TypeError:
        # older style: positional args and fewer kwargs
        agent = Agent(A, B, C, D, policy_len=policy_len)
        # try to set attributes if present
        if hasattr(agent, "gamma"):
            agent.gamma = gamma
        if hasattr(agent, "action_selection"):
            agent.action_selection = action_selection

    model = {
        "A": A, "B": B, "C": C, "D": D,
        "reward_idx": reward_idx, "punish_idx": punish_idx,
        "pos_to_idx": pos_to_idx,
    }
    return agent, model
