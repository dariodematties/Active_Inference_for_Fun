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
    sophisticated: bool = False,   # <-- NEW
):
    """
    Build a fully-observable Active Inference agent for GridWorld.
    Set sophisticated=True to request sophisticated inference (if supported by pymdp).

    Returns:
        agent : pymdp.agent.Agent
        model : dict with A,B,C,D and helpers
        controls: dict with 'infer_policies' wrapper (uses sophisticated mode if available)
    """
    S = n_rows * n_cols
    O = S
    U = 4

    def pos_to_idx(r: int, c: int) -> int:
        return r * n_cols + c

    def clip_move(r: int, c: int, u: int) -> tuple[int, int]:
        if u == 0: r = max(0, r - 1)           # up
        elif u == 1: c = min(n_cols - 1, c + 1)# right
        elif u == 2: r = min(n_rows - 1, r + 1)# down
        elif u == 3: c = max(0, c - 1)         # left
        return r, c

    # A: identity (obs == state)
    A = np.eye(O, S, dtype=np.float64)

    # B: deterministic transitions (S x S x U)
    B = np.zeros((S, S, U), dtype=np.float64)
    for s_prev in range(S):
        r, c = divmod(s_prev, n_cols)
        for u in range(U):
            r2, c2 = clip_move(r, c, u)
            s_next = pos_to_idx(r2, c2)
            B[s_next, s_prev, u] = 1.0

    # C: outcome preferences over observations
    C = np.zeros(O, dtype=np.float64)
    C[pos_to_idx(*reward_pos)] = c_reward
    C[pos_to_idx(*punish_pos)] = c_punish

    # D: prior over initial state
    if start_pos is not None:
        D = np.zeros(S, dtype=np.float64)
        D[pos_to_idx(*start_pos)] = 1.0
    else:
        D = np.ones(S, dtype=np.float64)
        D[[pos_to_idx(*reward_pos), pos_to_idx(*punish_pos)]] = 0.0
        D /= D.sum()

    # --- Instantiate Agent with broad compatibility ---
    agent = None
    err = None
    for ctor in (
        # try passing everything, including the new flag
        lambda: Agent(A=A, B=B, C=C, D=D,
                      policy_len=policy_len, gamma=gamma,
                      action_selection=action_selection,
                      sophisticated=sophisticated),
        # without 'sophisticated' kw
        lambda: Agent(A=A, B=B, C=C, D=D,
                      policy_len=policy_len, gamma=gamma,
                      action_selection=action_selection),
        # positional fallback
        lambda: Agent(A, B, C, D, policy_len=policy_len),
    ):
        try:
            agent = ctor()
            break
        except TypeError as e:
            err = e
    if agent is None:
        raise TypeError(f"Could not construct pymdp.Agent with provided arguments: {err}")

    # Try to set sophisticated mode post-hoc if attribute exists
    for attr in ("sophisticated", "use_sophisticated_inference", "sophisticated_inference"):
        if hasattr(agent, attr):
            try:
                setattr(agent, attr, bool(sophisticated))
            except Exception:
                pass

    # --- Small control wrapper so your loop can request the right policy update ---
    def infer_policies_wrapper():
        """
        Call the appropriate policy inference.
        Uses 'mode'/'method' kw if supported; otherwise falls back to default.
        """
        # common patterns across pymdp versions:
        for kw in ({"mode": "sophisticated"}, {"method": "sophisticated"}):
            if sophisticated:
                try:
                    agent.infer_policies(**kw)
                    return
                except TypeError:
                    pass
        # classical or fallback
        agent.infer_policies()

    model = {
        "A": A, "B": B, "C": C, "D": D,
        "pos_to_idx": pos_to_idx,
        "reward_idx": pos_to_idx(*reward_pos),
        "punish_idx": pos_to_idx(*punish_pos),
    }
    controls = {
        "infer_policies": infer_policies_wrapper,
        "sophisticated": sophisticated,
        "policy_len": policy_len,
        "gamma": gamma,
        "action_selection": action_selection,
    }
    return agent, model, controls
