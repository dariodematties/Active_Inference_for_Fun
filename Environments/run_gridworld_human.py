# run_gridworld_human.py
import time
import numpy as np

from gridworld_env import GridWorld  # make sure gridworld_env.py is on PYTHONPATH

def main():
    env = GridWorld(
        n_rows=5, n_cols=7,
        reward_pos=(4, 6), punish_pos=(0, 6),
        start_pos=(0, 0),
        step_cost=0.0, reward=1.0, punish=-1.0,
        max_steps=200,
        slip_prob=0.0,
        render_mode="human",
    )
    rng = np.random.default_rng(0)

    obs, info = env.reset(seed=0)
    total_reward = 0.0

    try:
        for t in range(500):  # run some steps; adjust as you like
            # choose an action (random policy for demo)
            action = int(rng.integers(0, env.action_space.n))
            obs, r, terminated, truncated, info = env.step(action)
            total_reward += r

            # dynamic in-place update (don’t print!)
            env.render()

            # throttle the animation a bit
            time.sleep(0.05)

            if terminated or truncated:
                # brief pause so you can see the terminal state
                time.sleep(1.5)
                obs, info = env.reset()
                total_reward = 0.0

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()

