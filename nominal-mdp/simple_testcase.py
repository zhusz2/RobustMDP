### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from test_env import *
from value_iteration import value_iteration
from robust_value_iteration import robust_value_iteration


def run_single(env, policy, seed_feed=99, iter_tot=100, gamma=0.8):
    """Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

    episode_reward = 0
    env = gym.make("MachineReplacement-v1")
    ob = env.reset()
    for t in range(iter_tot):
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    # assert done
    print(episode_reward)
    return episode_reward


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    # TODO: make this an arg.
    env = gym.make("MachineReplacement-v1")
    V_vi, p_vi = value_iteration(
        env.Q, env.nS, env.nA, gamma=0.9, max_iteration=100, tol=1e-3)

    # for _ in range(100):
    #     run_single(env, p_vi)
    # assert abs(V_vi[3] - 2.9) <= 1e-6
    # 3 is the initial state, 2.9 is hand calculated value.
    env.reset()
    # V_vi, p_vi, _ = robust_value_iteration(
    #     env.P, env.nS, env.nA, gamma=0.9, max_iteration=100, tol=1e-3)
    # print(V_vi, p_vi)
    # assert abs(V_vi[3] - 4) <= 0.1
    for _ in range(1000):
        run_single(env, p_vi)
