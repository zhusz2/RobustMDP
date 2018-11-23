### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from test_env import *
from value_iteration import value_iteration
from robust_value_iteration import robust_value_iteration

# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    # TODO: make this an arg.
    env = gym.make("AirCraftRoutingSimple-v1")
    V_vi, p_vi = value_iteration(
        env.P, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3)
    assert abs(V_vi[3] - 2.9) <= 1e-6
    # 3 is the initial state, 2.9 is hand calculated value.
    env.reset()
    V_vi, p_vi = robust_value_iteration(
        env.P, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3)
    print(V_vi, p_vi)
    # assert abs(V_vi[3] - 4) <= 0.1
