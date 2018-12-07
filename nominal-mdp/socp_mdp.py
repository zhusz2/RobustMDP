### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from test_env import *
import os
import cvxpy as cvx
import math
from scipy.stats import norm

np.set_printoptions(precision=3)


def solve_rho(P, mean, variance, nS, nA, gamma=0.8, eps=0.01):
    rho = cvx.Variable((nS * nA), nonneg=True)
    y = cvx.Variable(1)
    soc_constraint = []
    # soc_constraint.append(rho >= 0)

    q = np.zeros(nS)
    q = np.ones(nS) / nS

    # q[0] = 1

    def flat(s, a):
        return s * nA + a

    for s in range(nS):
        rho_sum = 0
        p_sum = 0
        for a in range(nA):
            rho_sum += rho[flat(s, a)]
            for entry in P[s][a]:
                p = entry[0]
                nextstate = entry[1]
                p_sum += p * rho[flat(nextstate, a)]
        soc_constraint.append(q[s] - rho_sum + gamma * p_sum == 0)
    mean_sum = 0
    for s in range(nS):
        for a in range(nA):
            mean_sum += rho[flat(s, a)] * mean[s][a]

    var_sum = 0
    for s in range(nS):
        for a in range(nA):
            var_sum += rho[flat(s, a)] * math.sqrt(variance[s][a])

    # print(f'variance: {variance}')
    # print(f'mean: {mean}')
    # soc_constraint.append(mean_sum + (1 - eps) / eps * cvx.norm(var_sum) <= y)
    print(f'Phi_invers: {norm.ppf(1-eps)}')
    soc_constraint.append(
        mean_sum + norm.ppf(1 - eps) * cvx.norm(var_sum) <= y)
    # soc_constraint.append(mean_sum <= y)

    objective = cvx.Minimize(y)
    prob = cvx.Problem(objective, soc_constraint)
    result = prob.solve()
    print(result)
    print(rho.value)

    rho_dict = {s: {a: [] for a in range(nA)} for s in range(nS)}
    for s in range(nS):
        for a in range(nA):
            rho_dict[s][a] = rho.value[s * nA + a]
    return rho_dict


def solve(P, nS, nA, mean, variance, gamma=0.9, max_iteration=20, tol=1e-3):
    """
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, cost, terminal)
	nS: int
		number of states
	nA: int
		number of actions
        mean:  dict of [s][a]
        variance: dict of [s][a]
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
    rho = solve_rho(P, mean, variance, nS, nA)
    # Then use rho[s][a] to find the policy
    print(f'rho: {rho}')
    policy = np.zeros(nS)
    for s in range(nS):
        rho_sum = 0
        for a in range(nA):
            rho_sum += rho[s][a]
        assert rho_sum > 0
        rho_max = rho[s][0]
        rho_max_action = 0
        for a in range(nA):
            if rho[s][a] > rho_max:
                rho_max = rho[s][a]
                rho_max_action = a
        policy[s] = rho_max_action

    return policy


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
    # print("Here is an example of state, action, cost, and next state")
    # example(env)
    # print(env.P)
    p_vi = solve(
        env.P,
        env.nS,
        env.nA,
        env.mean,
        env.variance,
        gamma=0.999,
        max_iteration=100,
        tol=1e-3)
    print(p_vi)
    for _ in range(1000):
        run_single(env, p_vi)
