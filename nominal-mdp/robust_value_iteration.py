### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

from test_env.envs.common import render_state
import numpy as np
import gym
import time
from test_env import *
from likelihood2 import SigmaLikelihood
from entropy import SigmaEntropy
from value_iteration import value_iteration

np.set_printoptions(precision=3)

EPSILON = 0.1


def CalculateSigma(P, V, nS, nA, gamma, sigma):
    '''
        Here sigma is assumed to be allocated and be modified in place.
        Here we calculate Sigma defined (7) of Nilm, 2005
    '''
    for state in range(nS):
        for action in range(nA):
            BV = 0
            for t in P[state][action]:
                probability = t[0]
                nextstate = t[1]
                done = t[3]
                if done:
                    BV += 0
                else:
                    BV += probability * gamma * V[nextstate]
            # Calculate per state action hat sigma.
            hat_sigma = BV
            # I simply have a sigma(v) for now.
            # This should be a bound hat{sigma}(v) which is the bound
            # hat{sigma}(v) - epsilon/N <= hat(sigma) <= hat(sigma)
            ########################################################
            #####################
            ## IMPLEMENT HERE ###
            #####################
            # TODO(team):

            ########################################################
            assert hat_sigma >= BV
            assert hat_sigma <= EPSILON / nS + BV
            sigma[state, action] = hat_sigma
    print(sigma.shape)


def RobustBellmanOp(P, Sigma, state, action, gamma):
    """Represent R(s,a) + gamma * Sigma
    Notice that R(s,a) is the expected cost of execute |a| at state s.
    Returns float value

    Ve is the estimated robust value function.
    Returns
    -------
    value function of state: float
    	The value function of state.
    """
    BV = 0
    # Here there is a strong assumption that
    #   sum(p(s'|(s,a)) * R(s,a,s')) = R(s,a,s') = R(s,a)
    for t in P[state][action]:
        probability = t[0]
        nextstate = t[1]
        cost = t[2]
        done = t[3]
        BV += probability * cost

        if done:
            BV += 0
        else:
            BV += probability * gamma * Sigma[state, action]
    return BV


def robust_value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
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
    # V = np.zeros(nS)
    V = 1000 * np.ones(nS)

    sigma = np.zeros((nS, nA))

    policy = np.zeros(nS, dtype=int)
    for _ in range(max_iteration):
        print('one iter')
        # Need to estimate sigma, which is of dimension |nS|*|nA|
        # This can simply be p^T V for now.
        # CalculateSigma(P, V, nS, nA, gamma, sigma)
        # SigmaLikelihood(P, V, nS, nA, sigma, tol)
        SigmaEntropy(P, V, nS, nA, sigma, tol)

        newV = np.zeros(nS)
        for state in range(nS):
            BV = np.zeros(nA)
            for action in range(nA):
                BV[action] = RobustBellmanOp(P, sigma, state, action, gamma)
            newV[state] = BV.min()
        # Calculate difference of the value functions.
        Vdiff = np.max(np.abs(newV - V))
        V = newV
        if Vdiff < tol:
            break

    # Calculate the policy.
    for state in range(nS):
        BV = np.zeros(nA)
        for action in range(nA):
            BV[action] = RobustBellmanOp(P, sigma, state, action, gamma)
        policy[state] = np.argmin(BV)

    print(V)
    print(policy)
    return V, policy


def example(env):
    """Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
    env.seed(0)
    from gym.spaces import prng
    prng.seed(10)  # for print the location
    # Generate the episode
    ob = env.reset()
    for t in range(20):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render()


def render_single(env, policy, seed_feed=99, if_render=True, iter_tot=100):
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
    ob_list = []
    ob = env.reset()
    ob_list.append(ob)
    env.seed(seed_feed)
    for t in range(iter_tot):
        env.render()
        time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        ob_list.append(ob)
        episode_reward += rew
        if done:
            break
    assert done
    if if_render:
        env.render()
    print("Episode cost: %f" % episode_reward)
    return episode_reward, ob_list


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
def main_render_robust():
    # TODO: make this an arg.
    env = gym.make("AirCraftRouting-v4")
    print(env.__doc__)
    print("Here is an example of state, action, cost, and next state")
    # example(env)
    V_vi, p_vi = robust_value_iteration(
        env.P, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3)
    render_single(env, p_vi)
    print('------------ All the storm map ------------')
    render_state(env.nS - 1, env.nrow, env.ncol, env.storm_maps, env.terminal_pos)
    # print(env.storm_maps.max(0))
    print('-------------------------------------------')


def main_experiments():
    env = gym.make("AirCraftRouting-v4")
    V_vi, p_vi = robust_value_iteration(
        env.P, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3)
    V_old, p_old = value_iteration(
        env.P, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3)
    ret_robust = []
    ret_normial = []
    exp_tot = 5
    for j in range(exp_tot):
        ret, _ = render_single(env, p_vi, j, False)
        ret_robust.append(ret)
        ret, _ = render_single(env, p_old, j, False)
        ret_normial.append(ret)
        print("Finish exp iter %d out of %d" % (j, exp_tot))
    print(sum(ret_robust) / float(exp_tot))
    print(sum(ret_normial) / float(exp_tot))
    import ipdb
    ipdb.set_trace()

if __name__ == '__main__':
    main_experiments()
