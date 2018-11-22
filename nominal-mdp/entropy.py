# Implementation of maximum likelihood model

import numpy as np
from test_env import *

eps = 0.01
large = 15


def SigmaDerivative(beta, q1, V, mu, nS):
    if mu <= eps or max(V / mu) > large:
        # limiting value when mu -> 0
        arg = np.argmax(V)
        q_m = q1[arg]
        return beta + np.log(q_m)

    const1 = sum(q1 * np.exp(V / mu))

    const2 = sum(q1 * np.exp(V / mu) * V)
    return np.log(const1) + beta - const2 / (mu * const1)


def Sigma(beta, q1, V, mu, nS):
    if mu <= eps or max(V / mu) > large:
        # limiting value when mu -> 0
        return max(V)

    const1 = np.log(sum(q1 * np.exp(V / mu)))

    return mu * const1 + beta * mu


def SigmaEntropy(P, V, nS, nA, sigma, tol):
    """
    Parameters:
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, cost, terminal)
    V: list of floats
        Value functions for nS states
    nS: int
        number of states
    nA: int
        number of actions
    sigma: 2D array of floats
        Robust transition matrix
    tol: float
        beta margin
    """
    for a in range(nA):
        # implement independently for every action
        for s in range(nS):
            # implement independently for every state
            q1 = []
            V1 = []
            nS_next = 0
            for s_next in P[s][a]:
                # initialize V and f
                if s_next[0] > 0.0:
                    q1.append(s_next[0])
                    V1.append(V[s_next[1]])
                    nS_next += 1
            q1 = np.asarray(q1)
            V1 = np.asarray(V1)
            V_bar = sum(V1 * q1)
            V_max = max(V1)
            beta = max(- np.log(q1)) - tol

            mu_minus = 0
            mu_plus = (V_max - V_bar) / beta
            mu = mu_plus
            delta = 0.001
            while mu_plus - mu_minus > delta * (1 + mu_plus + mu_minus):
                mu = (mu_plus + mu_minus) / 2
                # calculate SigmaDerivative at mu
                sigma_der = SigmaDerivative(beta, q1, V1, mu, nS_next)
                if abs(sigma_der) <= delta:
                    break
                if sigma_der > 0:
                    mu_plus = mu
                else:
                    mu_minus = mu

            sigma[s, a] = Sigma(beta, q1, V1, mu, nS_next)
