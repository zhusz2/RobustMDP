# Implementation of maximum likelihood model

import numpy as np
from test_env import *

eps = 0.001


def SigmaDerivative(beta_max, beta, f, V, mu, nS):
    den = mu * np.ones(nS) - V
    for i in range(nS):
        den[i] = eps if den[i] <= 0 else den[i]
    lambda_mu = 1 / sum(f / den)
    const1 = sum(f / den**2)
    const2 = sum(f * np.log(lambda_mu / den))
    return (beta_max - beta + const2) * lambda_mu**2 * const1


def Sigma(beta, f, V, mu, nS):
    den = mu * np.ones(nS) - V
    for i in range(nS):
        den[i] = eps if den[i] <= 0 else den[i]
    lambda_mu = 1 / sum(f / den)
    const1 = 0
    for i in range(nS):
        if f[i]:
            const1 += f[i] * np.log(lambda_mu * f[i] / den[i])
    return mu - (1 + beta) * lambda_mu + lambda_mu * const1


def SigmaLikelihood(P, V, nS, nA, sigma):
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
    gamma: float
        Discount factor. Number in range [0, 1)
    sigma: 2D array of floats
        Robust transition matrix
    """
    tol = 0.001
    for a in range(nA):
        # implement independently for every action
        log_like = np.zeros(nS)
        for s in range(nS):
            for s_next in P[s][a]:
                log_like[s] += s_next[0] * np.log(s_next[0])

        beta_max = sum(log_like)
        beta_s_list = log_like - tol * np.ones(nS)

        for s in range(nS):
            # implement independently for every state
            f = np.zeros(nS)
            for s_next in P[s][a]:
                # initialize V and f
                f[s_next[1]] = s_next[0]
            V_bar = sum(V * f)
            V_max = max(V)
            beta_s = beta_s_list[s]

            mu_minus = V_max
            mu_plus = (V_max - np.exp(beta_s - beta_max) * V_bar) / (1 - np.exp(beta_s - beta_max))
            mu = mu_plus
            delta = 0.001
            while mu_plus - mu_minus > delta * (1 + mu_plus + mu_minus):
                mu = (mu_plus + mu_minus) / 2
                # calculate SigmaDerivative at mu
                sigma_der = SigmaDerivative(beta_max, beta_s, f, V, mu, nS)
                if abs(sigma_der) <= delta:
                    break
                if sigma_der > 0:
                    mu_plus = mu
                else:
                    mu_minus = mu

            sigma[s, a] = Sigma(beta_s, f, V, mu, nS)
