import os
import glob
import numpy as np
import random
from gym.envs.toy_text import discrete


class MachineReplacementRewardEnv(discrete.DiscreteEnv):
    def __init__(self, desc=None):
        nA = 2
        nS = 50

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.Q = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.mean = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.variance = {s: {a: [] for a in range(nA)} for s in range(nS)}
        mean = self.mean
        variance = self.variance

        isd = np.ones(nS) / nS

        for s in range(nS):
            for a in range(nA):
                if a == 0:
                    if s < nS - 1:
                        mean[s][a] = 0
                        variance[s][a] = 1e-4
                    else:
                        mean[s][a] = 100
                        variance[s][a] = 800
                if a == 1:
                    if s < nS - 1:
                        mean[s][a] = 130
                        variance[s][a] = 1
                    else:
                        mean[s][a] = 130
                        variance[s][a] = 20

        for s in range(nS):
            for a in range(nA):
                li = P[s][a]
                li_q = self.Q[s][a]
                cost = np.random.normal(mean[s][a], variance[s][a])
                if a == 0:
                    if s < nS - 1:
                        newstate = s + 1
                    else:
                        newstate = 0
                if a == 1:
                    newstate = 0
                # finite horizon
                done = False
                li.append((1.0, newstate, cost, done))
                li_q.append((1.0, newstate, mean[s][a], done))

        super(MachineReplacementRewardEnv, self).__init__(nS, nA, P, isd)

    def reset(self, seed=None):
        np.random.seed(seed)
        # Need to resample env
        return super(MachineReplacementRewardEnv, self).reset()
