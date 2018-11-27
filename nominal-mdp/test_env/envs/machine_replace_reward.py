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

        for s in range(nS):
            for a in range(nA):
                li = P[s][a]
                if a == 0:
                    if s < nS - 1:
                        newstate = s + 1
                        cost = np.random.normal(0, 1e-4)  #N(0, 1e-4)
                    else:
                        newstate = 0
                        cost = np.random.normal(100, 800)
                if a == 1:
                    newstate = 0
                    if s < nS - 1:
                        cost = np.random.normal(130, 1)
                    else:
                        cost = np.random.normal(130, 20)

                # finite horizon
                done = False
                li.append((1.0, newstate, cost, done))

        super(MachineReplacementRewardEnv, self).__init__(nS, nA, P, done)

    def reset(self, seed=None):
        np.random.seed(seed)
        return super(MachineReplacementRewardEnv, self).reset()
