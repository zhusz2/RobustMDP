import os
import glob
import numpy as np
import random
from gym.envs.toy_text import discrete


class MachineReplacementDirichletEnv(discrete.DiscreteEnv):
    def __init__(self, desc=None):
        nA = 2
        nS = 10
        done = False
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        # 8 represents R1, 9 represents R2
        for s in range(nS):
            for a in range(nA):
                # [prob, newstate, cost, done]
                li = P[s][a]
                if a == 1:
                    # update state
                    if s == 7:
                        li.append((0.6, 8, 2, done))
                        li.append((0.1, 9, 10, done))
                        li.append((0.3, 7, 20, done))
                    elif s == 8:
                        li.append((1.0, 8, 2, done))
                    elif s == 9:
                        li.append((0.6, 8, 2, done))
                        li.append((0.4, 9, 10, done))
                    else:
                        li.append((0.6, 8, 2, done))
                        li.append((0.1, 9, 10, done))
                        if s == 6:
                            li.append((0.3, 7, 20, done))
                        else:
                            li.append((0.3, s + 1, 0, done))

                if a == 0:
                    if s == 7:
                        li.append((1.0, 7, 20, done))
                    elif s == 8:
                        li.append((0.2, 8, 0, done))
                        li.append((0.8, 0, 0, done))
                    elif s == 9:
                        li.append((1.0, 9, 0, done))
                    else:
                        li.append((0.2, s, 0, done))
                        if s == 6:
                            li.append((0.8, 7, 20, done))
                        else:
                            li.append((0.8, s + 1, 0, done))

        super(MachineReplacementDirichletEnv, self).__init__(nS, nA, P, done)

    def reset(self, seed=None):
        np.random.seed(seed)
        return super(MachineReplacementDirichletEnv, self).reset()
