import os
import glob
import numpy as np
import random
from gym.envs.toy_text import discrete

class MachineReplacementEnv(discrete.DiscreteEnv):
    def __init__(self, desc=None):
        nA = 2
        nS = 50

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        for s in range(nS):
            for a in range(nA):
                li = P[s][a]
                if a == 0:
                    if s < nS-1:
                        newstate = s+1
                        cost = np.random.normal(0, 1e-4) #N(0, 1e-4)
                    else:
                        newstate = 0
                        cost = np.random.normal(100, 800)
                if a == 1:
                    newstate = 0
                    if s < nS-1:
                        cost = np.random.normal(130, 1)
                    else:
                        cost = np.random.normal(130, 20)

                # finite horizon
                done = False
                li.append((1.0, newstate, cost, done))
                        

        super(MachineReplacementEnv, self).__init__(nS, nA, P, done)


    def reset(self, seed=None):
        np.random.seed(seed)
        return super(MachineReplacementEnv, self).reset()

    '''
    def step(self, action):

        if action == 0:
            if self.current_state < self.state_num:
                self.current_state += 1
                cost = np.random.normal(0, 1e-4) #N(0, 1e-4)
            else:
                self.current_state = 1
                cost = np.random.normal(100, 800)
        if action == 1:
            self.current_state = 1
            if self.current_state < self.state_num:
                cost = np.random.normal(130, 1)
            else:
                cost = np.random.normal(130, 20)
            self.terminal_status = 1
        return cost, self.terminal_status
    '''


class MachineReplacementDirichletEnv(discrete.DiscreteEnv):
    def __init__(self, desc=None):
        nA = 2
        nS = 50
        done = False
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        # 9 represents R1, 10 represents R2
        for s in range(nS):
            for a in range(nA):
                # [prob, newstate, cost, done]
                li = P[s][a]
                if a == 1:
                    # update state
                    if s == 8:
                        li.append((0.6, 9, 2, done))
                        li.append((0.1, 10, 10, done))
                        li.append((0.3, 8, 20, done))
                    elif s == 9:
                        li.append((1.0, 9, 2, done))
                    elif s == 10:
                        li.append((0.6, 9, 2, done))
                        li.append((0.4, 10, 10, done))
                    else:
                        li.append((0.6, 9, 2, done))
                        li.append((0.1, 10, 10, done))
                        if s == 7:
                            li.append((0.3, 8, 20, done))
                        else:
                            li.append((0.3, s+1, 0, done))
                    
                if a == 0:
                    if s == 8:
                        li.append((1.0, 8, 20, done))
                    elif s == 9:
                        li.append((0.2, 9, 0, done))
                        li.append((0.8, 1, 0, done))
                    elif s == 10:
                        li.append((1.0, 10, 0, done))
                    else:
                        li.append((0.2, s, 0, done))
                        if s == 7:
                            li.append((0.8, 8, 20, done))
                        else:
                            li.append((0.8, s+1, 0, done))
                    
        super(MachineReplacementDirichletEnv, self).__init__(nS, nA, P, done)


    def reset(self, seed=None):
        np.random.seed(seed)
        return super(MachineReplacementDirichletEnv, self).reset()

    '''
    def step(self, action):
        if action == 1:
            # update state
            if self.current_state == 8:
                self.current_state = np.random.choice([-1, -2, 8], p=[0.6, 0.1, 0.3])
            elif self.current_state > 0:
                self.current_state = np.random.choice([-1, -2, self.current_state+1], \
                                        p=[0.6, 0.1, 0.3])
            elif self.current_state == -2:
                self.current_state == np.random.choice([-1, -2], p=[0.6, 0.4])

            # the cost
            if self.current_state == -2:
                cost = 10
            elif self.current_state == -1:
                cost = 2
            elif self.current_state == 8:
                cost = 20
            else:
                cost = 0

        if action == 0:
            if self.current_state == 8:
                cost = 20
            elif self.current_state > 0:
                self.current_state += np.random.choice(np.arange(2), p=[0.2, 0.8])
                cost = 0
            elif self.current_state == -1:
                self.current_state += np.random.choice([-1, 1], p=[0.2, 0.8])
                cost = 0
            else:
                cost = 0

        return cost, self.current_state
        
    def reset(self):
        self.current_state = 1
    '''
