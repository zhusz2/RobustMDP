"""
A classic Aircraft routing problem described in Nilm 2005.
"""

import logging
import sys
import random

import numpy as np
from gym import utils
from gym.envs.toy_text import discrete

from math import factorial
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Action
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Storm State
GRID_SIZE = 11
K = 2
assert 2 * K + 1 <= GRID_SIZE

# Storm Transformation
SINGLE_FLIP = 0.05  # for the binomial distribution
PERTURB_FACTOR = .5


class ComplexRoutingRandom(discrete.DiscreteEnv):
    """
    The aircraft must go through a stormy area.

    The space is a N x N grid.

    Denote A as the aircraft, E as the end point.

                    X
                    X
    A               X           E
                    X
                    X

    Here Xs are a stormy area that has some chance of storm.

    For simplicity, we place A at the middle of very left.
    E at the middle of very right.

    The storm follow a transition matrix of 2^k * 2^k.
    Each entry is the probability transit from one possible state to the other.

    Assume k obstacle.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, desc=None):
        self.nrow, self.ncol = nrow, ncol = (GRID_SIZE, GRID_SIZE)

        k = K
        tk = 2**k
        nS = nrow * ncol * tk
        nA = 4

        # Get the binary: 12 -> 01100 -> [False, True, True, False, False], suppose K is 5
        fs = '{0:0' + str(K) + 'b}'

        def stormID2BoolList(id):
            return [int(x) > 0 for x in fs.format(id)]

        # calc distribution probability, treat p to be the probability of flip, 1-p fr hold
        def calcP(n, r, p):
            return (p**r) * ((1 - p)**(n - r))

        # (row, col, storm_index) to stateID
        def to_s(row, col, storm_index):
            return nrow * ncol * storm_index + row * ncol + col

        # step
        def inc(row, col, a):
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # down
                row = min(row + 1, nrow - 1)
            elif a == 2:  # right
                col = min(col + 1, ncol - 1)
            elif a == 3:  # up
                row = max(row - 1, 0)
            return (row, col)

        self.terminal_pos = (nrow // 2, ncol - 1)

        # Randomly generate cols
        storm_cols = np.random.permutation((ncol - 1) // 2)  # RANDOMNESS
        storm_cols = sorted(2 * storm_cols + 1)
        # For each row, randomly generate cols
        row_range = {}
        for c in storm_cols:  # RANDOMNESS
            a = random.randint(0, ncol)
            b = random.randint(0, ncol)
            row_range[c] = min(a, b), max(a, b)
            # it is likely that the whole column is blocked
            # note: if row_range[c] is [0, 7] then the storm length is just 7 (not 8), like python

        # Generate storm maps storm_maps
        storm_maps = np.zeros((tk, nrow, ncol)).astype('float64')
        for i in range(tk):
            bool_list = stormID2BoolList(i)
            for j in range(len(bool_list)):
                if bool_list[j]:  # there is the storm at this column
                    c = storm_cols[j]
                    storm_maps[i, row_range[c][0]:row_range[c][1], c] = 1.

        # Conventions: Q is the normial distribution and P is the perturbed one
        # Define Qs: only care about the transformation between storms
        Qs = np.zeros((tk, tk), np.float32)
        for i in range(tk):
            bool_list_i = stormID2BoolList(i)
            for j in range(tk):
                bool_list_j = stormID2BoolList(j)
                flip_bool_list = [
                    bool_list_i[z] ^ bool_list_j[z] for z in range(k)
                ]
                n_flip = sum(flip_bool_list)
                Qs[i, j] = calcP(k, n_flip, SINGLE_FLIP)
        # Perturb Qs into Ps
        perturb = PERTURB_FACTOR * np.random.rand(tk, tk)
        Ps = Qs + perturb
        for iter in range(100):  # magic, please dont laugh at me
            Ps[Ps < 0.] = 0.
            Ps[Ps > 1.] = 1.
            for i in range(tk):
                Ps[i] -= (Ps[i].mean() - 1. / tk)
        # Put Ps into P, Qs into Q:
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        Q = {s: {a: [] for a in range(nA)} for s in range(nS)}
        Pmatrix = np.zeros((nS, nA, nS), dtype=np.float64)
        Qmatrix = np.zeros((nS, nA, nS), dtype=np.float64)
        for row in range(nrow):
            for col in range(ncol):
                for old_storm_id in range(tk):
                    old_state_id = to_s(row, col, old_storm_id)
                    # Note: cost is irrelavant to the new state!
                    if storm_maps[old_storm_id, row, col] == 1:
                        cost = 100.
                    else:
                        cost = 1.
                    for new_storm_id in range(tk):
                        for a in range(4):
                            newrow, newcol = inc(row, col, a)
                            new_state_id = to_s(newrow, newcol, new_storm_id)
                            done1 = (row, col) == self.terminal_pos
                            done2 = (newrow, newcol) == self.terminal_pos
                            P[old_state_id][a].append(
                                (Ps[old_storm_id, new_storm_id], new_state_id,
                                 cost, done1 or done2))
                            Pmatrix[old_state_id, a, new_state_id] = Ps[
                                old_storm_id][new_storm_id]
                            Q[old_state_id][a].append(
                                (Qs[old_storm_id, new_storm_id], new_state_id,
                                 cost, done1 or done2))
                            Qmatrix[old_state_id, a, new_state_id] = Qs[
                                old_storm_id][new_storm_id]

        # Take home
        self.storm_cols = storm_cols
        self.row_range = row_range
        self.storm_maps = storm_maps
        self.Qs = Qs
        self.Q = Q
        self.Qmatrix = Qmatrix
        self.Ps = Ps
        self.P = P
        self.Pmatrix = Pmatrix

        # new object
        isd = np.zeros((tk, nrow, ncol)).astype('float64')
        isd[0, nrow // 2, 0] = 1
        super(ComplexRoutingRandom, self).__init__(nS, nA, P, isd)

    def reset(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return super(ComplexRoutingRandom, self).reset()

    def _decode(self, s):
        '''
            s: a number that represent the state.
            return: a tuple like (row, col), storm_index
        '''
        return ((s % (self.nrow * self.ncol)) // self.ncol,
                s % self.ncol), s // (self.ncol * self.nrow)

    def render(self, random='human', close=False):
        def print_grid(air_map):
            for i in range(air_map.shape[0]):
                print(air_map[i, :].tostring().decode('utf-8'))

        # Generate map:
        air_map = np.zeros((self.nrow, self.ncol), dtype='U10')
        for i in range(self.nrow):
            for j in range(self.ncol):
                air_map[i, j] = '.'
        air_map[self.terminal_pos] = 'E'

        pos, storm = self._decode(self.s)
        '''
        if storm == 1:
            for s in self.storms:
                air_map[s] = 'S'
        '''
        storm_map_now = self.storm_maps[storm]
        air_map[storm_map_now > .5] = 'S'
        air_map[pos] = utils.colorize('A', 'red', highlight=True)
        print_grid(air_map)
        print('')
        return
