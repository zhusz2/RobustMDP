"""
A classic Aircraft routing problem described in Nilm 2005. 
"""

import logging
import sys

import numpy as np
from gym import utils
from gym.envs.toy_text import discrete

logger = logging.getLogger(__name__)

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

GRID_SIZE = 3


class AirCraftRoutingSimple(discrete.DiscreteEnv):
    """
    The aircraft must go through a stormy area. 
    NOTE: This is a testcase for robust/non-robust.

    The space is a 3 x 3 grid.

    Denote A as the aircraft, E as the end point.
    
    ...
    ASE
    ...

    There is exactly one storm in the area, which cost 10.
    The transition matrix is still simple 0.9, 0.1 format.

    Here if we have a huge likelihood entropy, the robust policy
    will take a detour.
    """

    def __init__(self, desc=None):

        self.nrow, self.ncol = nrow, ncol = (GRID_SIZE, GRID_SIZE)

        k = 1
        t_matrix = [[0.9, 0.1], [0.1, 0.9]]

        storm_possibilities = 2**k

        assert nrow % 2 == 1

        nA = 4
        nS = nrow * ncol * storm_possibilities

        storm_state = [0] * k

        isd = np.zeros((storm_possibilities, nrow, ncol)).astype('float64')
        isd[0, nrow // 2, 0] = 1
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col, storm_index):
            assert storm_index <= 1
            return nrow * ncol * storm_index + row * ncol + col

        def inc(row, col, a):
            orig_row = row
            orig_col = col
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # down
                row = min(row + 1, nrow - 1)
            elif a == 2:  # right
                col = min(col + 1, ncol - 1)
            elif a == 3:  # up
                row = max(row - 1, 0)
            return (row, col)

        # Storms, only one.
        self.storms = []
        self.storms.append((nrow // 2, ncol // 2))

        # Add the terminal state.
        self.terminal_pos = (nrow // 2, ncol - 1)

        for row in range(nrow):
            for col in range(ncol):
                for storm_index in range(storm_possibilities):
                    for storm_index_after in range(storm_possibilities):
                        s = to_s(row, col, storm_index)
                        for a in range(4):
                            li = P[s][a]
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol, storm_index_after)
                            cost = 1
                            # Note that this is important, since in the paper
                            # we assume the cost of a (s,a) pair is the same for whatever s'.
                            # This doesn't have much effect on this case.
                            if (row, col) in self.storms and storm_index == 1:
                                cost *= 10
                            done = (row, col) == self.terminal_pos
                            if (newrow, newcol) == self.terminal_pos:
                                li.append(
                                    (t_matrix[storm_index][storm_index_after],
                                     newstate, cost, True))
                            else:
                                li.append(
                                    (t_matrix[storm_index][storm_index_after],
                                     newstate, cost, done))

        super(AirCraftRoutingSimple, self).__init__(nS, nA, P, isd)

    def reset(self, seed=None):
        np.random.seed(seed)
        return super(AirCraftRoutingSimple, self).reset()

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
        if storm == 1:
            for s in self.storms:
                air_map[s] = 'S'
        air_map[pos] = utils.colorize('A', 'red', highlight=True)
        print_grid(air_map)
        print('')
        return
