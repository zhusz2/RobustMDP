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


class AirCraftRouting(discrete.DiscreteEnv):
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
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, desc=None):

        self.nrow, self.ncol = nrow, ncol = (GRID_SIZE, GRID_SIZE)

        init_pos = (nrow // 2, 0)
        assert nrow % 2 == 1

        nA = 4
        nS = nrow * ncol

        isd = np.zeros((nrow, ncol)).astype('float64')
        isd[init_pos] = 1
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

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

        # Storms, let a vertical line be the block.
        # BTW, here we are defining go in a storm cost high.
        self.storms = []
        for row in range(nrow // 5, 4 * nrow // 5):
            col = ncol // 2
            self.storms.append((row, col))

        # Add the terminal state.
        self.terminal_pos = (nrow // 2, ncol - 1)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    newrow, newcol = inc(row, col, a)
                    newstate = to_s(newrow, newcol)
                    cost = 1
                    # Note that this is important, since in the paper
                    # we assume the cost of a (s,a) pair is the same for whatever s'.
                    # This doesn't have much effect on this case.
                    if (row, col) in self.storms:
                        cost *= 100
                    done = (row, col) == self.terminal_pos
                    if (newrow, newcol) == self.terminal_pos:
                        li.append((1.0, newstate, cost, True))
                    else:
                        li.append((1.0, newstate, cost, done))

        super(AirCraftRouting, self).__init__(nS, nA, P, isd)

    def _decode(self, s):
        '''
            s: a number that represent the state.
            return: a tuple like (1,2)
        '''
        return (s // self.ncol, s % self.ncol)

    def reset(self, seed=None):
        np.random.seed(seed)
        return super(AirCraftRouting, self).reset()

    def render(self, mode='human', close=False):
        def print_grid(air_map):
            for i in range(air_map.shape[1]):
                print(air_map[i, :].tostring().decode('utf-8'))

        # Generate map:
        air_map = np.zeros((self.nrow, self.ncol), dtype='U10')
        for i in range(self.nrow):
            for j in range(self.ncol):
                air_map[i, j] = '.'
        for s in self.storms:
            air_map[s] = 'S'
        air_map[self.terminal_pos] = 'E'
        air_map[self._decode(self.s)] = utils.colorize(
            'A', 'red', highlight=True)
        print_grid(air_map)
        print('')
        return
