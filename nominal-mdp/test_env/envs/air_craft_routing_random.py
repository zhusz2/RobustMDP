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

GRID_SIZE = 11


class AirCraftRoutingRandom(discrete.DiscreteEnv):
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

        k = 1
        t_matrix = [[0.9, 0.1], [0.1, 0.9]]

        storm_possibilities = 2**k

        assert nrow % 2 == 1

        nA = 4
        nS = nrow * ncol * storm_possibilities

        storm_state = [0] * k

        isd = np.zeros((nrow, ncol, 2**k)).astype('float64')
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

        # Storms, let a vertical line be the block.
        for row in range(nrow // 5, 4 * nrow // 5):
            for storm_index in range(storm_possibilities):
                for storm_index_after in range(storm_possibilities):
                    col = ncol // 2
                    s = to_s(row, col, storm_index)
                    for a in range(4):
                        li = P[s][a]
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol, storm_index_after)
                        # TODO: extend to case with more obstacles
                        cost = 100 if storm_index_after == 1 else 1
                        li.append((t_matrix[storm_index][storm_index_after],
                                   newstate, cost, False))

        # Add the terminal state.
        for storm_index in range(storm_possibilities):
            terminal_s = to_s(nrow // 2, ncol - 1, storm_index)
            for storm_index_after in range(storm_possibilities):
                for a in range(4):
                    li = P[terminal_s][a]
                    li.append((0.0, s, 0, True))

        for row in range(nrow):
            for col in range(ncol):
                for storm_index in range(storm_possibilities):
                    for storm_index_after in range(storm_possibilities):
                        s = to_s(row, col, storm_index)
                        for a in range(4):
                            li = P[s][a]
                            if len(li) is 0:
                                newrow, newcol = inc(row, col, a)
                                newstate = to_s(newrow, newcol,
                                                storm_index_after)
                                cost = 1
                                li.append(
                                    (t_matrix[storm_index][storm_index_after],
                                     newstate, cost, False))

        super(AirCraftRoutingRandom, self).__init__(nS, nA, P, isd)

    def reset(self, seed=None):
        np.random.seed(seed)
        return super(AirCraftRoutingRandom, self).reset()
