from gym import utils
import numpy as np


def render_state(state_id, nrow, ncol, storm_maps, terminal_pos):
    def _decode(s):
        '''
            s: a number that represent the state.
            return: a tuple like (row, col), storm_index
        '''
        return ((s % (nrow * ncol)) // ncol,
                s % ncol), s // (ncol * nrow)
    def print_grid(air_map):
        for i in range(air_map.shape[0]):
            print(air_map[i, :].tostring().decode('utf-8'))

    # Generate map:
    air_map = np.zeros((nrow, ncol), dtype='U10')
    for i in range(nrow):
        for j in range(ncol):
            air_map[i, j] = '.'
    air_map[terminal_pos] = 'E'

    pos, storm = _decode(state_id)
    storm_map_now = storm_maps[storm]
    air_map[storm_map_now > .5] = 'S'
    air_map[pos] = utils.colorize('A', 'red', highlight=True)
    print_grid(air_map)
    print('')
    return
