from gym.envs.registration import register

import gym
from test_env.envs import *

register(
    id='AirCraftRouting-v1',
    entry_point='test_env.envs.air_craft_routing:AirCraftRouting',
)
