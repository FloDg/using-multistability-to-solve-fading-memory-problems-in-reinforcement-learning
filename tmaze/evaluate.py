from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np

from cells import *
from model import CustomModel, DoubleLayerModel
from memory import Memory
from gamerunner import GameRunner
from trainer import Trainer
from tmaze import TMaze


# This script aims at testing several sets of weights on the T-maze environment.

# List of corridor lengths on which the agents have to be tested
corridor_lengths = [10, 100, 1000, 10000, 100000]

# List of weights of the agents
weights = ['weights/gru/file']

# List of number of units for each agent
units = [32 for _ in weights]

num_states = 3
num_actions = 4

for w, u in zip(weights, units):
    print('\n----------- Evaluating weights:', w, '-----------')

    if 'double' in w:
        Model = DoubleLayerModel

    else:
        Model = CustomModel

    if 'gru' in w:
        cell = GruCellLayer

    elif 'nbrc' in w:
        cell = NeuromodulatedBistableRecurrentCellLayer

    elif 'brc' in w:
        cell = BistableRecurrentCellLayer

    elif 'janet' in w:
        cell = JANETCellLayer

    elif 'lstm' in w:
        cell = LSTMCellLayer

    else:
        print('Unknown cell')
        continue

    model = Model(
        num_states,
        num_actions,
        cell,
        u,
        weights_load_file=w
    )

    for l in corridor_lengths:
        env_up = TMaze(l, init=1)
        env_down = TMaze(l, init=0)

        game_up = GameRunner(
            model,
            env_up,
            max_steps=l+100
        )

        game_down = GameRunner(
            model,
            env_down,
            max_steps=l+100
        )

        max_reward = -0.1 * (l-1) + l

        r_up = game_up.run()
        r_down = game_down.run()

        print('Length: {}, up reward: {}, down reward: {}, max reward: {}' \
              .format(l, r_up, r_down, max_reward))
