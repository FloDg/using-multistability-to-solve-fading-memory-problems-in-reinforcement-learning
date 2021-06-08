from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np

from cells import *
from model import CustomModel
from memory import Memory
from gamerunner import GameRunner
from trainer import Trainer
from xortmaze import XorTMaze


# This script aims at testing several sets of weights on the Xor-T-maze environment.

# List of corridor lengths on which the agents have to be tested
corridor_lengths = [100, 1000]
# Position at which the second information is should be given
middle = 50

# List of weights of the agents
weights = ['weights/gru/list']

# List of number of units for each agent
units = [32 for _ in weights]

num_states = 3
num_actions = 4

test_names = ['Down 1-2',
              'Down 2-1',
              '  Up 1-1',
              '  Up 2-2']

for w, u in zip(weights, units):
    print('\n===================== Evaluating weights:', w, '=====================')

    if 'gru' in w:
        cell = GruCellLayer

    elif 'nbrc' in w:
        cell = NeuromodulatedBistableRecurrentCellLayer

    elif 'brc' in w:
        cell = BistableRecurrentCellLayer

    elif 'janet' in w:
        cell = JANETCellLayer

    else:
        print('Unknown cell')
        continue

    model = CustomModel(
        num_states,
        num_actions,
        cell,
        u,
        weights_load_file=w
    )

    for l in corridor_lengths:
        test_envs = []
        for i in range(4):
            test_envs.append(XorTMaze(l, middle, init=i))

        test_games = []
        for e in test_envs:
            test_games.append(GameRunner(
                model,
                e,
                max_steps=l+100
            ))

        rewards = []
        for g in test_games:
            rewards.append(g.run())

        max_reward = -0.1 * (l-1) + l
        print('---------------------- Length {}, max reward: {} ----------------------' \
              .format(l, max_reward))

        for n, r in zip(test_names, rewards):
            print('{}: {:>.2f}'.format(n, r))
