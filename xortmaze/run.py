from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np

from cells import *
from model import CustomModel
from memory import Memory
from gamerunner import GameRunner
from trainer import Trainer
from xortmaze import XorTMaze


# This script runs a game with a given model on the Xor-T-maze environment

# Variables
corridor_length = 100
middle = 5
max_steps = corridor_length + 1000
render = False
print_actions = False
print_qvalues = True

# 0 or 1 for a down reward, 2 or 3 for an up one, 4 for random
init = 0

# Choose the cell type, the number of units and the weights file

# cell = GruCellLayer
# units = 32
# weights_load_file = 'weights/gru/N32a'

# cell = LSTMCellLayer
# units = 32
# weights_load_file = 'weights/lstm/L10'

cell = NeuromodulatedBistableRecurrentCellLayer
units = 32
weights_load_file = 'weights/nbrc/O3'

# cell = JANETCellLayer
# units = 32
# weights_load_file = 'weights/janet/file'

env = XorTMaze(corridor_length, middle, render=render, init=init)

num_states = env.num_states
num_actions = env.num_actions

model = CustomModel(num_states, num_actions, cell, units, weights_load_file=weights_load_file)

game = GameRunner(model, env, render=render, max_steps=max_steps,
                  print_actions=print_actions, print_qvalues=print_qvalues)

max_reward = (corridor_length-1) * -0.1 + corridor_length
print('reward = ', game.run(), ' , max reward: ', max_reward)
