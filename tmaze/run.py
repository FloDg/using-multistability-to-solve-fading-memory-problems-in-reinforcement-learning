from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np

from cells import *
from model import CustomModel, DoubleLayerModel
from memory import Memory
from gamerunner import GameRunner
from trainer import Trainer
from tmaze import TMaze


# This script runs a game with a given model on the T-maze environment

# Variables
corridor_length = 100
max_steps = corridor_length + 10
render = False
print_actions = True
print_qvalues = True

# 0 for a down reward, 1 for an up one, 2 for random
init = 1

# Choose the model
Model = CustomModel
# Model = DoubleLayerModel

# Choose the cell type, the number of units and the weights file

# cell = GruCellLayer
# units = 32
# weights_load_file = 'weights/gru/file'

# cell = LSTMCellLayer
# units = 32
# weights_load_file = 'weights/lstm/file'

cell = NeuromodulatedBistableRecurrentCellLayer
units = 32
weights_load_file = 'weights/nbrc/file'

# cell = JANETCellLayer
# units = 32
# weights_load_file = 'weights/janet/file'

env = TMaze(corridor_length, render, init=init)

num_states = env.num_states
num_actions = env.num_actions

model = Model(num_states, num_actions, cell, units, weights_load_file=weights_load_file)

game = GameRunner(model, env, render=render, max_steps=max_steps,
                  print_actions=print_actions, print_qvalues=print_qvalues)

max_reward = (corridor_length-1) * -0.1 + corridor_length
print('reward = ', game.run(), ' , max reward: ', max_reward)
