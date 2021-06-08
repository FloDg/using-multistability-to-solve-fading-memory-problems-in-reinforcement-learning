from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from cells import *
from model import CustomModel
from xortmaze import XorTMaze

# This script runs a game on the Xor-T-maze environment and retains the hidden
# states of the cell at each timestep.

# Variables
corridor_length = 1000
middle = 500
render = False

# When using NBRC cell, set to True to also retain the 'z' and 'r' values
z_and_r = False

# 0 or 1 for a down reward, 2 or 3 for an up one, 4 for random
init = 1

# Choose the cell type, the number of units and the weights file

cell = GruCellLayer
units = 32
weights_load_file = 'weights/gru/file'

# cell = LSTMCellLayer
# units = 32
# weights_load_file = 'weights/lstm/file'

# cell = NeuromodulatedBistableRecurrentCellLayer
# units = 32
# weights_load_file = 'weights/nbrc/file'

# cell = JANETCellLayer
# units = 32
# weights_load_file = 'weights/janet/file'


max_steps = corridor_length + 100
env = XorTMaze(corridor_length, middle, init=init, render=render, print_end=True)


num_states = env.num_states
num_actions = env.num_actions

model = CustomModel(
    num_states,
    num_actions,
    cell,
    nb_units=units,
    weights_load_file=weights_load_file,
    return_state=True
)

i = 0
state = env.reset()
tot_reward = 0
hidden_states = [np.zeros((1, units))]

if z_and_r:
    zs = [np.zeros((1, units))]
    rs = [np.zeros((1, units))]

if render:
    env.render_now()

# Play the game
while True:
    # Change one or more values of the hidden state
    # if i == corridor_length//2:
    #       model.change_hidden(2, -1)

    out, h = model.predict(state)

    hidden_states.append(h[0].numpy())

    if z_and_r:
        zs.append(h[1].numpy())
        rs.append(h[2].numpy())

    action = np.argmax(out)

    next_state, reward, done = env.step(action)

    if render:
        env.render_now()

    tot_reward += reward
    state = next_state

    if done or i == max_steps-1:
        model.reset_state()
        break

    i += 1

max_reward = (corridor_length-1) * -0.1 + corridor_length
print('reward = ', tot_reward, ' , max reward: ', max_reward)

hidden_states = np.array(hidden_states)

# Plot the hidden states (and 'z' and 'r' if necessary)

fig = plt.figure()

x = range(hidden_states.shape[0])
plt.plot(x, hidden_states[:, 0], linewidth=1)

# plt.ylim(-1.1, 1.1)
plt.xlabel('Number of steps')
plt.ylabel('Hidden states')

if z_and_r:
    zs = np.array(zs)

    fig = plt.figure()

    x = range(zs.shape[0])
    plt.plot(x, zs[:, 0], linewidth=1)

    plt.ylim(-0.1, 1.1)
    plt.xlabel('Number of steps')
    plt.ylabel('Z')


    rs = np.array(rs)

    fig = plt.figure()

    x = range(rs.shape[0])
    plt.plot(x, rs[:, 0], linewidth=1)

    plt.xlabel('Number of steps')
    plt.ylabel('R')

plt.show()
