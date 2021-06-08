from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import math
import numpy as np
import matplotlib.pyplot as plt

from cells import *
from model import CustomModel, DoubleLayerModel
from memory import Memory
from gamerunner import GameRunner
from trainer import Trainer
from tmaze import TMaze


# Function called during training to get the current learning rate.
def get_learning_rate():
    return min_lr + (max_lr - min_lr) * math.exp(-decay_lr * i)


# This script implements a train session on the T-maze environment.
# It is composed of some number of iterations. At each iteration, the model
# plays some number of games and then is trained for some epochs on the replay
# buffer.


# Length of the corridor
corridor_length = 10


# Choose which model to use
Model = CustomModel
# Model = DoubleLayerModel

# Choose the cell type and the number of units

cell = GruCellLayer
nb_units = 32

# cell = LSTMCellLayer
# nb_units = 32

# cell = BistableRecurrentCellLayer
# nb_units = 128

# cell = NeuromodulatedBistableRecurrentCellLayer
# nb_units = 32

# cell = JANETCellLayer
# nb_units = 32

# Start with random weights or load from a file
weights_load_file = None
# weights_load_file = 'weights/gru/file'

# Save or not the weights at the end of the training
# weights_save_file = None
weights_save_file = 'weights/gru/file'


# Training parameters

# Number of iterations of the training algorithm
iters = 200
# Number of sequences in a batch
batch_size = 32

# Set to 'True' to warmup the model
warmup = True
# Number of iterations of the warmup
warmup_iters = 300
# Maximum warmup length
max_length = 5000

# Number of games at each iteration
games = 32
# Maximum number of sequences retained in the replay buffer
memory_size = 512
# Number of epochs at each iteration
epochs = 10
# Discount factor
gamma = 0.9

# Parameters related to the learning rate
max_lr = 0.001
min_lr = 0.0001
decay_lr = 0.0

# Parameters related to epsilon (for the exploration policy)
max_eps = 0.9
min_eps = 0.01
decay_eps = 0.0005


# Create environment
env = TMaze(corridor_length)

# Create testing environments
test_env_up = TMaze(corridor_length, init=1)
test_env_down = TMaze(corridor_length, init=0)


num_states = env.num_states
num_actions = env.num_actions

# Create model
model = Model(
    num_states,
    num_actions,
    cell,
    nb_units=nb_units,
    weights_load_file=weights_load_file,
    lr=get_learning_rate
)

# Create replay buffer
memory = Memory(memory_size)

# Create the game runners
game = GameRunner(
    model,
    env,
    memory,
    max_eps,
    min_eps,
    decay_eps,
    max_steps=10*corridor_length
)

test_game_up = GameRunner(
    model,
    test_env_up,
    max_steps=corridor_length+10
)

test_game_down = GameRunner(
    model,
    test_env_down,
    max_steps=corridor_length+10
)

# Create trainer
trainer = Trainer(model, memory, batch_size, gamma)

# Create buffers for generating graphs
rewards = np.zeros((games))
means = []
stddevs = []
vars = []

max_reward = -0.1 * (corridor_length-1) + corridor_length

# First fill the replay number with a number of samples equal to the batch size
print('======================== PREGENERATION ========================')
for g in range(batch_size):
    r = game.run()/corridor_length * 10
    print('Game: {:4}/{}    reward: {: >5.1f}    eps: {:>5.3f}    reward: {}'.format(
        g+1, 1 * batch_size, r, game.eps, 'Up' if env.reward_is_up else 'Down')
    )

if warmup and iters != 0:
    print('\n=========================== WARMUP ============================')
    vars_warmup = model.warmup_multistability(memory, batch_size, warmup_iters, max_length)

i = 0
# Start the iterations
for _ in range(iters):
    print('\n======================== ITER {:>4}/{} ========================\n' \
          .format(i+1, iters)
    )

    samples = memory.sample(batch_size)
    inputs = [s[0] for s in samples]
    v = model.check_bistability(inputs, length=max_length)
    vars.append(v)
    print('Variance: {:>4.3f}'.format(v))

    print('--------------------------- Playing ---------------------------')

    # Play some games
    for g in range(games):
        rewards[g] = game.run()/corridor_length * 10
        print('Game: {:4}/{}    reward: {: >5.1f}    eps: {:>5.3f}    reward: {}'.format(
            g+1, games, rewards[g], game.eps, 'Up' if env.reward_is_up else 'Down')
        )

    # Buffer results
    means.append(rewards.mean())
    stddevs.append(rewards.std())

    print('\n--------------------------- Training --------------------------')

    # Train model
    trainer.train(epochs)

    i += 1

    # Every 10 iterations, check if model obtains the best rewards both when
    # the reward is up and when it is down.
    if i%10 == 0:
        print('\n--------------------------- Testing ---------------------------')
        reward_up = test_game_up.run()
        reward_down = test_game_down.run()

        print('Up env: Obtained reward:', reward_up, 'Max reward:', max_reward)
        print('Down env: Obtained reward:', reward_down, 'Max reward:', max_reward)

# Save weights
if weights_save_file is not None:
    model.save_weights(weights_save_file)

# Create graph
x = range(len(means))

plt.figure()
plt.errorbar(x, means, yerr=stddevs, linewidth=2, elinewidth=0.5)
plt.xlabel('Number of iterations')
plt.ylabel('Mean reward')
title = ('corridor\_length = {}, iters = {}, memory\_size = {}, '
         'batch\_size = {}, games = {}, epochs = {}')
plt.title(title.format(corridor_length, iters, memory_size,
                       batch_size, games, epochs))

if warmup and iters != 0:
    plt.figure()
    plt.plot(range(len(vars_warmup)), vars_warmup, linewidth=1)
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean variance of the stable states')
    plt.title('Evolution of variance during Warmup')

plt.figure()
plt.plot(range(len(vars)), vars, linewidth=1)
plt.xlabel('Number of iterations')
plt.ylabel('Mean variance of the stable states')
plt.title('Evolution of variance during Training')

plt.show()
