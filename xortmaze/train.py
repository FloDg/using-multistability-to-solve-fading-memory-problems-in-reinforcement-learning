from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import math
import random
import numpy as np
import matplotlib.pyplot as plt

from cells import *
from model import CustomModel
from memory import Memory
from gamerunner import GameRunner
from trainer import Trainer
from xortmaze import XorTMaze


# Function called during training to get the current learning rate.
def get_learning_rate():
    return min_lr + (max_lr - min_lr) * math.exp(-decay_lr * i)


# This script implements a train session on the Xor-T-maze environment.
# It is composed of some number of iterations. At each iteration, the model
# plays some number of games and then is trained for some epochs on the replay
# buffer.


# Length of the corridor
corridor_length = 10
# List of the positions at which the second indication shoudl be given
middles = [5]


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


# Create environments
envs = []
for m in middles:
    envs.append(XorTMaze(corridor_length, m))

# Create testing environments
test_names = ['Down 1-2',
              'Down 2-1',
              '  Up 1-1',
              '  Up 2-2']

test_envs = []
for i in range(4):
    test_envs.append(XorTMaze(corridor_length, middles[0], init=i))


num_states = envs[0].num_states
num_actions = envs[0].num_actions

# Create model
model = CustomModel(
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
runners = []
for e in envs:
    runners.append(GameRunner(
        model,
        e,
        memory,
        max_eps,
        min_eps,
        decay_eps,
        max_steps=10*corridor_length
    ))

test_games = []
for e in test_envs:
    test_games.append(GameRunner(
        model,
        e,
        max_steps=corridor_length+10
    ))

# Create trainer
trainer = Trainer(model, memory, batch_size, gamma)

# Create buffers for generating graphs
rewards = np.zeros((games))
means = []
stddevs = []

max_reward = -0.1 * (corridor_length-1) + corridor_length

# First fill the replay number with a number of samples equal to the batch size
print('======================== PREGENERATION ========================')
for g in range(batch_size):
    game = random.choice(runners)
    r = game.run()/corridor_length * 10
    print('Game: {:4}/{}    reward: {: >5.1f}    eps: {:>5.3f}    reward: {}'.format(
        g+1, batch_size, r, game.eps, 'Up' if game.env.reward_is_up else 'Down')
    )

i = 0
# Start the iterations
for _ in range(iters):
    print('\n======================== ITER {:>4}/{} ========================\n' \
          .format(i+1, iters)
    )

    print('--------------------------- Playing ---------------------------')

    # Play some games
    for g in range(games):
        game = random.choice(runners)
        rewards[g] = game.run()/corridor_length * 10
        print('Game: {:4}/{}    reward: {: >5.1f}    eps: {:>5.3f}    reward: {}'.format(
            g+1, games, rewards[g], game.eps, 'Up' if game.env.reward_is_up else 'Down')
        )

    # Buffer results
    means.append(rewards.mean())
    stddevs.append(rewards.std())

    print('\n--------------------------- Training --------------------------')

    # Train model
    trainer.train(epochs)

    i += 1

    # Every 10 iterations, check if model obtains the best rewards on the test
    # envs.
    if i%10 == 0:
        print('\n--------------------------- Testing ---------------------------')
        best_achieved = 0
        for g, n in zip(test_games, test_names):
            rew = g.run()

            print('{}: {}.'.format(n, rew))

            if abs(rew - max_reward) < 0.0001:
                best_achieved += 1

# Save weights
if weights_save_file is not None:
    model.save_weights(weights_save_file)

# Create graph
x = range(len(means))

fig = plt.figure()

plt.errorbar(x, means, yerr=stddevs, linewidth=2, elinewidth=0.5)

plt.xlabel('Number of iterations')
plt.ylabel('Mean reward')

title = ('corridor\_length = {}, iters = {}, memory\_size = {}, '
         'batch\_size = {}, games = {}, epochs = {}')
plt.title(title.format(corridor_length, iters, memory_size,
                       batch_size, games, epochs))

plt.show()
