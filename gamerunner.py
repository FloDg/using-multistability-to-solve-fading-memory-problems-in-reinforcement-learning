import numpy as np
import random
import math


class GameRunner:
    """
    Run episodes on a given environment with a given model.
    Can store the history of the episode in a replay buffer.
    Also implement a epsilon-greedy policy, where actions are taken randomly with
    a probability of 'eps', whose value is updated at each new episode.
    """

    def __init__(self, model, env, memory=None, max_eps=0, min_eps=0, decay=0,
                 render=False, max_steps=100, print_actions=False, print_qvalues=False):
        """Initialize a new GameRunner."""
        self.model = model
        self.memory = memory
        self.env = env
        self.render = render

        self.max_eps = max_eps
        self.min_eps = min_eps
        self.decay = decay

        self.games = 0
        self.max_steps = max_steps

        self.print_actions = print_actions
        self.print_qvalues = print_qvalues

    def run(self):
        """Run a game on the environment."""
        i = 0
        state = self.env.reset()
        tot_reward = 0

        # Retain all the states, rewards and actions which have occured.
        state_buffer = []
        reward_buffer = []
        action_buffer = []

        # Update 'eps'
        self.eps = self.min_eps + (self.max_eps - self.min_eps) \
            * math.exp(-self.decay * self.games)

        self.games += 1

        if self.render:
            self.env.render_now()

        # Play the game.
        while True:
            action = self.choose_action(state)

            if self.print_actions is True:
                print(i, state, action)

            next_state, reward, done = self.env.step(action)
            if self.render:
                self.env.render_now()

            state_buffer.append(state)
            reward_buffer.append(reward)
            action_buffer.append(action)

            tot_reward += reward
            state = next_state

            if done or i == self.max_steps - 1:
                if self.memory is not None:
                    # Save the episode in the replay buffer
                    self.memory.add_sample(
                        (state_buffer, action_buffer, reward_buffer))

                self.model.reset_state()
                return tot_reward

            i += 1

    def choose_action(self, state):
        """
        Choose a new action, either at random with probability 'eps', or
        predicted by the model.
        """
        out = self.model.predict(state)

        if random.random() < self.eps:
            return random.randint(0, self.model.num_actions - 1)

        if self.print_qvalues:
            print(out)

        return np.argmax(out)
