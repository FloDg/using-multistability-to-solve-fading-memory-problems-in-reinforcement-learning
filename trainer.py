import numpy as np


class Trainer:
    def __init__(self, model, memory, batch_size, gamma):
        """
        Aims at training a given model with a given replay buffer using
        Q-learning.
        """
        self.model = model
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma

    def train(self, epochs=1):
        """Train the model."""
        # Get the samples
        count = self.memory.count()
        batch = self.memory.sample(count - count%self.batch_size)

        # Isolate the states.
        states_sequences = [val[0] for val in batch]

        # Compute the Q values.
        qsa = self.model.predict_batch(states_sequences).to_list()

        # Update the Q values.
        for i, b in enumerate(batch):
            states_seq, actions_seq, rewards_seq = b[0], b[1], b[2]

            q_seq = qsa[i]
            for j, (s, a, r) in enumerate(zip(states_seq, actions_seq, rewards_seq)):
                if j == len(states_seq)-1:
                    new_q = r

                else:
                    new_q = r + self.gamma * max(q_seq[j+1])

                # Soft update of the Q value
                q_seq[j][a] = 0.3 * q_seq[j][a] + 0.7 * new_q

        # Train the model to predict the new Q values.
        self.model.train(states_sequences, qsa, epochs, self.batch_size)
