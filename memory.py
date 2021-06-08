import random


class Memory:
    """Implements the replay buffer."""
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.samples = []

    def add_sample(self, sample):
        """Add a new sample in the buffer."""
        self.samples.append(sample)

        if len(self.samples) > self.max_memory:
            self.samples.pop(0)

    def sample(self, no_samples):
        """Get some number of random samples."""
        if no_samples > len(self.samples):
            return random.sample(self.samples, len(self.samples))

        else:
            return random.sample(self.samples, no_samples)

    def count(self):
        """Get the number of stored samples."""
        return len(self.samples)
