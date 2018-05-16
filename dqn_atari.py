import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym


class DQNAgent:
    def __init__(self):



from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size=10000, batch_size=28):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add_exp(self, experience):
        self.buffer.append(experience)

    def sample(self):
        r_idx = np.random.choice(np.arange(len(buffer)),
                                 size=self.batch_size,
                                 replace=False)
        return [self.buffer[i] for i in r_idx]
