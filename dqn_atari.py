import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym


class DQNAgent:
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, state_size], name='inputs')
        self.targetQs = tf.placeholder(tf.float32, [None], name='targetQs')

        self.conv1 = tf.contrib.layers.conv2d()
        self.conv2 = tf.contrib.layers.conv2d()
        self.output = tf.contrib.layers.fully_connected()

        


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
