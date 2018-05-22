import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym


class DQNAgent:
    def __init__(self, learning_rate=0.0001, state_size=6, action_size=3,
                 hidden_size=32):
        self.inputs = tf.placeholder(tf.float32, [None, state_size], name='inputs
        self.actions = tf.placeholder(tf.int32, [None])
        one_hot_actions = tf.one_hot(self.actions, action_size)
        self.targetQs = tf.placeholder(tf.float32, [None], name='targetQs')

        self.fc1 = tf.contrib.layers.fully_connected(self.inputs, hidden_size)
        self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)
        self.output = tf.contrib.layers.fully_connected(self.fc2, action_size,
                                                        activation=None)

        self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.targetQs - self.Q))
        self.opt = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)


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





env = gym.make('Acrobot-v1')
