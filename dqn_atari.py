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


buffer_size = 10000
batch_size = 28

n_episodes = 1000
max_steps = 200
gamma = 0.99

learning_rate = 0.0001
hidden_size = 64

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001


env = gym.make('Acrobot-v1')
env.reset()
r_act = env.step(env.action_space.sample())

replay_buffer = ReplayBuffer(buffer_size, batch_size)

state, reward, done, _ = env.step(r_act)

for ex in range(batch_size):
    action = r_act
    next_state, reward, done, _ = env.step(action)

    if done:
        next_state = tf.zeros(state.shape[0])
        exp = (state, action, reward, next_state)
        replay_buffer.add_exp(exp)

        env.reset()
        state, reward, done, _ = env.step(r_act)
    else:
        exp = (state, action, reward, next_state)
        replay_buffer.add_exp(exp)
        state = next_state

tf.reset_default_graph()
dqn = DQNAgent(learning_rate, hidden_size=n_hidden_layer)
