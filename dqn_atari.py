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
r_act = env.action_space.sample()

replay_buffer = ReplayBuffer(buffer_size, batch_size)

state, reward, done, _ = env.step(r_act)

for ex in range(batch_size):
    action = r_act
    next_state, reward, done, _ = env.step(action)

    if done:
        next_state = tf.zeros(state.shape)
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

rewards_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer)
    step = 0

    for episode in range(1, n_episodes+1):
        total_reward = 0
        t = 0
        while t < max_steps:
            step += 1

            explore_p = explore_stop + (explore_start-explore_stop)*np.exp(-decay_rate*step)
            if np.rand < explore_p:
                action = r_act
            else:
                Qs = sess.run(dqn.output, feed_dict={dqn.inputs: state.reshape((1, state.shape))})
                action = np.argmax(Qs)

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                next_state = tf.zeros(state.shape)
                exp = (state, action, reward, next_state)
                replay_buffer.add_exp(exp)
                rewards_list.append((episode, total_reward))

                print('Episode: {}'.format(episode),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_p))

                t = max_steps

                env.reset()
                state, reward, done, _ = env.step(r_act)
            else:
                exp = (state, action, reward, next_state)
                replay_buffer.add_exp(exp)
                t += 1
                state = next_state

            batch = replay_buffer.sample()
            states = np.array([ex[0] for ex in batch])
            actions = np.array([ex[1] for ex in batch])
            rewards = np.array([ex[2] for ex in batch])
            next_states = np.array([ex[3] for ex in batch])

            target_Qs = sess.run(dqn.output, feed_dict={dqn.inputs: next_states})
            episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            target_Qs[episode_ends] = (0, 0, 0)

            targets = rewards + gamma * np.max(target_Qs, axis=1)

            loss, _ = sess.run([less, opt], feed_dict={dqn.inputs: states,
                                                       dqn.targetQs: targets,
                                                       dqn.actions: actions})

import matplotlib.pyplot as plt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

eps, rews = np.array(rewards_list).T
smoothed_rews = running_mean(rews, 10)
plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
plt.plot(eps, rews, color='grey', alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
