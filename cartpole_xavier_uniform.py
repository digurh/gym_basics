import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Categorical

import gym

torch.manual_seed(0)

env = gym.make('CartPole-v0')
env.seed(1)

# print(env.observation_space)
# print(env.action_space)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, n_hidden_units=16, state_size=2, action_size=1):
        # super(class, self) to make work with python2
        super(Policy, self).__init__()
        self.weights, self.bias = self.params_init(state_size, n_hidden_units, action_size)
        self.parameters = nn.ParameterList([self.weights[w] for w in self.weights.keys()] +
                                           [self.bias[b] for b in self.bias.keys()])

    def forward(self, state):
        l1 = torch.mm(state, self.weights['1']) + self.bias['1']
        l1 = self.relu(l1)
        l2 = torch.mm(l1, self.weights['2']) + self.bias['2']
        return F.softmax(l2, dim=1)

    # turn state into torch usable tensor
    # get action probabilities by sending state through forward pass of network
    # pull distribution from output of network
    # sample and return action and prob of taking action
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    # initialize parameters by wrapping tensors in torch parameters
    def params_init(self, state_size, n_hidden_units, action_size):
        weights = {
            '1': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(state_size, n_hidden_units))),
            '2': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(n_hidden_units, action_size))),
            '3': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(n_hidden_units, action_size)))
        }
        bias = {
            '1': nn.Parameter(torch.zeros([n_hidden_units])),
            '2': nn.Parameter(torch.zeros([action_size])),
            '3': nn.Parameter(torch.zeros([action_size]))
        }
        return weights, bias

    def relu(self, X):
        return F.relu(X)

def lr_decay(opt, ep, lr_init):
    # lr = lr_init * (0.55 ** (ep // 500))
    alpha = (1 - (ep/1000))
    lr = (lr_init * alpha) + (0.001 * alpha)
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def train(net, opt, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, learning_rate=0.0001):
    scores_deque = deque(maxlen=100)
    scores = []

    solved = ''

    lr_init = learning_rate

    for episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()

        for t in range(max_t):
            # env.render()

            # select action by passing current state to net, get back action and
            # log_prob of taking action
            # step environment forward, record reward
            action, log_prob = net.select_action(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)

            # if episode terminated
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # calculate discounted Return
        discount = [gamma**i for i in range(len(rewards)+1)]
        Return = sum([d*r for d,r in zip(discount, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob*Return)
        policy_loss = torch.cat(policy_loss).sum()

        opt.zero_grad()
        policy_loss.backward(retain_graph=True)
        opt.step()

        if episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0 and solved is '':
            solved = 'Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_deque))

        if episode >= 200:
            lr_decay(opt, episode, lr_init)

    return scores, solved


n_hidden_units = 16
state_size = env.observation_space.shape[0]
action_size = 2 # env.action_space.shape[0]

n_episodes = 1000
max_t = 1000
gamma = 1.0
print_every = 100

learning_rate = 0.025


r_net = Policy(n_hidden_units, state_size, action_size).to(device)
opt = optim.Adam(r_net.parameters(), lr=learning_rate)

scores, solved = train(r_net, opt, n_episodes, max_t, gamma, print_every, learning_rate)
print(solved)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env = gym.make('CartPole-v0')

state = env.reset()
for t in range(1000):
    action, _ = r_net.select_action(state)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break

env.close()
