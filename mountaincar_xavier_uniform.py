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

env = gym.make('MountainCar-v0')
env.seed(1)

print(env.observation_space)
print(env.action_space)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, n_hidden_units=16, state_size=2, action_size=1):
        # super(class, self) to make work with python2
        super(Policy, self).__init__()
        self.weights, self.bias = self.params_init(state_size, n_hidden_units, action_size)
        self.parameters = nn.ParameterList([self.weights[w] for w in self.weights.keys()] +
                                           [self.bias[b] for b in self.bias.keys()])

    def forward(self, state):
        X = torch.mm(state, self.weights['1']) + self.bias['1']
        X = self.relu(X)
        X = torch.mm(X, self.weights['2']) + self.bias['2']
        X = self.relu(X)
        X = torch.mm(X, self.weights['3']) + self.bias['3']
        return F.softmax(X, dim=1)

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
            '2': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(n_hidden_units, n_hidden_units))),
            '3': init.xavier_uniform_(nn.Parameter(torch.FloatTensor(n_hidden_units, action_size)))
        }
        bias = {
            '1': nn.Parameter(torch.zeros([n_hidden_units])),
            '2': nn.Parameter(torch.zeros([n_hidden_units])),
            '3': nn.Parameter(torch.zeros([action_size]))
        }
        return weights, bias

    def relu(self, X):
        return F.relu(X)

# cosine annealing
def cosine_decay(opt, lr_max, lr_min, ep, ds=1000):
    gs = ep
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos((gs / ds) * np.pi))
    for param_group in opt.param_groups:
        param_group['lr'] = lr


def train(net, opt, sch=None, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, lr_max=0.1, lr_min=0.0001):
    scores_deque = deque(maxlen=100)
    scores = []

    solved = ''

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
        if np.mean(scores_deque)>=-100.0 and solved is '':
            solved = 'Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_deque))

        cosine_decay(opt, lr_max, lr_min, episode)

    return scores, solved


n_hidden_units = 32
state_size = env.observation_space.shape[0]
action_size = 3

n_episodes = 1000
max_t = 1000
gamma = 0.99
print_every = 100

lr_max = 0.0006
lr_min = 0.000007


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

env = gym.make('MountainCar-v0')

state = env.reset()
for t in range(1000):
    action, _ = r_net.select_action(state)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break

env.close()
