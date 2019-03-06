from torch import nn
from torch.nn import functional as F
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from RLAgents.utils import discount

class Policy(nn.Module):

    def __init__(self, env, hidden_size, lr):
        super(Policy, self).__init__()
        self.env = env
        self.fc1 = nn.Linear(*self.env.observation_space.shape, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, self.env.action_space.n)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.log_probs = []


    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out


    def action(self, ob):
        self.eval()
        ob = torch.from_numpy(ob).float().to('cuda:0')
        out = self.forward(ob)
        dist = Categorical(out)
        action = dist.sample()

        # take also the log probability of this action for computing the loss
        # later on without having to run forward again.
        self.log_probs.append(dist.log_prob(action))
        return action.item()


    def train_policy(self, rewards, gamma):
        self.train()

        # discount returns
        G = torch.tensor(discount(rewards, gamma)).to('cuda:0')
        # always normalize before feeding something into network
        G = (G - G.mean()) / (G.std()+ 1e-6)
        loss = []
        self.log_probs = torch.cat(self.log_probs)
        loss = -self.log_probs * G
        self.optimizer.zero_grad()
        loss = loss.sum()
        loss.backward()
        self.optimizer.step()
        self.log_probs = []
        return loss.item()



        