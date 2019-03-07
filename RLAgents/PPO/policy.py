from torch import nn
from torch.nn import functional as F
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from RLAgents.utils import discount_td, get_space_shape


class SingleHiddenActorNetwork(nn.Module):
    def __init__(self, hidden_size, input_shape, action_size):
        super (SingleHiddenActorNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_shape, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1)


class CriticNetwork(nn.Module):
    def __init__(self, hidden_size, input_shape):
        super (CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_shape, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Policy():

    def __init__(self, env, hidden_size, lr):
        self.env = env

        obs_shape = get_space_shape(self.env.observation_space)
        action_shape = get_space_shape(self.env.action_space)

        self.actor = SingleHiddenActorNetwork(128, obs_shape, action_shape).to("cuda:0")
        self.critic = CriticNetwork(128, obs_shape).to("cuda:0")

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.log_probs = []
        self.values = []
        self.actions = []
        self.entropies = []


    def action(self, ob):
        self.actor.eval()
        self.critic.eval()
        ob = torch.from_numpy(ob).float().to('cuda:0')
        out = self.actor(ob)
        value = self.critic(ob)
        dist = Categorical(out)
        action = dist.sample()

        # take also the log probability of this action for computing the loss
        # later on without having to run forward again.
        self.entropies.append(dist.entropy().item())
        self.log_probs.append(dist.log_prob(action))
        self.actions.append(action)
        self.values.append(value)

        return action.item()


    def train_policy(self, rewards, states, gamma, bootstrap):
        self.actor.train()
        self.critic.train()
        # discount returns
        self.values = torch.cat(self.values)
        advs, rtg = discount_td(np.array(rewards), self.values.data.cpu().squeeze().numpy(), bootstrap, gamma, 0.8)
        advs = torch.cuda.FloatTensor(advs)
        rtg = torch.cuda.FloatTensor(rtg)

        critic_loss = nn.functional.smooth_l1_loss(self.values, rtg)

        actor_loss = torch.cat(self.log_probs) * advs

        self.actor_optimizer.zero_grad()
        actor_loss = -actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.log_probs = []
        self.values = []
        self.entropies = []
        return critic_loss.item(), actor_loss.item()



        