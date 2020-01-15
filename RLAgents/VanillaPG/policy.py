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
        self.relu = nn.ReLU6()
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(action_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.batchnorm(out)
        out = self.fc2(out)
        out = self.batchnorm2(out)
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

        self.actor = SingleHiddenActorNetwork(256, obs_shape, action_shape).to("cuda:0")
        self.critic = CriticNetwork(256, obs_shape).to("cuda:0")
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=5e-4)
        self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=1e-3)



    def action(self, ob):
        self.actor.eval()
        self.critic.eval()
        ob = torch.from_numpy(ob).float().to('cuda:0')
        prob_actor = self.actor(ob).clamp(0,1)
        prob, sample = prob_actor.max(dim=1)
        value = self.critic(ob)
        log = torch.log(prob)
        entropy = -prob * log
        # take also the log probability of this action for computing the loss
        # later on without having to run forward again.
        return sample.item(), log, value, entropy


    def train_policy(self, rewards, states, values, log_probs, entropies,gamma, bootstrap):
        self.actor.train()
        self.critic.train()
        # discount returns
        values = torch.cat(values)
        entropies = torch.cat(entropies)
        advs, rtg = discount_td(np.array(rewards), values.data.cpu().squeeze().numpy(), bootstrap, gamma, 0.99)
        advs = torch.cuda.FloatTensor(advs)
        rtg = torch.cuda.FloatTensor(rtg)

        advs = (advs - advs.mean()) / advs.std() + np.finfo(np.float32).eps

        critic_loss = nn.functional.mse_loss(values, rtg)

        actor_loss = torch.cat(log_probs)  * advs + 0.1 * entropies

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



        