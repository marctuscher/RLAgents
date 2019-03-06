import roboschool
import gym
from tqdm import tqdm
from RLAgents.VanillaPG.policy import Policy
import numpy as np
from tensorboardX import SummaryWriter
from RLAgents.utils import get_cool_looking_datestring
import torch

class VanillaPGAgent():

    def __init__(self, env_name, gamma, max_ep_steps, train_steps):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.policy = Policy(self.env, 256 , 1e-4)
        self.gamma = gamma
        self.max_ep_steps = max_ep_steps
        self.train_steps = train_steps
        self.writer = SummaryWriter('logs/oneStepAC/'+ env_name + get_cool_looking_datestring())

    def train(self, episodes):
        for i in tqdm(range(episodes)):
            rewards = []
            states = []
            r = 0
            ob = self.env.reset()
            done = False
            for j in range(self.max_ep_steps):
                action = self.policy.action(np.array([ob]))
                states.append(ob)
                # self.env.env_step() for roboschool envs
                ob_, reward, done, _ = self.env.step(action)
                r += reward
                rewards.append(reward)
                if done:
                    value_loss, policy_loss = self.policy.train_policy(rewards, states,self.gamma, 0)
                    self.writer.add_scalar(self.env_name+'/VanillaPG_valueLoss', value_loss , i)
                    self.writer.add_scalar(self.env_name+'/VanillaPG_policyLoss', policy_loss , i)
                    self.writer.add_scalar(self.env_name+'/VanillaPG_reward', r, i)
                    break
                if j == self.train_steps:
                    value_loss, policy_loss = self.policy.train_policy(rewards, states,self.gamma, self.policy.critic(torch.from_numpy(ob_).float().to('cuda:0')).item())
                    self.writer.add_scalar(self.env_name+'/VanillaPG_valueLoss', value_loss , i)
                    self.writer.add_scalar(self.env_name+'/VanillaPG_policyLoss', policy_loss , i)
                    self.writer.add_scalar(self.env_name+'/VanillaPG_reward', r, i)
                    rewards = []
                    states = []
                ob = ob_
        self.writer.close()