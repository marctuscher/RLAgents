import roboschool
import gym
from tqdm import tqdm
from RLAgents.REINFORCE.policy import Policy
import numpy as np
from tensorboardX import SummaryWriter
from RLAgents.utils import get_cool_looking_datestring

class ReinforceAgent():

    def __init__(self, env_name, gamma, max_ep_steps):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.policy = Policy(self.env, 128 , 1e-4).to('cuda:0')
        self.gamma = gamma
        self.max_ep_steps = max_ep_steps

        self.writer = SummaryWriter('logs/reinforce/'+ env_name + get_cool_looking_datestring())

    def train(self, episodes):
        for i in tqdm(range(episodes)):
            rewards = []
            r = 0
            ob = self.env.reset()
            done = False
            for j in range(self.max_ep_steps):
                action = self.policy.action(np.array([ob]))
                
                # self.env.env_step() for roboschool envs
                ob, reward, done, _ = self.env.step(action)
                r += reward
                rewards.append(reward)
                if done:
                    loss = self.policy.train_policy(rewards, self.gamma)
                    self.writer.add_scalar(self.env_name+'/reinforce_loss', loss , i)
                    self.writer.add_scalar(self.env_name+'/reinforce_reward', r, i)
                    break
        self.writer.close()
