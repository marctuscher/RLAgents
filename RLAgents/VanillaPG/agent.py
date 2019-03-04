import gym
from tqdm import tqdm
from RLAgents.VanillaPG.policy import Policy





class Agent():

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.policy = Policy(self.env)


    def train(self, steps):
        ob = self.env.reset()

        for i in tqdm(steps):

