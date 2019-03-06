from RLAgents.REINFORCE.agent import ReinforceAgent
from RLAgents.VanillaPG.agent import VanillaPGAgent



agent = VanillaPGAgent("CartPole-v0", gamma=0.99, max_ep_steps=1000, train_steps=50)

agent.train(50000)


