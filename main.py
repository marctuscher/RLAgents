from RLAgents.REINFORCE.agent import ReinforceAgent



agent = ReinforceAgent("CartPole-v0", gamma=0.99, max_ep_steps=1000)

agent.train(5000)


