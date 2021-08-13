import gym
from rl2.agents.dqn import DQNModel, DQNAgent
from rl2.workers.base import MaxStepWorker

# Create env
env = gym.make('CartPole-v0')

# Create default DQN model
model = DQNModel(
    env.observation_space.shape,
    env.action_space.n,
    default=True
)

# Create default DQN agent
agent = DQNAgent(model)

# Create and run max step worker
worker = MaxStepWorker(env, agent, max_steps=int(1e6), training=True)
worker.run()
