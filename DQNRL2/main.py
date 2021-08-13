# region import
import pommerman
import gym
import torch
import argparse
import numpy as np

try:
    from envs.playground.pommerman import helpers, make, utility

except gym.error.Error as e:
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'Pomme' in env or 'OneVsOne' in env:
            # print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]

    from envs.playground.pommerman import helpers, make, utility

from pommerman import agents
from pommerman.configs import one_vs_one_env

from easydict import EasyDict
from common.util import featurize
from rl2.agents.dqn import DQNModel, DQNAgent
from rl2.workers import MaxStepWorker
from rl2.examples.temp_logger import Logger
# endregion

# region parser
parser = argparse.ArgumentParser(description='DQN pommerman MARL')
parser.add_argument('--episodes', type=int, default=3000, help='episodes')
parser.add_argument('--maxsteps', type=int, default=200, help='maximum steps')
parser.add_argument('--showevery', type=int, default=300, help='report loss every n episodes')

parser.add_argument('--epsilon', type=float, default=0.05, help='parameter for epsilon greedy')
parser.add_argument('--eps_decay', type=float, default=0.995, help='epsilon decay rate')
parser.add_argument('--min_eps', type=float, default=0.05, help='minimum epsilon for decaying')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

parser.add_argument('--capacity', type=int, default=100000, help='capacity for replay buffer')
parser.add_argument('--batch', type=int, default=201, help='batch size for replay buffer')

parser.add_argument('--gpu', type=str, default='0', help='gpu number')

args = parser.parse_args()

# GPU
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
print("GPU using status: ", args.device)


# endregion


def main():
    agent_list = [agents.SimpleAgent(), agents.SimpleAgent()]  # placeholder
    env = pommerman.make('OneVsOne-v0', agent_list)
    env_config = one_vs_one_env()
    observation_shape = env.observation_space.shape
    action_shape = (env.action_space.n,)

    model = DQNModel(observation_shape,
                     action_shape,
                     optim='torch.optim.Adam',
                     defalut=True,
                     )
    agent1 = DQNAgent(model,
                     buffer_size=100000,
                     decay_step=10000,
                     update_interval=1000,
                     # character=Bomber(0, env_config["game_type"])
                     )

    agent2 = agents.SimpleAgent()

    agent_list = [agent1, agent2]
    env = pommerman.make('OneVsOne-v0', agent_list)

    myconfig = {
        'log_dir': './runs',
        'tag': 'DQN/cartpole',
        'log_level': 10
    }

    myconfig = EasyDict(myconfig)
    logger = Logger(name='DEFAULT', args=myconfig)

    worker = MaxStepWorker(env, 1, agent1,
                           max_steps=int(1e6),
                           training=True,
                           logger=logger)
    worker.run()

if __name__ == '__main__':
    main()
