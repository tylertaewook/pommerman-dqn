import pommerman
import gym
import torch
import argparse
import random
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
from agent import DQNAgent
from utility.utils import featurize, CustomEnvWrapper


def main():
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

    config = one_vs_one_env()
    env = CustomEnvWrapper(config)

    agent_list = [agents.SimpleAgent(), agents.SimpleAgent()]  # placeholder
    env = pommerman.make('OneVsOne-v0', agent_list)

    agent1 = DQNAgent(env, args)  # TODO: assertionerror; not agents.BaseAgent??
    agent2 = agents.SimpleAgent()

    agent_list = [agent1, agent2]
    env = pommerman.make('OneVsOne-v0', agent_list)

    episode_rewards = []
    action_n = env.action_space.n

    for episode in range(args.episodes):
        states = env.reset()
        state_feature = featurize(env, states)
        done = False
        episode_reward = 0
        for step in range(args.maxsteps):
            # if agent1.epsilon > random.random():
            #     action = random.randint(0, action_n - 1)
            # else:
            #     action = agent1.act(state_feature[0])

            # env.render()

            actions = env.act(states)
            # TODO: env.set_training_agent(agents[-1].agent_id)으로 training agent를 명시
            # 그리고 위에 env.act(states)에서 training agent action만 따로 append하기

            next_state, reward, done, info = env.step(actions)  # n-array with action for each agent
            next_state_feature = featurize(env, next_state)
            episode_reward += (reward[0]+reward[1])
            agent1.buffer.append([state_feature, actions, reward, next_state_feature, done])
            # env.get_observation -> 찾아보기 (forward model)

            # if len(agent1.buffer) > args.batch:
            #     agent1.update(args.gamma, args.batch)
            # if agent1.buffer.size() > args.batch:
            #     agent1.update(args.gamma, args.batch)

        if done:
            episode_rewards.append(episode_reward)
        if episode % args.showevery == 0:
            print(f"Episode: {episode + 1:2d} finished, result: {'Win' if 0 in info.get('winners', []) else 'Lose'}")
            print(f"Avg Episode Reward: {np.mean(episode_rewards)}")

        agent1.epsdecay()

    env.close()

    # TODO: Implement Target Network


if __name__ == '__main__':
    main()
