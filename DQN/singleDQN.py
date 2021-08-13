import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pommerman
import argparse
import random
import gym
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
from utility.replay_buffer import ReplayBuffer2
from utility.utils import featurize

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

    agent_list = [agents.SimpleAgent(), agents.SimpleAgent()]  # placeholder
    env = pommerman.make('OneVsOne-v0', agent_list)

    agent1 = Qnet(env.observation_space.shape[0])  # TODO: assertionerror; not agents.BaseAgent??
    agent2 = agents.SimpleAgent()


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
            episode_reward += (reward[0] + reward[1])
            agent1.memory.append([state_feature, actions, reward, next_state_feature, done])
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

        # agent1.epsdecay()

    train_network(agent1, agent1, train_number=300, batch_size=32)

    env.close()




class Qnet(nn.Module):
    def __init__(self, ncells):
        super(Qnet, self).__init__()

        self.memory = ReplayBuffer2(100000)
        self.effdata = []
        self.score_sum = []
        self.score_init_final = []
        self.ncells = ncells
        self.fc1 = nn.Linear(ncells, 2 * ncells)
        self.fc2 = nn.Linear(2 * ncells, 2 * ncells)
        self.fc3 = nn.Linear(2 * ncells, ncells)
        self.m = nn.LeakyReLU(0.1)

        init_params(self)

    def forward(self, x):
        x = self.m(self.fc1(x))
        x = self.m(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = torch.reshape(obs, (1, self.ncells))
        # print(obs.shape)
        out = self.forward(obs)
        coin = random.random()  # 0<coin<1
        if coin < epsilon:
            return np.random.randint(0, self.ncells)
        else:
            return out.argmax().item()


def init_params(net, val=np.sqrt(2)):
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, val)
            module.bias.data.zero_()
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, val)
            if module.bias is not None:
                module.bias.data.zero_()


def merge_network_weights(q_target_state_dict, q_state_dict, tau):
    dict_dest = dict(q_target_state_dict)
    for name, param in q_state_dict:
        if name in dict_dest:
            dict_dest[name].data.copy_((1 - tau) * dict_dest[name].data
                                       + tau * param)


def train_network(q, memory, q_target, train_number, batch_size, gamma=0.05, double=False):
    optimizer = optim.Adam(q.parameters(), 0.01)

    for i in range(int(train_number)):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        if double:
            max_a_prime = q(s_prime).argmax(1, keepdim=True)
            with torch.no_grad():
                max_q_prime = q_target(s_prime).gather(1, max_a_prime)
        else:
            with torch.no_grad():
                max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = done_mask * (r + gamma * max_q_prime) + (1 - done_mask) * 1 / (1 - gamma) * r
        loss = F.smooth_l1_loss(q_a, target)  # huber loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss  # for logging


if __name__ == '__main__':
    main()
