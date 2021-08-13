import torch
import torch.nn as nn
import torch.optim as optim
from envs.playground.pommerman import characters

import random
import gym
import numpy as np

from envs.playground.pommerman.agents import BaseAgent
from utility.replay_buffer import ReplayBuffer, ReplayBuffer2


class DQNAgent(BaseAgent):
    """
    DQN from scratch
    """
    def __init__(self, env, args, character=characters.Bomber):
        super(DQNAgent, self).__init__(character)
        self.obs_n = env.observation_space.shape[0]  # output_dim
        self.action_n = env.action_space.n  # input_dim

        self.epsilon = args.epsilon
        self.eps_decay = args.eps_decay
        self.min_eps = args.min_eps
        self.gamma = args.gamma
        self.lr = args.lr

        self.episodes = args.episodes
        self.maxsteps = args.maxsteps
        self.showevery = args.showevery

        self.capacity = args.capacity
        self.batch = args.batch
        # self.buffer = ReplayBuffer(self.capacity, self.batch)
        self.buffer = ReplayBuffer2(self.capacity)

        self.model = nn.Sequential(
            nn.Linear(self.obs_n, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_n)
        )

        self.optim = optim.Adam(self.model.parameters(), self.lr)
        self.MSE_loss = nn.MSELoss()

    def forward(self, state):
        state = torch.FloatTensor(state)
        qvals = self.model(state)
        # qvals (20,2,6) state (20,2,201)
        return qvals

    def act(self, obs):
        return np.argmax(self.forward(obs))

    def update(self, gamma, batch_size):
        batch = self.buffer.sample(batch_size)

        states, actions, rewards, next_states, done = batch

        curr_Q = self.forward(states)
        next_Q = self.forward(next_states)

        # expected_Q = rewards + self.gamma * torch.max(next_Q, 1)
        expected_Q = rewards + gamma * (done) * np.max(self.forward(next_states))

        max_q_prime = next_Q.max(1)[0].unsqueeze(1)
        # expected_Q = done * (rewards + gamma * max_q_prime) + (1 - done) * 1 / (1 - gamma) * rewards
        # expected_Q = done * (rewards + gamma * max_q_prime) + 1 / (1 - gamma) * rewards
        loss = self.MSE_loss(curr_Q, expected_Q) # TODO: try Huber Loss later too

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def epsdecay(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_eps else self.epsilon
