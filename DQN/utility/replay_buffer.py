import numpy as np
import random
import collections
import torch

Experience = collections.namedtuple('Experience',
                                    field_names=['state', 'action', 'reward',
                                                 'done', 'new_state'])


class ReplayBuffer(object):
    """
    Implementing DQN's Replay Buffer from scratch
    """

    def __init__(self, capacity, batch_size):
        self.buffer = collections.deque(maxlen=capacity)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer.maxlen:
            self.buffer.popleft()

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, done_batch, next_state_batch)

    def sample2(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

class ReplayBuffer2():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_list, done_mask_list = [], [], [], [], []

        for transition in mini_batch:   #transition: tuple
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_list.append(s_prime)
            done_mask_list.append([done_mask])

        return (torch.tensor(s_lst, dtype=torch.float),
                torch.tensor(a_lst), torch.tensor(r_lst),
                torch.tensor(s_prime_list, dtype=torch.float),
                torch.tensor(done_mask_list))

    def size(self):
        return len(self.buffer)