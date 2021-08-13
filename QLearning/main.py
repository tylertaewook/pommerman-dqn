# https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env.reset()
# env.observation_space.high = [0.6  0.07] ( max vals for [position, velocity])
# env.action_space.n = 3 (# of actions)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500

# Observation Space; avoided hard-coding so it works for all envs
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
# 'bucket' size; like histogram, we divide the range btw high and low into 20 buckets
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_tab = np.random.uniform(low=-2, high=0, size=DISCRETE_OS_SIZE + [env.action_space.n])  # [20,20,3]

ep_rewards = []
aggr_ep_rewards = dict(ep=[], avg=[], min=[], max=[])

def get_discrete_state(state):
    """
    Returns discrete state from given continuous state
    """
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())  # (6,10); env.reset only returns initial vals
    done = False
    madeit = False

    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_tab[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render() # for speed
        if not done:
            max_future_q = np.max(q_tab[new_discrete_state])
            current_q = q_tab[discrete_state + (action,)]  # get 'the' q-value instead of three
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_tab[discrete_state + (action,)] = new_q
            # Note we're updating Q-value AFTER we took the state. basing on the new state info
        elif new_state[0] >= env.goal_position:
            madeit = True
            print(f"We made it on episode {episode}")
            q_tab[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")

    env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()