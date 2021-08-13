
# from envs.playground.pommerman import characters
# uncommenting this gives error...why? idk


from pommerman.agents import BaseAgent
from pommerman import constants

import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


class dqnBaseline(BaseAgent):
    """
    Baseline Agent using DQN from keras-rl
    """

    def __init__(self, featureSpaceDim, *args, **kwargs):
        super(dqnBaseline, self).__init__(*args, **kwargs)

        self.observation_dimension = featureSpaceDim

        model = Sequential()
        model.add(Flatten(input_shape= (1,) + self.observation_dimension))  #(1,)))#observation_space.shape))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(len(constants.Action)))
        model.add(Activation('softmax'))
        print(model.summary())

        memory = SequentialMemory(limit=50000, window_length=1)
        policy = EpsGreedyQPolicy()

        dqn = DQNAgent(model=model, nb_actions=len(constants.Action), memory=memory, nb_steps_warmup=0,
                       target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        self.dqn = dqn

    def act(self, obs, action_space=len(constants.Action)):
        state = self.feature_builder(obs)
        action = self.dqn.forward(state)
        #action = np.random.randint(0,len(constants.Action))
        return action

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent know that
        the episode has ended and what is the reward.

        Args:
          reward: The single reward scalar to this agent.
        """
        pass

    def feature_builder(self, obs):
        state = np.zeros(self.observation_dimension)
        state[0] = obs[0]['can_kick']
        assert state.shape == self.observation_dimension
        return state



