from typing import Dict, List
import numpy as np
from pommerman.agents import BaseAgent
from pommerman.envs.v0 import Pomme


class CustomAgent(BaseAgent):
    def act(self, *args):
        pass


class CustomEnvWrapper(Pomme):
    def __init__(self, config) -> None:
        super().__init__(**config['env_kwargs'])
        self.seed(0)
        agents = {}
        for agent_id in range(4):
            agents[agent_id] = CustomAgent(
                config["agent"](agent_id, config["game_type"]))
        self.set_agents(list(agents.values()))
        self.set_init_game_state(None)
        view_range = 2 * self._agent_view_size + 1
        locational_shape = (5, view_range, view_range)
        additional_shape = (8,)
        self.observation_shape = (locational_shape, additional_shape)

    def reset(self):
        obs = super().reset()
        obs = self._preprocessing(obs)

        return obs

    def step(self, acs):
        obs, reward, done, info = super().step(acs)
        info['original_obs'] = obs
        obs = self._preprocessing(obs)

        return obs, reward, done, info

    def get_global_obs(self):
        obs = self.model.get_observations(curr_board=self._board,
                                          agents=self._agents,
                                          bombs=self._bombs,
                                          flames=self._flames,
                                          is_partially_observable=False,
                                          agent_view_size=self._agent_view_size,
                                          game_type=self._game_type,
                                          game_env=self._env)
        obs = self._preprocessing(obs)

        return obs

    def _preprocessing(self, obs: List[Dict], **kwargs) -> List[Dict]:
        out = []
        for d in obs:
            custom_obs = {}
            keys = ['alive', 'game_type', 'game_env']
            _ = list(map(d.pop, keys))  # remove useless obs

            # Change enums into int
            d.update({'teammate': d.get('teammate').value})
            enemies = list(map(lambda x: x.value, d.get('enemies')))
            enemies.remove(9)  # Remove dummy agent from enemies list
            d.update({'enemies': enemies})

            # Gather infos
            locational = []
            additional = []
            for k, v in d.items():
                if hasattr(v, 'shape'):
                    # Make border walls for locational obs
                    # obs['board'] borders are represented as 2(= Rigid wall)
                    # else borders are filled with 0 values.
                    if k != 'board':
                        for _ in range(self._agent_view_size):
                            v = np.insert(v, (0, v.shape[0]), 0, axis=0)
                            v = np.insert(v, (0, v.shape[1]), 0, axis=1)
                    else:
                        for _ in range(self._agent_view_size):
                            v = np.insert(v, (0, v.shape[0]), 2, axis=0)
                            v = np.insert(v, (0, v.shape[1]), 2, axis=1)

                    # Cut views by centering the agent for localized observation
                    if not kwargs.setdefault('global', False):
                        pos = np.array(d.get('position'))
                        view_range = 2 * self._agent_view_size + 1
                        v = v[pos[0]:pos[0] + view_range,
                              pos[1]:pos[1] + view_range]

                    locational.append(v)

                else:
                    if hasattr(v, '__iter__'):
                        additional += v
                    else:
                        additional.append(v)

            custom_obs.update({'locational': np.stack(locational),
                               'additional': np.array(additional,
                                                      dtype='float64')})

            out.append(custom_obs)

        return out



def featurize(env, states):
    '''
    Converts the states(dict) into list of 1D numpy arrays

    Input:
    - env: gym environment
    - states: list[num_agents, dict(15)] for each agent
    Output:
    - feature: list[num_agents, 372]
    '''
    feature = []
    for state in states:
        feature.append(env.featurize(state).tolist())
        # changes to 1D numpy array
    return feature
