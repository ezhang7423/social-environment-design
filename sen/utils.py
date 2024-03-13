# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PettingZoo interface to meltingpot environments."""

import functools

import matplotlib.pyplot as plt
from gymnasium import utils as gym_utils
from meltingpot import substrate
from meltingpot.examples.gym import utils
from ml_collections import config_dict
from pettingzoo import utils as pettingzoo_utils
from pettingzoo.utils import wrappers

PLAYER_STR_FORMAT = "player_{index}"
MAX_CYCLES = 5000


def parallel_env(env_config, max_cycles=MAX_CYCLES, principal=None):
    return _ParallelEnv(env_config, max_cycles, principal)


def raw_env(env_config, max_cycles=MAX_CYCLES):
    return pettingzoo_utils.parallel_to_aec_wrapper(
        parallel_env(env_config, max_cycles)
    )


def env(env_config, max_cycles=MAX_CYCLES):
    aec_env = raw_env(env_config, max_cycles)
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


def timestep_to_observations(timestep):
    gym_observations = {}
    for index, observation in enumerate(timestep.observation):
        gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
            key: value for key, value in observation.items() if "WORLD." not in key
        }
    nearby_observations = {}
    for index in range(len(timestep.observation)):
        nearby = timestep.observation[index]["NEARBY"]
        nearby[index] = 0  # players shouldn't count themselves as nearby
        nearby_observations[PLAYER_STR_FORMAT.format(index=index)] = nearby
    return gym_observations, nearby_observations, timestep.observation[0]["WORLD.RGB"]


class _MeltingPotPettingZooEnv(pettingzoo_utils.ParallelEnv):
    """An adapter between Melting Pot substrates and PettingZoo's ParallelEnv."""

    def __init__(self, env_config, max_cycles, principal=None):
        self.env_config = config_dict.ConfigDict(env_config)
        self.max_cycles = max_cycles
        if principal is None:
            self._env = substrate.build_from_config(
                self.env_config, roles=self.env_config.default_player_roles
            )
        else:
            self._env = substrate.build_principal_from_config(
                self.env_config,
                roles=self.env_config.default_player_roles,
                principal=principal,
            )
        self._num_players = len(self._env.observation_spec())
        self.possible_agents = [
            PLAYER_STR_FORMAT.format(index=index) for index in range(self._num_players)
        ]
        observation_space = utils.remove_world_observations_from_space(
            utils.spec_to_space(self._env.observation_spec()[0])
        )
        self.observation_space = functools.lru_cache(maxsize=None)(
            lambda agent_id: observation_space
        )
        action_space = utils.spec_to_space(self._env.action_spec()[0])
        self.action_space = functools.lru_cache(maxsize=None)(
            lambda agent_id: action_space
        )
        self.state_space = utils.spec_to_space(
            self._env.observation_spec()[0]["WORLD.RGB"]
        )

    def state(self):
        return self._env.observation()

    def reset(self, seed=None, options=None):
        """See base class."""
        timestep = self._env.reset()
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        observations, nearby_obs, world_obs = timestep_to_observations(timestep)

        return observations, {
            agent: ({}, world_obs, nearby_obs[agent]) for agent in self.agents
        }

    def step(self, action):
        """See base class."""
        actions = [action[agent] for agent in self.agents]
        timestep = self._env.step(actions)
        rewards = {
            agent: timestep.reward[index] for index, agent in enumerate(self.agents)
        }
        self.num_cycles += 1
        done = timestep.last() or self.num_cycles >= self.max_cycles
        dones = {agent: done for agent in self.agents}
        observations, nearby_obs, world_obs = timestep_to_observations(timestep)
        infos = {agent: ({}, world_obs, nearby_obs[agent]) for agent in self.agents}
        # infos = {agent: {} for agent in self.agents}

        if done:
            self.agents = []
        return observations, rewards, dones, dones, infos

    def close(self):
        """See base class."""
        self._env.close()

    def render(self, mode="not human", filename=None):
        print(len(self.state()))
        rgb_arr = self.state()[0]["WORLD.RGB"]
        if mode == "human":
            plt.cla()
            plt.imshow(rgb_arr, interpolation="nearest")
            if filename is None:
                plt.show(block=False)
            else:
                plt.savefig(filename)
            return None
        return rgb_arr


class _ParallelEnv(_MeltingPotPettingZooEnv, gym_utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_config, max_cycles, principal=None):
        gym_utils.EzPickle.__init__(self, env_config, max_cycles)
        _MeltingPotPettingZooEnv.__init__(self, env_config, max_cycles, principal)
