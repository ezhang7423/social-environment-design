import warnings

import cloudpickle
import gymnasium
import gymnasium.vector
import numpy as np
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from pettingzoo.utils.env import ParallelEnv


def vec_env_args(env, num_envs):
    def env_fn():
        env_copy = cloudpickle.loads(cloudpickle.dumps(env))
        return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space


def sb3_concat_vec_envs_v1(vec_env, num_vec_envs, num_cpus=0):
    num_cpus = min(num_cpus, num_vec_envs)
    vec_env = ConcatVecEnv(*vec_env_args(vec_env, num_vec_envs))
    return SB3VecEnvWrapper(vec_env)


def pettingzoo_env_to_vec_env_v1(parallel_env):
    assert isinstance(
        parallel_env, ParallelEnv
    ), "pettingzoo_env_to_vec_env takes in a pettingzoo ParallelEnv. Can create a parallel_env with pistonball.parallel_env() or convert it from an AEC env with `from pettingzoo.utils.conversions import aec_to_parallel; aec_to_parallel(env)``"
    assert hasattr(
        parallel_env, "possible_agents"
    ), "environment passed to pettingzoo_env_to_vec_env must have possible_agents attribute."
    return MarkovVectorEnv(parallel_env)


class MarkovVectorEnv(gymnasium.vector.VectorEnv):
    def __init__(self, par_env, black_death=False):
        """
        parameters:
            - par_env: the pettingzoo Parallel environment that will be converted to a gymnasium vector environment
            - black_death: whether to give zero valued observations and 0 rewards when an agent is done, allowing for environments with multiple numbers of agents.
                            Is equivalent to adding the black death wrapper, but somewhat more efficient.

        The resulting object will be a valid vector environment that has a num_envs
        parameter equal to the max number of agents, will return an array of observations,
        rewards, dones, etc, and will reset environment automatically when it finishes
        """
        self.par_env = par_env
        self.metadata = par_env.metadata
        self.render_mode = par_env.unwrapped.render_mode
        self.observation_space = par_env.observation_space(par_env.possible_agents[0])
        self.action_space = par_env.action_space(par_env.possible_agents[0])
        assert all(
            self.observation_space == par_env.observation_space(agent)
            for agent in par_env.possible_agents
        ), "observation spaces not consistent. Perhaps you should wrap with `supersuit.multiagent_wrappers.pad_observations_v0`?"
        assert all(
            self.action_space == par_env.action_space(agent) for agent in par_env.possible_agents
        ), "action spaces not consistent. Perhaps you should wrap with `supersuit.multiagent_wrappers.pad_action_space_v0`?"
        self.num_envs = len(par_env.possible_agents)
        self.black_death = black_death

    def concat_obs(self, obs_dict):
        obs_list = []
        for i, agent in enumerate(self.par_env.possible_agents):
            if agent not in obs_dict:
                raise AssertionError(
                    "environment has agent death. Not allowed for pettingzoo_env_to_vec_env_v1 unless black_death is True"
                )
            obs_list.append(obs_dict[agent])

        return concatenate(
            self.observation_space,
            obs_list,
            create_empty_array(self.observation_space, self.num_envs),
        )

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def reset(self, seed=None, options=None):
        # TODO: should this be changed to infos?
        _observations, infos = self.par_env.reset(seed=seed, options=options)
        observations = self.concat_obs(_observations)
        infs = [infos.get(agent, {}) for agent in self.par_env.possible_agents]
        return observations, infs

    def step(self, actions):
        actions = list(iterate(self.action_space, actions))
        agent_set = set(self.par_env.agents)
        act_dict = {
            agent: actions[i]
            for i, agent in enumerate(self.par_env.possible_agents)
            if agent in agent_set
        }
        observations, rewards, terms, truncs, infos = self.par_env.step(act_dict)

        # adds last observation to info where user can get it
        terminations = np.fromiter(terms.values(), dtype=bool)
        truncations = np.fromiter(truncs.values(), dtype=bool)
        env_done = (terminations | truncations).all()
        if env_done:
            for agent, obs in observations.items():
                infos[agent][0]["terminal_observation"] = obs

        rews = np.array(
            [rewards.get(agent, 0) for agent in self.par_env.possible_agents],
            dtype=np.float32,
        )
        tms = np.array(
            [terms.get(agent, False) for agent in self.par_env.possible_agents],
            dtype=np.uint8,
        )
        tcs = np.array(
            [truncs.get(agent, False) for agent in self.par_env.possible_agents],
            dtype=np.uint8,
        )
        infs = [infos.get(agent, {}) for agent in self.par_env.possible_agents]

        if env_done:
            observations, reset_infs = self.reset()
        else:
            observations = self.concat_obs(observations)
            # empty infos for reset infs
            reset_infs = [{} for _ in range(len(self.par_env.possible_agents))]

        # combine standard infos and reset infos
        infs = infs

        assert (
            self.black_death or self.par_env.agents == self.par_env.possible_agents
        ), "MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True"
        return observations, rews, tms, tcs, infs

    def render(self):
        return self.par_env.render()

    def close(self):
        return self.par_env.close()

    def env_is_wrapped(self, wrapper_class):
        """
        env_is_wrapped only suppors vector and gymnasium environments
        currently, not pettingzoo environments
        """
        return [False] * self.num_envs


import gymnasium.vector
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.vector.utils import concatenate, create_empty_array, iterate


def transpose(ll):
    return [[ll[i][j] for i in range(len(ll))] for j in range(len(ll[0]))]


@iterate.register(Discrete)
def iterate_discrete(space, items):
    try:
        return iter(items)
    except TypeError:
        raise TypeError(f"Unable to iterate over the following elements: {items}")


class ConcatVecEnv(gymnasium.vector.VectorEnv):
    def __init__(self, vec_env_fns, obs_space=None, act_space=None):
        self.vec_envs = vec_envs = [vec_env_fn() for vec_env_fn in vec_env_fns]
        for i in range(len(vec_envs)):
            if not hasattr(vec_envs[i], "num_envs"):
                vec_envs[i] = SingleVecEnv([lambda: vec_envs[i]])
        self.metadata = self.vec_envs[0].metadata
        self.render_mode = self.vec_envs[0].render_mode
        self.observation_space = vec_envs[0].observation_space
        self.action_space = vec_envs[0].action_space
        tot_num_envs = sum(env.num_envs for env in vec_envs)
        self.num_envs = tot_num_envs

    def reset(self, seed=None, options=None):
        _res_obs = []
        _res_infos = []

        if seed is not None:
            for i in range(len(self.vec_envs)):
                _obs, _info = self.vec_envs[i].reset(seed=seed + i, options=options)
                _res_obs.append(_obs)
                _res_infos.append(_info)
        else:
            for i in range(len(self.vec_envs)):
                _obs, _info = self.vec_envs[i].reset(options=options)
                _res_obs.append(_obs)
                _res_infos.append(_info)

        # flatten infos (also done in step function)
        flattened_infos = [info for sublist in _res_infos for info in sublist]

        return self.concat_obs(_res_obs), flattened_infos

    def concat_obs(self, observations):
        return concatenate(
            self.observation_space,
            [item for obs in observations for item in iterate(self.observation_space, obs)],
            create_empty_array(self.observation_space, n=self.num_envs),
        )

    def concatenate_actions(self, actions, n_actions):
        return concatenate(
            self.action_space,
            actions,
            create_empty_array(self.action_space, n=n_actions),
        )

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def step(self, actions):
        data = []
        idx = 0
        actions = list(iterate(self.action_space, actions))
        for venv in self.vec_envs:
            data.append(
                venv.step(
                    self.concatenate_actions(actions[idx : idx + venv.num_envs], venv.num_envs)
                )
            )
            idx += venv.num_envs
        observations, rewards, terminations, truncations, infos = transpose(data)
        observations = self.concat_obs(observations)
        rewards = np.concatenate(rewards, axis=0)
        terminations = np.concatenate(terminations, axis=0)
        truncations = np.concatenate(truncations, axis=0)
        infos = [info for sublist in infos for info in sublist]  # flatten infos from nested lists
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.vec_envs[0].render()

    def close(self):
        for vec_env in self.vec_envs:
            vec_env.close()

    def env_is_wrapped(self, wrapper_class):
        return sum([sub_venv.env_is_wrapped(wrapper_class) for sub_venv in self.vec_envs], [])


import gymnasium
import numpy as np


class SingleVecEnv:
    def __init__(self, gym_env_fns, *args):
        assert len(gym_env_fns) == 1
        self.gym_env = gym_env_fns[0]()
        self.render_mode = self.gym_env.render_mode
        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space
        self.num_envs = 1
        self.metadata = self.gym_env.metadata

    def reset(self, seed=None, options=None):
        # TODO: should this include info
        return np.expand_dims(self.gym_env.reset(seed=seed, options=options), 0)

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def render(self):
        return self.gym_env.render()

    def close(self):
        self.gym_env.close()

    def step(self, actions):
        observations, reward, term, trunc, info = self.gym_env.step(actions[0])
        if term or trunc:
            observations = self.gym_env.reset()
        observations = np.expand_dims(observations, 0)
        rewards = np.array([reward], dtype=np.float32)
        terms = np.array([term], dtype=np.uint8)
        truncs = np.array([trunc], dtype=np.uint8)
        infos = [info]
        return observations, rewards, terms, truncs, infos

    def env_is_wrapped(self, wrapper_class):
        env_tmp = self.gym_env
        while isinstance(env_tmp, gymnasium.Wrapper):
            if isinstance(env_tmp, wrapper_class):
                return [True]
            env_tmp = env_tmp.env
        return [False]


from typing import Any, List, Optional

import warnings

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices


class SB3VecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.render_mode = venv.render_mode
        self.reset_infos = []

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed=seed)
        # Note: SB3's vector envs return only observations on reset, and store infos in `self.reset_infos`
        observations, self.reset_infos = self.venv.reset()
        return observations

    def step_wait(self):
        observations, rewards, terminations, truncations, infos = self.venv.step_wait()
        # Note: SB3 expects dones to be an np.array
        dones = np.array([terminations[i] or truncations[i] for i in range(len(terminations))])
        return observations, rewards, dones, infos

    def env_is_wrapped(self, wrapper_class, indices=None):
        # ignores indices
        return self.venv.env_is_wrapped(wrapper_class)

    def getattr_recursive(self, name):
        raise AttributeError(name)

    def getattr_depth_check(self, name, already_found):
        return None

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        attr = self.venv.get_attr(attr_name)
        # Note: SB3 expects render_mode to be returned as an array, with values for each env
        if attr_name == "render_mode":
            return [attr for _ in range(self.num_envs)]
        else:
            return attr

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        warnings.warn(
            "PettingZoo environments do not take the `render(mode)` argument, to change rendering mode, re-initialize the environment using the `render_mode` argument."
        )
        return self.venv.render()
