import os

import gym
import gym.spaces
import numpy as np


class UnityEnv(gym.Env):

    def __init__(self, name, action_repeat=1, size=(64, 64)):
        from gym_unity.envs import UnityToGymWrapper
        from mlagents_envs.environment import UnityEnvironment
        
        unity_env = UnityEnvironment(file_name=f"UnityBuild/{name}/{name}", seed=0, no_graphics=False)
        env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False, allow_multiple_obs=False, action_space_seed=0)
        self.env = env

    @property
    def observation_space(self):
        return gym.spaces.Dict({'image': self.env.observation_space})

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, action):
        
        image, reward, done, info = self.env.step(action)
        obs = {'image': image}
        return obs, reward, done, info

    def reset(self):
        image = self.env.reset()  # type: ignore
        obs = {'image': image}
        return obs
