import os

import gym
import gym.spaces
import numpy as np


class RLGridWorldEnv(gym.Env):

    def __init__(self, name, action_repeat=1, size=(64, 64)):
        from rlgridworld.gridenv import GridEnv
        env = GridEnv(load_chars_rep_fromd_dir=f'RLGridWorldSettings/{name}.txt', render_mode='human', obs_mode='single_rgb_array', render_width=size[0], render_height=size[1])
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
