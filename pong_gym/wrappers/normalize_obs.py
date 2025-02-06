import numpy as np

from gymnasium import Wrapper
from gymnasium.spaces import Box

from ..envs.pong_py.pong.paddle import Paddle

class NormalizeObservationPong(Wrapper):
    """A wrapper which normalizes observations of the Pong game."""

    def __init__(self, env):
        assert env.observation_space.shape == (12,) and env.observation_space == env.unwrapped.observation_space
        
        super().__init__(env)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        #Field parameters.
        self._field_center = (env.observation_space.low[[0, 1]] + env.observation_space.high[[0, 1]]) / 2
        self._field_size = np.abs(env.observation_space.high[[0, 1]] - env.observation_space.low[[0, 1]])

    def _normalize_obs(self, obs):
        obs_norm = np.zeros(12, dtype=np.float32)

        #Normalize the ball and paddle positions.
        for idx_obj_pos in [[0, 1], [4, 5], [8, 9]]:
            obs_norm[idx_obj_pos] = (obs[idx_obj_pos] - self._field_center) / (self._field_size/2)

        #Normalize the paddle velocity y.
        for idx_paddle_vel_y in [3, 7]:
            obs_norm[idx_paddle_vel_y] = obs[idx_paddle_vel_y] / Paddle.SPEED

        #Normalize the ball velocity.
        obs_norm[[10, 11]] = obs[[10, 11]] / np.linalg.norm(obs[[10, 11]])

        return obs_norm

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._normalize_obs(obs), info
    
    def step(self, action):
        next_obs, reward, terminated, truncated, info = super().step(action)
        return self._normalize_obs(next_obs), reward, terminated, truncated, info