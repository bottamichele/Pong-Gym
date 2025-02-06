from gymnasium import Wrapper

class PointReward(Wrapper):
    """A reward wrapper which considers only points as signal rewards."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward != 1.0 and reward != -1.0:
            reward = 0.0

        return obs, reward, terminated, truncated, info