from .envs.pong_env import PongEnv, BotControllerType

from gymnasium.envs.registration import register

register(id="pong_gym/Pong-v0", entry_point="pong_gym.envs:PongEnv",)