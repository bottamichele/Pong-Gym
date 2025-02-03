import numpy as np

from enum import Enum

from gymnasium import Env
from gymnasium.spaces import Discrete, Box

from .pong_py.pong.ball import Ball
from .pong_py.pong.paddle import Paddle
from .pong_py.pong.game import Game
from .pong_py.pong.controller.controller import PaddlePosition, MovingType
from .pong_py.pong.controller.basic_bot_controller import BasicBotController
from .pong_py.pong.controller.bot_controller import BotController

from .agent_controller import AgentController
from .train_pong_cl import TrainPongContactListener

# ==================================================
# ============= CLASS BotControllerType ============
# ==================================================

class BotControllerType(Enum):
    """A bot controller where a training agent to play against."""

    BASIC_BOT = 0,      #Bot with basic strategy.
    BOT = 1             #Bot with advanced strategy.

# ==================================================
# ================== CLASS PongEnv =================
# ==================================================

class PongEnv(Env):
    """An enviroment that support Pong clone game for Gymnasium."""

    def __init__(self, bot_controller_type=BotControllerType.BOT):
        """Create a new Pong enviroment.
        
        Parameter
        --------------------
        bot_controller_type: BotControllerType
            a bot controller where a training agent is trained to play against."""
        
        assert isinstance(bot_controller_type, BotControllerType)

        temp_game = Game()

        self.observation_space = Box(low=np.array([temp_game.field.center_position.x - temp_game.field.width/2,
                                                   temp_game.field.center_position.y - temp_game.field.height/2,
                                                   0,
                                                   -Paddle.SPEED,
                                                   temp_game.field.center_position.x - temp_game.field.width/2,
                                                   temp_game.field.center_position.y - temp_game.field.height/2,
                                                   0,
                                                   -Paddle.SPEED,
                                                   temp_game.field.center_position.x - temp_game.field.width/2,
                                                   temp_game.field.center_position.y - temp_game.field.height/2,
                                                   -Ball.SPEED,
                                                   -Ball.SPEED], 
                                                   dtype=np.float32), 
                                    high=np.array([temp_game.field.center_position.x + temp_game.field.width/2,
                                                   temp_game.field.center_position.y + temp_game.field.height/2,
                                                   0,
                                                   Paddle.SPEED,
                                                   temp_game.field.center_position.x - temp_game.field.width/2,
                                                   temp_game.field.center_position.y - temp_game.field.height/2,
                                                   0,
                                                   Paddle.SPEED,
                                                   temp_game.field.center_position.x - temp_game.field.width/2,
                                                   temp_game.field.center_position.y - temp_game.field.height/2,
                                                   Ball.SPEED,
                                                   Ball.SPEED], 
                                                   dtype=np.float32), 
                                    shape=(12,), 
                                    dtype=np.float32)
        self.action_space = Discrete(3)
        self._current_game = None
        self._agent_controller = None
        self._bot_controller = None
        self._bot_controller_type = bot_controller_type
        self._last_agent_score = 0
        self._last_bot_score = 0
        self._fps_limit = 60

    def _get_obs(self):
        """Return the current observation of current game.
        
        Return
        --------------------
        obs: np.ndarray
            current observation of current game"""
        
        assert self._current_game is not None, "No game is started."
        
        paddle_1_pos = self._current_game.paddle_1.position
        paddle_1_vel = self._current_game.paddle_1.velocity
        paddle_2_pos = self._current_game.paddle_2.position
        paddle_2_vel = self._current_game.paddle_2.velocity
        ball_pos = self._current_game.ball.position
        ball_vel = self._current_game.ball.velocity
        
        return np.array([paddle_1_pos.x,
                        paddle_1_pos.y,
                        paddle_1_vel.x,
                        paddle_1_vel.y,
                        paddle_2_pos.x, 
                        paddle_2_pos.y,
                        paddle_2_vel.x,
                        paddle_2_vel.y,
                        ball_pos.x,
                        ball_pos.y,
                        ball_vel.x,
                        ball_vel.y])
    
    def _get_info(self):
        """Return a info dict."""

        assert self._current_game is not None, "No game is started."
        assert self._agent_controller is not None, "Training agent does not have any controllers."

        return { "agent_score": self._current_game.score_paddle_1, 
                 "bot_score": self._current_game.score_paddle_2, 
                 "ball_touched": self._agent_controller.n_touch }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #Create and start a new game.
        cl = TrainPongContactListener()
        self._current_game = Game(contact_listener=cl)
        self._current_game.start()

        #Controllers are set.
        self._agent_controller = AgentController(self._current_game.paddle_1, PaddlePosition.LEFT)
        cl.agent_controller = self._agent_controller
        
        if self._bot_controller_type == BotControllerType.BASIC_BOT:
            self._bot_controller = BasicBotController(self._current_game.paddle_2, PaddlePosition.RIGHT)
        elif self._bot_controller_type == BotControllerType.BOT:
            self._bot_controller = BotController(self._current_game.paddle_2, PaddlePosition.RIGHT)

        #Initialize last player scores.
        self._last_agent_score = self._current_game.score_paddle_1
        self._last_bot_score = self._current_game.score_paddle_2

        #Return the initial observation.
        return self._get_obs(), self._get_info()

    def step(self, action):
        assert action in self.action_space, "Invalid action given."

        #Store action to perform into agent controller.
        self._agent_controller.set_next_move(MovingType(action))

        #Perform an update step.
        self._agent_controller.update(1 / self._fps_limit)
        self._bot_controller.update(1 / self._fps_limit)
        self._current_game.update(1 / self._fps_limit)

        #
        current_agent_score = self._current_game.score_paddle_1
        current_bot_score = self._current_game.score_paddle_2

        if current_agent_score > self._last_agent_score:        #Has training agent got a point?
            reward = 1.0
        elif current_bot_score > self._last_bot_score:          #Has bot got a point?
            reward = -1.0
        elif self._agent_controller.is_colliding_ball:          #Has training agent collided ball?
            self._current_reward = 0.1
        else:                                                   #Nothing happens.
            reward = 0.0

        #Update last player scores.
        self._last_agent_score = current_agent_score
        self._last_bot_score = current_bot_score

        return self._get_obs(), reward, self._current_game.is_ended(), False, self._get_info()