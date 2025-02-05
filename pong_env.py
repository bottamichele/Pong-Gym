import numpy as np
import pygame

from enum import Enum

from pygame import Vector2, Rect

from gymnasium import Env
from gymnasium.spaces import Discrete, Box

from pong_py.pong.ball import Ball
from pong_py.pong.paddle import Paddle
from pong_py.pong.game import Game
from pong_py.pong.controller.controller import PaddlePosition, MovingType
from pong_py.pong.controller.basic_bot_controller import BasicBotController
from pong_py.pong.controller.bot_controller import BotController

from agent_controller import AgentController
from train_pong_cl import TrainPongContactListener

# ==================================================
# ============= CLASS BotControllerType ============
# ==================================================

class BotControllerType(Enum):
    """A bot controller where a training agent to play against."""

    BASIC_BOT = 0       #Bot with basic strategy.
    BOT = 1             #Bot with advanced strategy.

# ==================================================
# ================== CLASS PongEnv =================
# ==================================================

class PongEnv(Env):
    """An enviroment that support Pong clone game for Gymnasium."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, bot_controller_type=BotControllerType.BOT, render_mode=None):
        """Create a new Pong enviroment.
        
        Parameter
        --------------------
        bot_controller_type: BotControllerType, optional
            a bot controller where a training agent is trained to play against
            
        render_mode: str, optional
            render mode to use"""
        
        assert isinstance(bot_controller_type, BotControllerType)
        assert render_mode is None or render_mode in self.metadata["render_modes"], "render_mode supports None, \"rgb_array\" and \"human\"."

        temp_game = Game()

        #Public attributes.
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
                                                   temp_game.field.center_position.x + temp_game.field.width/2,
                                                   temp_game.field.center_position.y + temp_game.field.height/2,
                                                   0,
                                                   Paddle.SPEED,
                                                   temp_game.field.center_position.x + temp_game.field.width/2,
                                                   temp_game.field.center_position.y + temp_game.field.height/2,
                                                   Ball.SPEED,
                                                   Ball.SPEED], 
                                                   dtype=np.float32), 
                                    shape=(12,), 
                                    dtype=np.float32)
        self.action_space = Discrete(3)
        self.render_mode = render_mode
        
        #Window parameters.
        self._window_size = (700, 550)
        self._window = None
        self._clock = None
        self._font = None
        self._font_color = (255, 255, 255)

        #Pong's game parameters.
        self._current_game = None
        self._agent_controller = None
        self._bot_controller = None
        self._bot_controller_type = bot_controller_type
        self._last_agent_score = 0
        self._last_bot_score = 0

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
            self._bot_controller = BasicBotController(self._current_game.paddle_2, PaddlePosition.RIGHT, self._current_game.ball)
        elif self._bot_controller_type == BotControllerType.BOT:
            self._bot_controller = BotController(self._current_game.paddle_2, PaddlePosition.RIGHT, self._current_game)

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
        self._agent_controller.update(1 / self.metadata["render_fps"])
        self._bot_controller.update(1 / self.metadata["render_fps"])
        self._current_game.update(1 / self.metadata["render_fps"])

        #
        current_agent_score = self._current_game.score_paddle_1
        current_bot_score = self._current_game.score_paddle_2

        if current_agent_score > self._last_agent_score:        #Has training agent got a point?
            reward = 1.0
        elif current_bot_score > self._last_bot_score:          #Has bot got a point?
            reward = -1.0
        elif self._agent_controller.is_colliding_ball:          #Has training agent collided ball?
            reward = 0.1
        else:                                                   #Nothing happens.
            reward = 0.0

        #Update last player scores.
        self._last_agent_score = current_agent_score
        self._last_bot_score = current_bot_score

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, self._current_game.is_ended(), False, self._get_info()
    
    def render(self):
        if self.render_mode is None:
            return
        
        #Initialize font.
        if self._font is None:
            pygame.init()
            self._font = pygame.font.Font(None, 50)
        
        #Initialize window and clock if render_mode is "human".
        if self.render_mode == "human":
            if self._window is None:
                pygame.display.init()
                pygame.display.set_caption("Pong")

                self._window = pygame.display.set_mode(self._window_size)
            if self._clock is None:
                self._clock = pygame.time.Clock()        

        #Create new canvas and fill with black color.
        canvas = pygame.Surface(self._window_size)
        canvas.fill("black")

        #Draw text.
        self._draw_score(canvas, self._current_game.score_paddle_1, (self._window_size[0] // 4, 25))
        self._draw_score(canvas, self._current_game.score_paddle_2, (3 * self._window_size[0] // 4, 25))

        #Draw the borders of field.
        self._draw_border_field(canvas)

        #Draw paddles and ball on canvas.
        self._draw_rect(canvas, self._current_game.paddle_1.position, self._current_game.paddle_1.width, self._current_game.paddle_1.height)
        self._draw_rect(canvas, self._current_game.paddle_2.position, self._current_game.paddle_2.width, self._current_game.paddle_2.height)
        self._draw_rect(canvas, self._current_game.ball.position, self._current_game.ball.radius, self._current_game.ball.radius)

        if self.render_mode == "human":
            self._window.blit(canvas, canvas.get_rect())

            #pygame.event.pump()
            # self._clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            
            #pygame.display.update()

            self._clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _translate_position(self, position):
        """Translate from position of an object to canvas coordinate."""
        
        new_x = position.x + self._window_size[0]/2
        new_y = -(position.y - self._window_size[1]/2)

        return Vector2(new_x, new_y)

    def _draw_score(self, canvas, score_paddle, position):
        """Draw score of a paddle on canvas."""
    
        #Text
        score_paddle_text = self._font.render("{}".format(score_paddle), True, self._font_color)
        
        #Text position on screen
        score_paddle_rect = score_paddle_text.get_rect()
        score_paddle_rect.center = position

        #Draw text
        canvas.blit(score_paddle_text, score_paddle_rect)

    def _draw_border_field(self, canvas, height=20):
        """Draw the borders of field on canvas."""
        
        #Top border of field.
        top_border_pos = self._translate_position(Vector2(self._current_game.field.center_position.x - self._current_game.field.width/2, self._current_game.field.center_position.y + self._current_game.field.height/2))
        pygame.draw.rect(canvas, "white", Rect(top_border_pos.x, top_border_pos.y - height, self._window_size[0], height))

        #Bottom border of field.
        bottom_border_pos = self._translate_position(Vector2(self._current_game.field.center_position.x - self._current_game.field.width/2, self._current_game.field.center_position.y - self._current_game.field.height/2))
        pygame.draw.rect(canvas, "white", Rect(bottom_border_pos.x, bottom_border_pos.y, self._window_size[0], height))

    def _draw_rect(self, canvas, position, width, height):
        """Draw a rectangle on canvas."""
    
        left_vertix_pos = self._translate_position(position + Vector2(-width/2, height/2))
        pygame.draw.rect(canvas, "white", Rect(left_vertix_pos.x, left_vertix_pos.y, width, height))

    def close(self):
        if self._window is not None:
            pygame.display.quit()

        if self._font is not None:
            pygame.quit()