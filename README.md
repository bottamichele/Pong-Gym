# Pong-Gymnasium
## About
This repository contains my personal library which states the support of 
my project [Pong clone](https://github.com/bottamichele/Pong-Python) for 
[Gymnasium](https://gymnasium.farama.org/).

## Installation
This library needs to be installed locally to be used. You need to do the following steps to install it:
1. download this repository by running the following command on console.
   ```
   git clone --recursive https://github.com/bottamichele/Pong-Gym
   ```
2. after you have downloaded the repository, you need to run `pip -m install <path_repository>`
   (where ***<path_repository>*** is the absolute path of the repository which is downloaded)
   on terminal to install the library and its all dependecies.

## Usage
Before you run any Pong enviroments, if you make an enviroment with `gymnasium.make()` then 
you must import `pong_gym` and allow Gymnasium to register Pong enviroment at runtime.
Otherwise, you can manually make a Pong enviroment importing the enviroment: 
```python
from pong_py import PongEnv
```
Then, you create an envinroment instancing a `PongEnv` object.

This library contains its wrappers as well and can be imported from `pong_gym.wrappers`.

The code below is a snippet to get start with this library.
```python
import gymnasium as gym
import pong_gym

env = gym.make("pong_gym/Pong-v0", render_mode="human")

for episode in range(10):
    episode_ended = False
    observation, info = env.reset()
    
    while not episode_ended:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        episode_ended = terminated or truncated

env.close()
```

## About enviroment and wrappers
### Observation
Pong's observations contains information about positions and velocities of the paddles and the ball. 
An observation is stored into an array of shape 12
```
[paddle_1.position.x, paddle_1.position.y, paddle_1.velocity.x, paddle_1.velocity.y, paddle_2.position.x, paddle_2.position.y, paddle_2.velocity.x, paddle_2.velocity.y, ball.position.x, ball.position.y, ball.velocity.x, ball.velocity.y]
```
and states:
- first and second item represent current position of the left paddle.
- third and fourth item represent current velocity of the left paddle.
- fifth and sixth item represent current position of the right paddle.
- seventh and eighth item represent current velocity of the right paddle.
- nineth and tenth item represent current position of the ball.
- eleventh and twelfth item represent current velocity of the ball.

### Action
The Pong's action space supports 3 actions and is:
- **0**: the agent does not move its paddle.
- **1**: the agent move up its paddle.
- **2**: the paddle moves down its paddle.

### Reward
The reward function states:
- **1.0**: when the agent got a point.
- **-1.0**: when the opponent got a point.
- **0.1**: when the agent touches the ball.
- **0.0**: otherwise.

### `info` dictionary
The `info` dictionary contains only debug information and are:
- **'agent_score'**: current score of the agent.
- **'bot_score'**: current score of the opponent.
- **'ball_touched'**:' current number of the agent's paddle touched the ball.

### Wrappers
The wrappers implemented in this library are located in `pong_gym.wrappers` and are:
- **PointReward**: it changes the original reward function and considers scores as signal rewards.
  The new reward function states:
  - **1.0**: when the agent got a point.
  - **-1.0**: when the opponent got a point.
  - **0.0**: otherwise.
- **NormalizeObservationPong**: it scales every observation value between -1.0 and 1.0

## License
This library is licensed under the MIT License. For more information, 
see [LICENSE](https://github.com/bottamichele/Pong-Gym/blob/main/LICENSE).
