from .pong_py.pong.game import PongGameContactListener
from .pong_py.pong.paddle import Paddle
from .pong_py.pong.ball import Ball

class TrainPongContactListener(PongGameContactListener):
    """A base collision system listener used for training of agents on Pong."""

    agent_controller = None          #Controller (AgentController) 's left paddle.

    def BeginContact(self, contact):
        super().BeginContact(contact)

        #Does paddle start contact with ball?
        if (isinstance(contact.fixtureA.userData, Paddle) and isinstance(contact.fixtureB.userData, Ball)) or (isinstance(contact.fixtureA.userData, Ball) and isinstance(contact.fixtureB.userData, Paddle)):
            paddle = contact.fixtureA.userData if isinstance(contact.fixtureA.userData, Paddle) else contact.fixtureB.userData

            #Is agent_controller's paddle?
            if self.agent_controller is not None and self.agent_controller.paddle == paddle:
                self.agent_controller.is_colliding_ball = True
                self.agent_controller.n_touch += 1

    def EndContact(self, contact):
        super().EndContact(contact)

        #Does paddle end contact with ball?
        if (isinstance(contact.fixtureA.userData, Paddle) and isinstance(contact.fixtureB.userData, Ball)) or (isinstance(contact.fixtureA.userData, Ball) and isinstance(contact.fixtureB.userData, Paddle)):
            paddle = contact.fixtureA.userData if isinstance(contact.fixtureA.userData, Paddle) else contact.fixtureB.userData

            #Is agent_controller's paddle?
            if self.agent_controller is not None and self.agent_controller.paddle == paddle:
                self.agent_controller.is_colliding_ball = False