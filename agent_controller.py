from .pong_py.pong.controller.controller import Controller, MovingType

class AgentController(Controller):
    """A controller used by a training agent to control a paddle."""

    def __init__(self, a_paddle, position):
        super().__init__(a_paddle, position)
        
        self.is_colliding_ball = False
        self.n_touch = 0
        self._next_move = MovingType.NONE

    def set_next_move(self, next_move):
        """Set next moving type to .
        
        Parameter
        --------------------
        next_move: MovingType
            a move type"""
        
        self._next_move = next_move

    def update(self, delta_time):
        self._move_paddle(self._next_move)