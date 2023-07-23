from abc import ABC

from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState


class BallTouchGround(TerminalCondition):

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        """
        Return `True` if the ball hit the gound
        """
        return current_state.ball.position[2] <= 103 
