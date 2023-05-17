from abc import ABC

from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState


class BallTouchGround(TerminalCondition):

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):  # Called once per reset.
        pass

    def is_terminal(self, current_state: GameState) -> bool:  # Called once per step.
        """
        Return `True` if the ball hit the gound
        """
        # print(current_state.ball.position)
        return current_state.ball.position[2] <= 103  # bcs it s the height of the top of the ball (94u) but the game is BUGED? so we put 103u
