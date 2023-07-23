from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils import math as maths_rl
import numpy as np

class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        linear_velocity = player.car_data.linear_velocity
        reward = maths_rl.vecmag(linear_velocity)

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class PlayerUnderBall2(RewardFunction):

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player, state, prev_state ):

        # Get the position of the agent and the ball
        agent_pos = state.players[0].car_data.position
        ball_pos = state.ball.position

        agent_pos[2] += 18.08
        ball_pos[2] -= 92.75

        # Calculate the distance between the agent and the ball: (x, y et z = 0, 1 et 2)
        distance = ((agent_pos[0] - ball_pos[0]) ** 2 + (agent_pos[1] - ball_pos[1]) ** 2 + (
                agent_pos[2] - ball_pos[2]) ** 2) ** 0.5

        # Calculate the reward based on the distances
        if agent_pos[2] < ball_pos[2] and distance < 1000:
            reward = 1 / distance
            # print("distance = " + str(distance),  "reward = " + str(reward))
        else:
            reward = 0

        return reward
