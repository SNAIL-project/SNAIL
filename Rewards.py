from rlgym.utils import RewardFunction
# from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils import math as maths_rl
import math
import numpy as np


class PlayerUnderBall(RewardFunction):
    def __init__(self):
        self.max_reward = 1000.0
        self.distance_scale = 1.0 / 93.0  # echelle à VERIF

    def reset(self, initial_state: GameState):  # Called every reset.
        pass

    def get_reward(self, player: PlayerData, state: GameState,
                   previous_action: np.ndarray) -> float:  # Called every step.
        car_pos = player.car_data.position
        ball_pos = state.ball.position
        ball_radius = 93.0  # le rayon de la balle est 93.0u

        # calculer la distance horizontale entre la voiture et la balle
        distance_to_ball = math.sqrt((ball_pos[0] - car_pos[0]) ** 2 + (ball_pos[1] - car_pos[1]) ** 2)

        # calculer la distance verticale entre la voiture et la balle
        roof_offset = max(0, car_pos[2] - ball_pos[2] - ball_radius)

        # calculer la récompense
        reward = self.max_reward - (distance_to_ball + roof_offset * self.distance_scale)

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState,
                         previous_action: np.ndarray) -> float:  # Called if the current state is terminal.
        return 0


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

    def get_reward(self, state, prev_state):
        #  calculates the distance between the agent and the ball using the Euclidean distance formula. If the agent is under the ball on the z-index and closer to the ball (distance less than 1000),
        #  it is rewarded with a value of 1/distance. Otherwise, the agent is not rewarded. You can adjust the distance threshold and the reward value to suit your needs.
        # La formule de la distance euclidienne entre deux points en trois dimensions est la suivante : =      distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5

        # Get the position of the agent and the ball
        agent_pos = state.observation.players[0].physics.location
        ball_pos = state.observation.ball.physics.location

        # Calculate the distance between the agent and the ball
        distance = ((agent_pos.x - ball_pos.x) ** 2 + (agent_pos.y - ball_pos.y) ** 2 + (
                agent_pos.z - ball_pos.z) ** 2) ** 0.5


        # Calculate the reward based on the distance
        if agent_pos.z < ball_pos.z and distance < 1000:
            reward = 1 / distance
            print("distance = " + distance, "reward = " + reward)
        else:
            reward = 0

        return reward
