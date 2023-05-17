import sys
import numpy as np

import rlgym
from rlgym.envs import Match
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3.common.vec_env import VecMonitor

from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3 import PPO

from game_state import CustomStateSetterExo1, CustomStateSetterExo2, CustomStateSetterExo3
from terminal_conditions import BallTouchGround
from Rewards import PlayerUnderBall

# =============================== option for model ============================================
agents_per_match = 1
num_instances = 1
target_steps = 100_000
steps = target_steps // (num_instances * agents_per_match)
batch_size = target_steps // 10
mmr_save_frequency = 200_000

# ==================================== EXO ==============================================
exo = 1
if exo == 1:
    print("Exo 1 launch")
    instance_state = CustomStateSetterExo1()
    tensorboard_log: str = "logs_exo1"
    path_model_save: str = "models/PPO_model_exo1.zip"

elif exo == 2:
    print("Exo 2 launch")
    instance_state = CustomStateSetterExo2()
    tensorboard_log: str = "logs_exo2"
    path_model_save: str = "models/PPO_model_exo2.zip"

elif exo == 3:
    print("Exo 3 launch")
    instance_state = CustomStateSetterExo3()
    tensorboard_log: str = "logs_exo3"
    path_model_save: str = "models/PPO_model_exo3.zip"
else:
    sys.exit("Veuillez sélectionner un exo.")


# ===================================== Define Instance =============================================

def get_match():
    return Match(
        team_size=agents_per_match,
        reward_function=PlayerUnderBall(),
        spawn_opponents=False,
        terminal_conditions=[TimeoutCondition(500), BallTouchGround()],
        obs_builder=AdvancedObs(),
        state_setter=instance_state,  # ball spawn on top of the car
        action_parser=DiscreteAction()  # Transform continuous action to [-1,0,1]
    )


# ==================================================================================

if __name__ == "__main__":
    env = SB3MultipleInstanceEnv(get_match, num_instances, wait_time=20)
    env = VecMonitor(env)

    try:
        model = PPO.load(
            path_model_save,
            env,
            device="auto",
            custom_objects={"n_envs": env.num_envs},
        )
        print("Successfully loaded model.")
    except:
        print("No saved model found, creating new model.")

        model = PPO(
            "MlpPolicy",
            env,
            n_epochs=10,  # Default value
            learning_rate=1e-2,  # Default value 3e-4
            gamma=0,  # Default value 0.99
            verbose=1,  # Print out all the info as we're going
            batch_size=batch_size,  # Batch size as high as possible within reason
            n_steps=steps,  # Number of steps to perform before optimizing network
            tensorboard_log=tensorboard_log,  # `tensorboard --logdir logs_exoXX/PPO_0` in terminal to see graphs
            device="auto"  # Uses GPU if available
        )

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            model.learn(target_steps, reset_num_timesteps=False)  # TODO fix problem de save logs
            model.save(path_model_save)
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    model.save(path_model_save)
    print("Save complete")
