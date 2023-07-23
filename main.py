import sys
import os
import numpy as np

import rlgym
from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3.common.vec_env import VecMonitor

from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO

from game_state import CustomStateSetterExo1, CustomStateSetterExo2, CustomStateSetterExo3, CustomStateSetterExo4, CustomStateSetterExo5, CustomStateSetterExo6
from terminal_conditions import BallTouchGround
from Rewards import PlayerUnderBall2

# =============================== option for model ============================================
agents_per_match = 1
num_instances = 5
target_steps = 10_000
steps = target_steps // (num_instances * agents_per_match)
batch_size = target_steps // 10
mmr_save_frequency = 1_000_000

# ==================================== EXO ==============================================
exo = 5
if exo == 1:
    instance_state = CustomStateSetterExo1()
    tensorboard_log: str = f"logs_exo{exo}"
    path_model_save: str = "models/PPO_model_exo1.zip"
    print("Exo 1 launch")

elif exo == 2:
    instance_state = CustomStateSetterExo2()
    tensorboard_log: str = f"logs_exo{exo}"
    path_model_save: str = "models/PPO_model_exo1-2.zip"
    print("Exo 2 launch")

elif exo == 3:
    instance_state = CustomStateSetterExo3()
    tensorboard_log: str = f"logs_exo{exo}"
    path_model_save: str = "models/PPO_model_exo1-2-3.zip"
    print("Exo 3 launch")
elif exo == 4:
    instance_state = CustomStateSetterExo4()
    tensorboard_log: str = f"logs_exo{exo}"
    path_model_save: str = "models/PPO_model_exo1-2-3-4.zip"
    print("Exo 4 launch")
elif exo == 5:
    instance_state = CustomStateSetterExo5()
    tensorboard_log: str = f"logs_exo{exo}"
    path_model_save: str = "models/PPO_model_exo1-2-3-4-5.zip"
    print("Exo 5 launch")
elif exo == 6:
    instance_state = CustomStateSetterExo6()
    tensorboard_log: str = f"logs_exo{exo}"
    path_model_save: str = "models/PPO_model_exo1-2-3-4-5-6.zip"
    print("Exo 6 launch")
else:
    sys.exit("Veuillez sélectionner un exo.")


# ===================================== Define Instance =============================================

def get_match():
    return Match(
        team_size=agents_per_match,
        reward_function=PlayerUnderBall2(),
        spawn_opponents=False,
        terminal_conditions=[TimeoutCondition(500), BallTouchGround()],
        obs_builder=AdvancedObs(),
        state_setter=instance_state,
        action_parser=DiscreteAction()
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
            n_epochs=10,
            learning_rate=1e-3,
            gamma=0.99,
            verbose=1,
            batch_size=batch_size,
            n_steps=steps,
            tensorboard_log=tensorboard_log,
            device="auto",
        )

    mmr_model_path = f"mmr_models/exo1-2-3-4-5/"
    if not os.path.exists(mmr_model_path):
        os.makedirs(mmr_model_path)

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            model.learn(target_steps, reset_num_timesteps=False)
            model.save(path_model_save)
            if model.num_timesteps >= mmr_model_target_count:
                model.save(mmr_model_path+str(model.num_timesteps))
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    model.save(path_model_save)
    print("Save complete")
