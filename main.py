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

from game_state import CustomStateSetterExo1, CustomStateSetterExo2, CustomStateSetterExo3
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
exo = 3
if exo == 1:
    instance_state = CustomStateSetterExo1()
    tensorboard_log: str = f"logs_exo{exo}"
    path_model_save: str = "models/PPO_model_exo1.zip"
    print("Exo 1 launch")

elif exo == 2:
    instance_state = CustomStateSetterExo2()
    tensorboard_log: str = f"logs_exo1-2"
    path_model_save: str = "models/PPO_model_exo1-2.zip"
    print("Exo 2 launch")

elif exo == 3:
    instance_state = CustomStateSetterExo3()
    tensorboard_log: str = f"logs_exo1-2-3"
    path_model_save: str = "models/PPO_model_exo1-2-3.zip"
    print("Exo 3 launch")
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
            learning_rate=1e-3,  # Default value 3e-4
            gamma=0.99,  # Default value 0.99
            verbose=1,  # Print out all the info as we're going
            batch_size=batch_size,  # Batch size as high as possible within reason
            n_steps=steps,  # Number of steps to perform before optimizing network
            tensorboard_log=tensorboard_log,  # `tensorboard --logdir logs_exoXX/PPO_0` in terminal to see graphs
            device="auto",  # Uses GPU if available

            #ent_coef = 0.1 # C'est le coefficient pour la pénalité d'entropie. Cela encourage l'exploration en ajoutant une pénalité pour les politiques de faible entropie
                            # (c'est-à-dire les politiques trop certaines). Si vous constatez que votre agent n'explore pas assez, augmenter cette valeur pourrait aider.

            # Utiliser une autre politique: Vous utilisez "MlpPolicy" qui est une politique à base de perceptron multicouche. Vous pourriez envisager d'essayer d'autres politiques,
                            #  comme "CnnPolicy" si votre environnement a des entrées basées sur des images, ou "LstmPolicy" si votre environnement a des dépendances temporelles.



        )

    mmr_model_path = f"mmr_models/exo{exo}/"
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
