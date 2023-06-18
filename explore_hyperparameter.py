import os
import time
from rlgym.envs import Match
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, BallTouchedCondition
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3.common.vec_env import VecMonitor

from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

from stable_baselines3 import PPO
def get_match():
    return Match(
        team_size=agents_per_match,
        reward_function=VelocityPlayerToBallReward(),
        spawn_opponents=False,
        terminal_conditions=[TimeoutCondition(timeout), BallTouchedCondition()],
        obs_builder=AdvancedObs(),
        state_setter=DefaultState(),  # Resets to kickoff position
        action_parser=DiscreteAction()  # Transform continuous action to [-1,0,1]
    )

if __name__ == "__main__":
    agents_per_match = 1
    num_instances = 5
    mmr_save_frequency = 1_000_000

    # target_steps = [10_000, 100_000]
    # learning_rate = [1e-3, 1e-2, 1e-1]
    # gamma = [0.9,0.99,1.0]
    # timeout = [100, 225]
    target_steps = 10_000
    learning_rate = 1e-3
    gamma = 0.99
    timeout = 225

    steps = target_steps // (num_instances * agents_per_match)
    batch_size = target_steps // 10

    model_path = f"models_exploration/PPO_model_steps{target_steps}_lr{learning_rate}_gamma{gamma}_timeout{timeout}.zip"
    mmr_path = f"mmr_models/steps{target_steps}_lr{learning_rate}_gamma{gamma}_timeout{timeout}/"
    if not os.path.exists("models_exploration/"):
        os.makedirs("models_exploration/")
    if not os.path.exists(mmr_path):
        os.makedirs(mmr_path)
    env = SB3MultipleInstanceEnv(get_match, num_instances, wait_time=15)
    env = VecMonitor(env)

    try:
        model = PPO.load(
            model_path,
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
            n_epochs=10,                 # Default value 10
            learning_rate=learning_rate,          # Default value 3e-4
            gamma=gamma,                  # Default value 0.99
            verbose=1,                   # Print out all the info as we're going
            batch_size=batch_size,             # Batch size as high as possible within reason
            n_steps=steps,               # Number of steps to perform before optimizing network
            tensorboard_log=f"logs_steps{target_steps}_lr{learning_rate}_gamma{gamma}_timeout{timeout}",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"                 # Uses GPU if available
        )
    start_time = time.time()
    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            model.learn(target_steps, reset_num_timesteps=False)
            model.save(model_path)
            end_time = time.time()
            duration = end_time - start_time
            print("Current time : ", duration, " seconds")
            print("Total steps per seconds :", model.num_timesteps / duration)
            if model.num_timesteps >= mmr_model_target_count:
                model.save(mmr_path + str(model.num_timesteps))
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    model.save(model_path)
    print("Save complete")

