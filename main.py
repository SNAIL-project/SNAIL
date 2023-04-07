import rlgym
from rlgym.envs import Match
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, BallTouchedCondition
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3.common.vec_env import VecMonitor

from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3 import PPO

agents_per_match = 1
num_instances = 3
target_steps = 100_000
steps = target_steps // (num_instances * agents_per_match)
batch_size = target_steps//10
mmr_save_frequency = 200_000

def get_match():
    return Match(
        team_size=agents_per_match,
        reward_function=CombinedReward(
            (
                VelocityPlayerToBallReward(),
                EventReward(
                    shot=100.0,
                ),
            ),
            (0.1, 1.0)),
        spawn_opponents=False,
        terminal_conditions=[TimeoutCondition(225), BallTouchedCondition()],
        obs_builder=AdvancedObs(),
        state_setter=DefaultState(),  # Resets to kickoff position
        action_parser=DiscreteAction()  # Transform continuous action to [-1,0,1]
    )

if __name__ == "__main__":
    env = SB3MultipleInstanceEnv(get_match, num_instances, wait_time=20)
    env = VecMonitor(env)

    try:
        model = PPO.load(
            "models/PPO_model_exo0.zip",
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
            n_epochs=10,                 # Default value
            learning_rate=1e-2,          # Default value 3e-4
            gamma=0,                  # Default value 0.99
            verbose=1,                   # Print out all the info as we're going
            batch_size=batch_size,             # Batch size as high as possible within reason
            n_steps=steps,               # Number of steps to perform before optimizing network
            tensorboard_log="logs_exo0",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"                 # Uses GPU if available
        )

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            model.learn(target_steps, reset_num_timesteps=False)
            model.save("models/PPO_model_exo0.zip")
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    model.save("models/PPO_model_exo0.zip")
    print("Save complete")

