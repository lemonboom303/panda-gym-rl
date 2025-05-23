# callback.py

import os
from stable_baselines3.common.callbacks import EvalCallback

def get_eval_callback(log_dir="./logs_sac/", best_model_dir="./best_model_sac/", eval_freq=5000):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
    import panda_gym

    eval_env = gym.make("PandaReachDense-v3")
    eval_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
    )
    return eval_callback
