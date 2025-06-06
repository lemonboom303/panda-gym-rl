# train_sac.py

import os
import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from callback_sac import get_eval_callback # 导入我们写的回调函数

def train_model():
    log_dir = "./logs_sac/"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make("PandaReachDense-v3")
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    eval_callback = get_eval_callback(log_dir=log_dir)

    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(total_timesteps=50_000, callback=eval_callback)
    model.save("sac_panda_reach")

if __name__ == "__main__":
    train_model()
