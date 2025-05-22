# train_ppo.py
import os
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from callback import SaveOnBestTrainingRewardCallback

def train_model():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make("PandaReachDense-v3")
    env = Monitor(env, log_dir)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=3e-4,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[dict(pi=[512, 512], vf=[512, 512])]),
    )

    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save("ppo_panda_reach")

if __name__ == "__main__":
    train_model()
