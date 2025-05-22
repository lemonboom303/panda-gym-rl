# eval_sac.py

import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import time

def evaluate_model(model_path="sac_panda_reach", episodes=5):
    env = gym.make("PandaReachDense-v3", render_mode="human")  # 使用 Dense 环境和 human 渲染
    model = SAC.load(model_path)

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            time.sleep(1 / 60)  

        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
        time.sleep(2)  

    print("Evaluation finished. Keeping window open...")
    time.sleep(10)  
    env.close()

if __name__ == "__main__":
    evaluate_model()
