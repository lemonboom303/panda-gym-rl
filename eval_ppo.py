import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
import time

def evaluate_model(model_path="./logs/best_model.zip", render=True, episodes=5):
    env = gym.make("PandaReach-v3", render_mode="human" if render else None)
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.03)  # 加这个防止窗口闪退太快
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")

    env.close()
    input("测试完成，按任意键关闭窗口...")  # 防止窗口一闪而过

if __name__ == "__main__":
    evaluate_model()
