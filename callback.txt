import os
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            results = self.model.ep_info_buffer
            if len(results) > 0:
                mean_reward = sum([ep["r"] for ep in results]) / len(results)
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: mean_reward={mean_reward:.2f}, best={self.best_mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
                    if self.verbose > 0:
                        print(f"Best model saved to: {self.save_path}")
        return True
