# Panda-Gym 强化学习机械臂抓取项目

本项目基于 `panda-gym` 仿真环境，使用 `Stable-Baselines3` 框架实现两种主流强化学习算法（PPO 和 SAC），训练机械臂完成抓取目标的控制任务。

## 项目结构

```
├── train_ppo.py         # PPO算法训练脚本
├── train_sac.py         # SAC算法训练脚本
├── eval_ppo.py          # PPO策略评估
├── eval_sac.py          # SAC策略评估
├── callback.py          # PPO自定义回调（保存最优模型）
├── callback_sac.py      # SAC评估回调
├── logs/                # PPO日志目录
├── logs_sac/            # SAC日志目录
├── best_model/          # 存储最优模型
├── README.md
```

## 环境配置

建议使用 Conda 管理虚拟环境：

```bash
conda create -n panda_env python=3.9
conda activate panda_env

pip install stable-baselines3[extra]
pip install gymnasium==0.29.1
pip install panda-gym==3.0.7
pip install pybullet
pip install tensorboard
```

如果环境注册失败，可运行以下命令触发 panda-gym 注册：

```bash
python -c "import panda_gym; import gymnasium as gym; env = gym.make('PandaReach-v3'); env.reset(); env.close()"
```

## 算法训练

### 训练 PPO 模型

```bash
python train_ppo.py
```

模型与日志将保存在 `./logs/` 目录下。

### 训练 SAC 模型

```bash
python train_sac.py
```

模型与日志将保存在 `./logs_sac/` 目录下。

## 策略评估

可视化测试训练好的策略：

```bash
# PPO
python eval_ppo.py

# SAC
python eval_sac.py
```

## 可视化日志（TensorBoard）

```bash
tensorboard --logdir ./logs/      # PPO
tensorboard --logdir ./logs_sac/  # SAC
```

浏览器访问：`http://localhost:6006` 查看训练曲线（如 success_rate、reward、actor_loss 等）。
