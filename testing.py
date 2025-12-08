from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import gym
import torch
from envs.sumo_lanechange_env import SumoLaneChangeEnv

model = PPO.load("ppo_sumo_lanechange")
env = SumoLaneChangeEnv("SUMO_sim/base_compl/base.sumocfg")
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

        