# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import gym
import torch
from envs.sumo_lanechange_env import SumoLaneChangeEnv

# Define macros for hyperparameters
STEP_LENGTH = 0.1 # step length in seconds for SUMO and environment step

LEARNING_RATE = 1e-4
N_STEPS = 512 #number of env steps PPO coellects before each update, it can be multiple episodes or one
BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TOTAL_TIMESTEPS = 500_000 #total timesteps to train the model
MODEL_NAME = "ppo_sumo_lanechange"
max_episode_steps = 256 #int(60 / STEP_LENGTH) * 2  # experimenting with 2 episode per update


env = SumoLaneChangeEnv("SUMO_sim/base_compl/base.sumocfg", STEP_LENGTH, max_episode_steps, ego_flow_id='f_0') 

# (Optional) small MLP that fits a 21-D state → policy/value
policy_kwargs = dict(
    activation_fn=nn.Tanh,
    net_arch=dict(pi=[64, 64], vf=[64, 64])  # policy/value separate heads
)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=LEARNING_RATE,      # paper
    n_steps=N_STEPS,             # horizen for each policy update ppo batch length
    batch_size=BATCH_SIZE,           # paper
    gamma=GAMMA,              # paper
    gae_lambda=GAE_LAMBDA,         # paper
    clip_range=CLIP_RANGE,          # paper
    ent_coef=ENT_COEF,           # Eq. (2) entropy weight c2
    vf_coef=VF_COEF,             # Eq. (2) value weight c1
    max_grad_norm=MAX_GRAD_NORM,
    policy_kwargs=policy_kwargs,
    verbose=1,
)

model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(MODEL_NAME)

#model = PPO.load(MODEL_NAME)