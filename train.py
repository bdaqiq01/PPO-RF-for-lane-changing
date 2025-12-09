# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback  # Add this import
from torch import nn
import gym
import torch
from envs.sumo_lanechange_env import SumoLaneChangeEnv
import os


# Define macros for hyperparameters
STEP_LENGTH = 0.2 # 0.1 # step length in seconds for SUMO and environment step 0.1 is very slow so changed

LEARNING_RATE = 1e-4
N_STEPS = 512 #number of env steps PPO coellects before each update, it can be multiple episodes or one
BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TOTAL_TIMESTEPS = 200000 #500_000 #total timesteps to train the model
MODEL_NAME = "ppo_sumo_lanechange"
max_episode_steps = 256 #int(60 / STEP_LENGTH) * 2  # experimenting with 2 episode per update

# --- IDM longitudinal controller (from paper) ---
IDM_V0 = 30.0        # m/s
IDM_T = 1.5          # s
IDM_A_MAX = 2.5      # m/s^2
IDM_B_COMF = 4.5     # m/s^2 (positive; used as braking limit)
IDM_S0 = 2.0         # m

# --- Lateral controller ---
LANE_CHANGE_DURATION = 3  # seconds lane change duration
LANE_CHANGE_DETECTION_DISTANCE = 10  # meters lane change detection distance 
FLOW_ID = 'f_0'   #flow id to choose the ego vehicle from



env = SumoLaneChangeEnv(
    sumo_cfg_path="SUMO_sim/base_compl/base.sumocfg", 
    step_length=STEP_LENGTH, 
    max_steps=max_episode_steps, 
    ego_flow_id=FLOW_ID, 
                        #IDM parameters
                        idm_params=dict(
                            v0=IDM_V0, 
                            T=IDM_T, a_max=IDM_A_MAX, 
                            b_comf=IDM_B_COMF, 
                            s0=IDM_S0), 
                        #Lateral parameters
                        lateral_params=dict(
                            lane_change_duration=LANE_CHANGE_DURATION, 
                            lane_change_detection_distance=LANE_CHANGE_DETECTION_DISTANCE)) 

# (Optional) small MLP that fits a 21-D state → policy/value
policy_kwargs = dict(
    activation_fn=nn.Tanh,
    net_arch=dict(pi=[64, 64], vf=[64, 64])  # policy/value separate heads
)


os.makedirs('./logs/', exist_ok=True)
os.makedirs('./checkpoints/', exist_ok=True)

# Create a separate evaluation environment
eval_env = SumoLaneChangeEnv(
    sumo_cfg_path="SUMO_sim/base_compl/base.sumocfg", 
    step_length=STEP_LENGTH, 
    max_steps=max_episode_steps, 
    ego_flow_id=FLOW_ID, 
    idm_params=dict(
        v0=IDM_V0, 
        T=IDM_T, a_max=IDM_A_MAX, 
        b_comf=IDM_B_COMF, 
        s0=IDM_S0), 
    lateral_params=dict(
        lane_change_duration=LANE_CHANGE_DURATION, 
        lane_change_detection_distance=LANE_CHANGE_DETECTION_DISTANCE))

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

# Create callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./logs/',
    log_path='./logs/',
    eval_freq=10000,  # Evaluate every 10k steps
    deterministic=True,
    render=False,
    n_eval_episodes=10  # Run 10 episodes for evaluation
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,  # Save checkpoint every 50k steps
    save_path='./checkpoints/',
    name_prefix='ppo_sumo'
)

# Update model.learn() to include callbacks
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback]  # Add callbacks here
)
model.save(MODEL_NAME)
