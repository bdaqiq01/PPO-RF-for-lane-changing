# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from torch import nn
import gym  # not strictly needed, but OK to leave
import torch  # not strictly needed either, but OK
from envs.sumo_lanechange_env import SumoLaneChangeEnv
import os

# ----------------- Hyperparameters -----------------

STEP_LENGTH = 0.2  # step length in seconds for SUMO and environment step

LEARNING_RATE = 1e-4
N_STEPS = 512          # env steps per env before each PPO update
BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TOTAL_TIMESTEPS = 200_000  # total timesteps to train the model
MODEL_NAME = "ppo_sumo_lanechange"
max_episode_steps = 256     # max steps per episode

# --- IDM longitudinal controller (from paper) ---
IDM_V0 = 30.0     # m/s
IDM_T = 1.5       # s
IDM_A_MAX = 2.5   # m/s^2
IDM_B_COMF = 4.5  # m/s^2 (positive; used as braking limit)
IDM_S0 = 2.0      # m

# --- Lateral controller ---
LANE_CHANGE_DURATION = 3   # seconds lane change duration
LANE_CHANGE_DETECTION_DISTANCE = 10  # meters lane change detection distance
FLOW_ID = "f_0"            # flow id to choose the ego vehicle from

RANK_NUM = 4  # number of parallel environments


# ----------------- Env factory for SubprocVecEnv -----------------

def make_env(rank: int):
    """
    Factory that returns a function creating a fresh SumoLaneChangeEnv.
    `rank` is currently unused, but it's useful if you later want per-env seeds or logging.
    """
    def _init():
        env_ins = SumoLaneChangeEnv(
            sumo_cfg_path="SUMO_sim/base_compl/base.sumocfg",
            step_length=STEP_LENGTH,
            max_steps=max_episode_steps,
            ego_flow_id=FLOW_ID,
            # IDM parameters
            idm_params=dict(
                v0=IDM_V0,
                T=IDM_T,
                a_max=IDM_A_MAX,
                b_comf=IDM_B_COMF,
                s0=IDM_S0,
            ),
            # Lateral parameters
            lateral_params=dict(
                lane_change_duration=LANE_CHANGE_DURATION,
                lane_change_detection_distance=LANE_CHANGE_DETECTION_DISTANCE,
            ),
        )
        return env_ins

    return _init


# ----------------- Main (Windows-safe multiprocessing) -----------------

if __name__ == "__main__":
    # Make sure log dirs exist
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./checkpoints/", exist_ok=True)

    # --- 1) Create vectorized training environment ---
    # This will launch RANK_NUM SUMO instances in parallel subprocesses
    env = SubprocVecEnv([make_env(i) for i in range(RANK_NUM)])

    # --- 2) Create separate evaluation environment (single instance) ---
    eval_env = SumoLaneChangeEnv(
        sumo_cfg_path="SUMO_sim/base_compl/base.sumocfg",
        step_length=STEP_LENGTH,
        max_steps=max_episode_steps,
        ego_flow_id=FLOW_ID,
        idm_params=dict(
            v0=IDM_V0,
            T=IDM_T,
            a_max=IDM_A_MAX,
            b_comf=IDM_B_COMF,
            s0=IDM_S0,
        ),
        lateral_params=dict(
            lane_change_duration=LANE_CHANGE_DURATION,
            lane_change_detection_distance=LANE_CHANGE_DETECTION_DISTANCE,
        ),
    )

    # --- 3) Policy architecture ---
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),  # separate policy/value networks
    )

    # --- 4) Create PPO model ---
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,           # per-env steps; total batch size per update = N_STEPS * RANK_NUM
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # --- 5) Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10_000,      # Evaluate every 10k training steps
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,      # Save checkpoint every 50k steps
        save_path="./checkpoints/",
        name_prefix="ppo_sumo",
    )

    # --- 6) Train ---
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
    )

    # --- 7) Save final model ---
    model.save(MODEL_NAME)
