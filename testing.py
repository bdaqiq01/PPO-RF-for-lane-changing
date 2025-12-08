# testing.py
from stable_baselines3 import PPO
from envs.sumo_lanechange_env import SumoLaneChangeEnv
import numpy as np

# Use the same constants as train.py
STEP_LENGTH = 0.1
max_episode_steps = 256

# IDM parameters (same as training)
IDM_V0 = 30.0
IDM_T = 1.5
IDM_A_MAX = 2.5
IDM_B_COMF = 4.5
IDM_S0 = 2.0

# Lateral controller parameters
LANE_CHANGE_DURATION = 3
LANE_CHANGE_DETECTION_DISTANCE = 10
FLOW_ID = 'f_0'

# Model path
MODEL_NAME = "ppo_sumo_lanechange"

# Create environment (same as training)
env = SumoLaneChangeEnv(
    sumo_cfg_path="SUMO_sim/base_compl/base.sumocfg", 
    step_length=STEP_LENGTH, 
    max_steps=max_episode_steps, 
    ego_flow_id=FLOW_ID, 
    idm_params=dict(
        v0=IDM_V0, 
        T=IDM_T, 
        a_max=IDM_A_MAX, 
        b_comf=IDM_B_COMF, 
        s0=IDM_S0), 
    lateral_params=dict(
        lane_change_duration=LANE_CHANGE_DURATION, 
        lane_change_detection_distance=LANE_CHANGE_DETECTION_DISTANCE))

# Load the trained model
try:
    model = PPO.load(MODEL_NAME)
    print(f"Loaded model: {MODEL_NAME}")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_NAME}' not found!")
    print("Make sure you've trained the model first using train.py")
    exit(1)

# Evaluation metrics
n_episodes = 100
success_count = 0
collision_count = 0
timeout_count = 0
exited_count = 0
episode_rewards = []
episode_lengths = []

print(f"\nRunning {n_episodes} evaluation episodes...\n")

# Run evaluation episodes
for episode in range(n_episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0.0
    episode_length = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        
        if done:
            # Count termination reasons
            reason = info.get("reason", "unknown")
            if info.get("success", False):
                success_count += 1
            elif info.get("collision", False):
                collision_count += 1
            elif reason == "timeout":
                timeout_count += 1
            elif reason == "exited":
                exited_count += 1
    
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    
    # Print progress every 10 episodes
    if (episode + 1) % 10 == 0:
        print(f"Completed {episode + 1}/{n_episodes} episodes...")

# Print results
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Total episodes: {n_episodes}")
print(f"\nTermination reasons:")
print(f"  Success: {success_count} ({success_count/n_episodes*100:.1f}%)")
print(f"  Collision: {collision_count} ({collision_count/n_episodes*100:.1f}%)")
print(f"  Timeout: {timeout_count} ({timeout_count/n_episodes*100:.1f}%)")
print(f"  Exited: {exited_count} ({exited_count/n_episodes*100:.1f}%)")
print(f"\nEpisode statistics:")
print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"  Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
print(f"  Min reward: {np.min(episode_rewards):.2f}")
print(f"  Max reward: {np.max(episode_rewards):.2f}")
print("="*50)

# Close environment
env.close()
