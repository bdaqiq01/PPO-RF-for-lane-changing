from stable_baselines3.common.env_checker import check_env
from envs.sumo_lanechange_env import SumoLaneChangeEnv

env = SumoLaneChangeEnv(
    sumo_cfg_path="SUMO_sim/base_compl/base.sumocfg",
    step_length=0.1,
    max_steps=256,
    ego_flow_id="f_0",
    idm_params=dict(v0=30.0, T=1.5, a_max=2.5, b_comf=4.5, s0=2.0),
    lateral_params=dict(lane_change_duration=3, lane_change_detection_distance=10)
)

check_env(env)
env.close()
print("✅ check_env finished without errors")