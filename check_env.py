from stable_baselines3.common.env_checker import check_env
from envs.sumo_lanechange_env import SumoLaneChangeEnv

# Use the same parameters as train.py for consistency
env = SumoLaneChangeEnv(
    sumo_cfg_path="SUMO_sim/base2_compl/2lane_oneOnOff.sumocfg",
    step_length=0.2,
    max_steps=256,
    ego_flow_id="f_2",
    gate_pos=235.0,  # Required: distance in meters from start where route decision is committed
    control_zone_edge="E0.212",  # Required: edge where ego vehicle is controlled
    start_lane=1,  # lane index the ego vehicle starts in the control zone
    target_lane=0,  # lane index the ego vehicle is supposed to change to
    idm_params=dict(v0=30.0, T=1.5, a_max=2.5, b_comf=4.5, s0=2.0),
    lateral_params=dict(lane_change_duration=3, lane_change_detection_distance=10),
    exit_edge_id="E2"  # edge the ego vehicle takes after committing to lane change
)

check_env(env)
env.close()
print("✅ check_env finished without errors")