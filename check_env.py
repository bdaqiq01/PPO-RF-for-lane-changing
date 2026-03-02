from stable_baselines3.common.env_checker import check_env
from envs.sumo_lanechange_env import SumoLaneChangeEnv

env = SumoLaneChangeEnv(
    sumo_cfg_path="SUMO_sim/base2_compl/2lane_oneOnOff.sumocfg",
    step_length=0.2,
    max_steps=256,
    ego_flow_id="f_2",
    control_zone_edge="E0.212",
    start_lane=1,
    target_lane=0,
    debug_mode=False,
)

check_env(env, warn=True)
env.close()
print("check_env finished without errors")
