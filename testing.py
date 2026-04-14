import argparse

from envs.sumo_lanechange_env import SumoLaneChangeEnv


# Keep these aligned with train.py for apples-to-apples debugging.
STEP_LENGTH = 0.2
MAX_EPISODE_STEPS = 256
IDM_V0 = 30.0
IDM_T = 1.5
IDM_A_MAX = 2.5
IDM_B_COMF = 4.5
IDM_S0 = 2.0
LANE_CHANGE_DURATION = 3
LANE_CHANGE_DETECTION_DISTANCE = 10
FLOW_ID = "f_2"
CONTROL_ZONE_EDGE = "E0.212"
START_LANE_ID = 1
TARGET_LANE_ID = 0
EXIT_EDGE_ID = "E2"


def make_env(use_gui: bool, debug_mode: bool) -> SumoLaneChangeEnv:
    return SumoLaneChangeEnv(
        sumo_cfg_path="SUMO_sim/base2_compl/2lane_oneOnOff.sumocfg",
        step_length=STEP_LENGTH,
        max_steps=MAX_EPISODE_STEPS,
        ego_flow_id=FLOW_ID,
        control_zone_edge=CONTROL_ZONE_EDGE,
        debug_mode=debug_mode,
        use_gui=use_gui,
        start_lane=START_LANE_ID,
        target_lane=TARGET_LANE_ID,
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
        exit_edge_id=EXIT_EDGE_ID,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Layer-2 deterministic debug rollout")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument(
        "--action-mode",
        choices=["force-right", "alternate-right"],
        default="force-right",
        help="Action schedule for debugging",
    )
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base seed for reproducible per-episode resets",
    )
    return parser.parse_args()


def pick_action(step_idx: int, action_mode: str) -> int:
    # Mapping: action = lon_cmd * 3 + lat_cmd
    # lat_cmd=1 requests lane change to right.
    if action_mode == "alternate-right":
        return 1 if step_idx % 2 == 0 else 4  # same lat_cmd=1, alternate lon_cmd
    return 1  # lon_cmd=0, lat_cmd=1


def run_debug_rollout(args):
    env = make_env(use_gui=args.gui, debug_mode=True)
    total_lane_changes = 0
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            prev_lane_idx = env._safe_get_lane_index()
            prev_pending_lc = info.get("pending_lc")
            print(
                f"\n[EP {ep + 1}] reset ego_id={info.get('ego_id')} "
                f"lane_idx={prev_lane_idx} missing_neighbors={info.get('missing_neighbors')} "
                f"pending_lc={info.get('pending_lc')} "
                f"lc_start_lane={info.get('lc_start_lane')} "
                f"lc_target_lane={info.get('lc_target_lane')}"
            )

            for t in range(1, args.steps + 1):
                action = pick_action(t, args.action_mode)
                obs, reward, terminated, truncated, info = env.step(action)
                lane_idx = env._safe_get_lane_index()
                pending_lc = info.get("pending_lc")
                lc_success = info.get("lc_success")
                lc_fail_reason = info.get("lc_fail_reason")

                # Print Layer-3 FSM info for first few steps and whenever it changes/events fire.
                if (
                    t <= 5
                    or pending_lc != prev_pending_lc
                    or lc_success
                    or lc_fail_reason is not None
                ):
                    print(
                        f"[EP {ep + 1} STEP {t}] FSM "
                        f"pending_lc={pending_lc} "
                        f"lc_success={lc_success} "
                        f"lc_fail_reason={lc_fail_reason} "
                        f"lc_start_lane={info.get('lc_start_lane')} "
                        f"lc_target_lane={info.get('lc_target_lane')} "
                        f"curr_lane={info.get('curr_lane')}"
                    )

                if lane_idx != prev_lane_idx:
                    print(
                        f"[EP {ep + 1} STEP {t}] lane change: "
                        f"{prev_lane_idx} -> {lane_idx} (action={action})"
                    )
                    total_lane_changes += 1
                prev_lane_idx = lane_idx
                prev_pending_lc = pending_lc

                if terminated or truncated:
                    print(
                        f"[EP {ep + 1}] done at step={t} "
                        f"terminated={terminated} truncated={truncated} "
                        f"reason={info.get('reason')}"
                    )
                    break
    finally:
        env.close()

    print(f"\nDone. Total observed lane-index transitions: {total_lane_changes}")


if __name__ == "__main__":
    run_debug_rollout(parse_args())
