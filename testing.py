import argparse
import csv
import json
import os
import sys

import numpy as np
import traci
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from envs.sumo_lanechange_env import SumoLaneChangeEnv
from utils.state_extraction import OBS21_SCHEMA, collect_neighbor_lane_evidence


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
EXIT_EDGE_AFTER_LEFT_LC = "E0.497"

# PPO defaults mirrored from train.py (override with CLI for quick runs)
LEARNING_RATE = 1e-4
N_STEPS = 512
BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
GLOBAL_SEED = 12345


def make_env(use_gui: bool, debug_mode: bool, **kwargs) -> SumoLaneChangeEnv:
    params = dict(
        sumo_cfg_path="SUMO_sim/base2_compl/2lane_oneOnOff.sumocfg",
        step_length=STEP_LENGTH,
        max_steps=MAX_EPISODE_STEPS,
        ego_flow_id=FLOW_ID,
        control_zone_edge=CONTROL_ZONE_EDGE,
        debug_mode=debug_mode,
        use_gui=use_gui,
        start_lane=START_LANE_ID,
        target_lane=TARGET_LANE_ID,
        idm_params=dict(v0=IDM_V0, T=IDM_T, a_max=IDM_A_MAX, b_comf=IDM_B_COMF, s0=IDM_S0),
        lateral_params=dict(
            lane_change_duration=LANE_CHANGE_DURATION,
            lane_change_detection_distance=LANE_CHANGE_DETECTION_DISTANCE,
        ),
        exit_edge_id=EXIT_EDGE_ID,
    )
    params.update(kwargs)
    return SumoLaneChangeEnv(**params)


def parse_args():
    MODE_CHOICES = ["proxy-train", "rollout", "metrics", "obs-evidence", "obs-episodes"]
    parser = argparse.ArgumentParser(
        description="Short proxy for train.py with debug-friendly settings",
        epilog="Example (no --mode):  python testing.py obs-episodes --episodes 8 --seed 42",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode_shortcut",
        nargs="?",
        default=None,
        choices=MODE_CHOICES,
        metavar="MODE",
        help="Optional shorthand for --mode, e.g. `python testing.py obs-episodes …` Overrides --mode if given.",
    )
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Global random seed")
    parser.add_argument("--timesteps", type=int, default=1024, help="Total PPO learning timesteps")
    parser.add_argument("--n-steps", type=int, default=128, help="PPO rollout horizon per update")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size")
    parser.add_argument("--eval-freq", type=int, default=512, help="Eval frequency in timesteps")
    parser.add_argument("--n-eval-episodes", type=int, default=2, help="Eval episodes per eval call")
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="proxy-train",
        help=(
            "Run mode; ignored if positional MODE is given first. "
            "proxy-train | rollout | metrics | obs-evidence | obs-episodes (see epilog)."
        ),
    )
    parser.add_argument("--episodes", type=int, default=2, help="Rollout mode episodes")
    parser.add_argument("--steps", type=int, default=120, help="Rollout mode max steps")
    parser.add_argument(
        "--action-mode",
        choices=["force-right", "alternate-right", "always-keep", "random"],
        default="force-right",
        help="Rollout mode action schedule",
    )
    parser.add_argument(
        "--evidence-csv",
        default="",
        help="obs-evidence: append one row per case to this CSV path (optional).",
    )
    parser.add_argument(
        "--evidence-json",
        default="",
        help="obs-evidence: write full evidence dicts to this JSON lines path (optional).",
    )
    parser.add_argument(
        "--flow-right-lc",
        default="f_2",
        help="flows for obs-evidence / obs-episodes: right maneuver (spawn lane 1).",
    )
    parser.add_argument(
        "--flow-left-lc",
        default="f_onramp_to_offramp",
        help=(
            "Left-maneuver ego flow (spawn filter: E0.212 lane 0). Must use a route that "
            "**enters** the control zone in lane 0 (see 2lane_oneOnOff.rou.xml): "
            "f_onramp_to_offramp from E1, not f_target_lane from E0 (those typically map to lanes 1–2)."
        ),
    )
    args = parser.parse_args()
    if args.mode_shortcut is not None:
        args.mode = args.mode_shortcut
    return args


def pick_action(step_idx: int, action_mode: str) -> int:
    if action_mode == "always-keep":
        return 0
    if action_mode == "alternate-right":
        return 1 if step_idx % 2 == 0 else 4
    if action_mode == "random":
        return int(np.random.randint(0, 6))
    return 1


def run_debug_rollout(args):
    env = make_env(use_gui=args.gui, debug_mode=True)
    total_lane_changes = 0
    total_blocked_outside_zone = 0
    total_route_commits = 0
    total_safety_interventions = 0
    total_negative_rp = 0
    try:
        for ep in range(args.episodes):
            _, info = env.reset(seed=args.seed + ep)
            prev_lane_idx = env._safe_get_lane_index()
            prev_pending_lc = info.get("pending_lc")
            print(
                f"\n[EP {ep + 1}] reset ego_id={info.get('ego_id')} "
                f"lane_idx={prev_lane_idx} pending_lc={info.get('pending_lc')}"
            )

            for t in range(1, args.steps + 1):
                action = pick_action(t, args.action_mode)
                _, _, terminated, truncated, info = env.step(action)
                lane_idx = env._safe_get_lane_index()
                pending_lc = info.get("pending_lc")
                lc_success = info.get("lc_success")
                lc_fail_reason = info.get("lc_fail_reason")
                lat_cmd_blocked = bool(info.get("lat_cmd_blocked_outside_control_zone", False))
                route_committed = bool(info.get("route_committed", False))
                safety_intervened = bool(info.get("safety_intervened", False))
                safety_reason = info.get("safety_reason")
                rp = info.get("rp")
                d0 = info.get("d0")
                d1 = info.get("d1")
                d2 = info.get("d2")

                if t <= 5 or pending_lc != prev_pending_lc or lc_success or lc_fail_reason is not None:
                    print(
                        f"[EP {ep + 1} STEP {t}] owner={info.get('controller_owner')} "
                        f"in_control_zone={info.get('in_control_zone')} "
                        f"raw={info.get('lat_cmd_raw')} applied={info.get('lat_cmd_applied')} "
                        f"pending={pending_lc} success={lc_success} fail={lc_fail_reason}"
                    )

                if safety_intervened:
                    print(
                        f"[EP {ep + 1} STEP {t}] SAFETY override: "
                        f"raw={info.get('lat_cmd_raw')} -> safe={info.get('lat_cmd_applied')} "
                        f"rp={rp} reason={safety_reason} d0={d0} d1={d1} d2={d2}"
                    )
                    total_safety_interventions += 1
                if isinstance(rp, (int, float)) and float(rp) < 0.0:
                    total_negative_rp += 1

                if lat_cmd_blocked:
                    total_blocked_outside_zone += 1
                if route_committed:
                    total_route_commits += 1
                if lane_idx != prev_lane_idx:
                    total_lane_changes += 1
                prev_lane_idx = lane_idx
                prev_pending_lc = pending_lc

                if terminated or truncated:
                    print(
                        f"[EP {ep + 1}] done step={t} terminated={terminated} truncated={truncated} "
                        f"reason={info.get('reason')}"
                    )
                    break
    finally:
        env.close()

    print(f"\nDone. lane_index_transitions={total_lane_changes}")
    print(f"control_zone_blocks={total_blocked_outside_zone}")
    print(f"safety_interventions={total_safety_interventions}")
    print(f"negative_rp_steps={total_negative_rp}")
    print(f"route_commits={total_route_commits}")


def run_proxy_train(args):
    set_random_seed(args.seed)
    train_env = Monitor(make_env(use_gui=args.gui, debug_mode=True))
    eval_env = Monitor(make_env(use_gui=args.gui, debug_mode=True))
    print("proxy train env initialized")
    print("proxy eval env initialized")

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=0,
        seed=args.seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=max(1, int(args.eval_freq)),
        deterministic=True,
        render=False,
        n_eval_episodes=max(1, int(args.n_eval_episodes)),
        verbose=0,
    )

    print(
        f"starting proxy-train: timesteps={args.timesteps}, n_steps={args.n_steps}, "
        f"batch_size={args.batch_size}, eval_freq={args.eval_freq}"
    )
    try:
        model.learn(total_timesteps=max(1, int(args.timesteps)), callback=eval_callback)
        print("proxy-train completed")
    finally:
        train_env.close()
        eval_env.close()


def _neighbor_present(obs: np.ndarray, key: str) -> bool:
    return float(obs[OBS21_SCHEMA[key]]) < 900.0


def run_obs_evidence(args):
    """
    Record observable evidence that C1/C3 use the *target* lane and lane_error matches
    SUMO semantics (left = higher index => lane_error +1 on a single-lane adjacent LC).
    """
    cases = [
        {
            "name": "right_adjacent_lc",
            "descr": "Target is one lane to the RIGHT (lower SUMO idx): lane_error should be -1.",
            "kwargs": dict(
                ego_flow_id=args.flow_right_lc,
                start_lane=1,
                exit_edge_id=EXIT_EDGE_ID,
            ),
            "reset_options": {"navigation_lane_offset": -1},
            "expect_lane_error": -1.0,
            "expect_intent": "right",
        },
        {
            "name": "left_adjacent_lc",
            "descr": "Target is one lane to the LEFT (higher SUMO idx): lane_error should be +1.",
            "kwargs": dict(
                ego_flow_id=args.flow_left_lc,
                start_lane=0,
                exit_edge_id=None,
            ),
            "reset_options": {"navigation_lane_offset": 1},
            "expect_lane_error": 1.0,
            "expect_intent": "left",
        },
    ]

    csv_path = str(args.evidence_csv or "").strip()
    json_path = str(args.evidence_json or "").strip()
    csv_f = None
    csv_w = None
    if csv_path:
        new_file = not os.path.exists(csv_path)
        csv_f = open(csv_path, "a", newline="")
        csv_w = csv.DictWriter(
            csv_f,
            fieldnames=[
                "case",
                "descr",
                "ego_id",
                "ego_flow_id",
                "curr_lane",
                "target_lane",
                "lane_error_obs",
                "lane_change_intent",
                "c1_dx",
                "c3_dx",
                "c1_present",
                "c3_present",
                "c1_on_target_lane",
                "c3_on_target_lane",
                "target_lane_id_traCI",
            ],
        )
        if new_file:
            csv_w.writeheader()

    json_lines = open(json_path, "a") if json_path else None
    failures: list[str] = []
    seed = int(args.seed)

    try:
        for case in cases:
            env = make_env(
                use_gui=args.gui,
                debug_mode=True,
                **case["kwargs"],
            )
            try:
                obs, info = env.reset(seed=seed, options=case["reset_options"])
                traci.switch(env._traci_label)
                le = float(obs[OBS21_SCHEMA["lane_error"]])
                tgt = int(env.target_lane)
                curr = info.get("current_lane_index")
                ev = collect_neighbor_lane_evidence(env.ego_id, tgt)

                row = dict(
                    case=case["name"],
                    descr=case["descr"],
                    ego_id=env.ego_id,
                    ego_flow_id=case["kwargs"]["ego_flow_id"],
                    curr_lane=curr,
                    target_lane=tgt,
                    lane_error_obs=le,
                    lane_change_intent=info.get("lane_change_intent"),
                    c1_dx=float(obs[OBS21_SCHEMA["c1.dx"]]),
                    c3_dx=float(obs[OBS21_SCHEMA["c3.dx"]]),
                    c1_present=_neighbor_present(obs, "c1.dx"),
                    c3_present=_neighbor_present(obs, "c3.dx"),
                    c1_on_target_lane=ev.get("c1_on_target_lane"),
                    c3_on_target_lane=ev.get("c3_on_target_lane"),
                    target_lane_id_traCI=ev.get("target_lane_id"),
                )

                print("\n===", case["name"], "===")
                print(case["descr"])
                print(json.dumps(row, indent=2, default=str))

                if abs(le - case["expect_lane_error"]) > 1e-5:
                    failures.append(f"{case['name']}: lane_error={le}, expected {case['expect_lane_error']}")
                if info.get("lane_change_intent") != case["expect_intent"]:
                    failures.append(
                        f"{case['name']}: intent={info.get('lane_change_intent')!r}, "
                        f"expected {case['expect_intent']!r}"
                    )
                if ev.get("c1_leader_id") is not None and ev.get("c1_on_target_lane") is not True:
                    failures.append(f"{case['name']}: C1 leader not on target lane (TraCI check failed)")
                if ev.get("c3_follower_id") is not None and ev.get("c3_on_target_lane") is not True:
                    failures.append(f"{case['name']}: C3 follower not on target lane (TraCI check failed)")

                if csv_w is not None:
                    csv_w.writerow(row)
                    csv_f.flush()
                if json_lines is not None:
                    json_lines.write(json.dumps({"case_meta": case, "row": row, "evidence": ev}, default=str) + "\n")
                    json_lines.flush()
            finally:
                env.close()
    finally:
        if csv_f is not None:
            csv_f.close()
        if json_lines is not None:
            json_lines.close()

    if failures:
        print("\nOBS-EVIDENCE FAILURES:", file=sys.stderr)
        for f in failures:
            print(" ", f, file=sys.stderr)
        sys.exit(1)
    print("\nobs-evidence: all checks passed.")


def format_obs_vector(obs: np.ndarray) -> str:
    """Pretty-print all 22 dims in schema order."""
    lines = []
    by_idx = sorted(OBS21_SCHEMA.items(), key=lambda kv: kv[1])
    flat = np.asarray(obs, dtype=np.float64).reshape(-1)
    lines.append(f"obs shape={flat.shape} compact: {np.array2string(flat, precision=6, separator=', ')}")
    for name, idx in by_idx:
        lines.append(f"  [{idx:2d}] {name:14s} {float(flat[idx]):.6f}")
    return "\n".join(lines)


def run_obs_episodes(args):
    """
    Multiple episodes: random left vs right maneuver, print full observation and TraCI checks.
    Each episode builds a fresh env with valid (start_lane, navigation_lane_offset) pairing.
    """
    rng = np.random.default_rng(int(args.seed))
    n_ep = max(1, int(args.episodes))
    failures: list[str] = []

    for ep in range(n_ep):
        go_left = bool(rng.integers(0, 2))
        if go_left:
            flow_id = args.flow_left_lc
            start_lane = 0
            nav = 1
            label = "LEFT (nav +1, start_lane 0)"
            exit_edge_id = EXIT_EDGE_AFTER_LEFT_LC
        else:
            flow_id = args.flow_right_lc
            start_lane = 1
            nav = -1
            label = "RIGHT (nav -1, start_lane 1)"
            exit_edge_id = EXIT_EDGE_ID

        env = make_env(
            use_gui=args.gui,
            debug_mode=False,
            ego_flow_id=flow_id,
            start_lane=start_lane,
            target_lane=0,
            exit_edge_id=exit_edge_id,
        )
        try:
            obs, info = env.reset(
                seed=int(args.seed) + ep,
                options={"navigation_lane_offset": nav},
            )
            traci.switch(env._traci_label)
            tgt = int(env.target_lane)
            curr = info.get("current_lane_index")
            le_obs = float(obs[OBS21_SCHEMA["lane_error"]])
            le_expected = float(tgt - int(curr)) if curr is not None else float("nan")
            ev = collect_neighbor_lane_evidence(env.ego_id, tgt)

            print("\n" + "=" * 72)
            print(f"EPISODE {ep + 1}/{n_ep}  {label}")
            print(f"  ego_id={env.ego_id}  flow={flow_id}  TraCI curr_lane={curr}  target_lane={tgt}")
            print(f"  navigation_lane_offset_applied={info.get('navigation_lane_offset_applied')}")
            print(f"  lane_error obs={le_obs:+.4f}   expected target-curr={le_expected:+.4f}")
            print("--- Full observation (22-d) ---")
            print(format_obs_vector(obs))
            print("--- TraCI neighbor resolution (C1/C3 must be on target lane if present) ---")
            print(json.dumps(ev, indent=2, default=str))

            if curr is not None and abs(le_obs - le_expected) > 1e-4:
                failures.append(
                    f"ep{ep}: lane_error mismatch obs={le_obs} expected {le_expected}"
                )
            if ev.get("c1_leader_id") is not None and ev.get("c1_on_target_lane") is not True:
                failures.append(f"ep{ep}: C1 vehicle not on target_lane_id {ev.get('target_lane_id')}")
            if ev.get("c3_follower_id") is not None and ev.get("c3_on_target_lane") is not True:
                failures.append(f"ep{ep}: C3 vehicle not on target_lane_id {ev.get('target_lane_id')}")
            if tgt == int(curr):
                failures.append(f"ep{ep}: target_lane equals curr_lane (unexpected for L/R test)")

        finally:
            env.close()

    if failures:
        print("\nOBS-EPISODES FAILURES:", file=sys.stderr)
        for f in failures:
            print(" ", f, file=sys.stderr)
        sys.exit(1)
    print(f"\nobs-episodes: {n_ep} episode(s) printed; all checks passed.")


def run_metrics(args):
    """
    Quick distribution check: are scenarios too easy/sparse for safety intervention?
    """
    set_random_seed(args.seed)
    env = make_env(use_gui=args.gui, debug_mode=False)
    episodes = max(1, int(args.episodes))
    steps_cap = max(1, int(args.steps))

    episodes_with_intervention = 0
    total_steps = 0
    total_intervention_steps = 0
    total_negative_rp_steps = 0
    total_collision_ends = 0
    total_arrived_ends = 0
    total_route_commits = 0
    d1_values = []

    try:
        for ep in range(episodes):
            _, info = env.reset(seed=args.seed + ep)
            had_intervention = False

            for t in range(1, steps_cap + 1):
                action = pick_action(t, args.action_mode)
                _, _, terminated, truncated, info = env.step(action)
                total_steps += 1

                if bool(info.get("safety_intervened", False)):
                    had_intervention = True
                    total_intervention_steps += 1
                rp = info.get("rp")
                if isinstance(rp, (int, float)) and float(rp) < 0.0:
                    total_negative_rp_steps += 1
                if bool(info.get("route_committed", False)):
                    total_route_commits += 1

                d1 = info.get("d1")
                if isinstance(d1, (int, float)) and np.isfinite(float(d1)):
                    d1_values.append(float(d1))

                if terminated or truncated:
                    reason = info.get("reason")
                    if reason == "ego_collision":
                        total_collision_ends += 1
                    if reason == "ego_arrived":
                        total_arrived_ends += 1
                    break

            if had_intervention:
                episodes_with_intervention += 1
    finally:
        env.close()

    eps_intervention_rate = episodes_with_intervention / episodes
    step_intervention_rate = (total_intervention_steps / total_steps) if total_steps > 0 else 0.0
    neg_rp_rate = (total_negative_rp_steps / total_steps) if total_steps > 0 else 0.0
    collision_rate = total_collision_ends / episodes
    arrived_rate = total_arrived_ends / episodes
    mean_d1 = float(np.mean(d1_values)) if d1_values else float("nan")

    print("\n=== Quick Safety Metrics ===")
    print(f"episodes={episodes} steps_cap={steps_cap} action_mode={args.action_mode}")
    print(f"episodes_with_intervention={episodes_with_intervention} ({eps_intervention_rate:.3f})")
    print(f"intervention_steps={total_intervention_steps}/{total_steps} ({step_intervention_rate:.3f})")
    print(f"negative_rp_steps={total_negative_rp_steps}/{total_steps} ({neg_rp_rate:.3f})")
    print(f"collision_end_rate={collision_rate:.3f} arrived_end_rate={arrived_rate:.3f}")
    print(f"route_commits={total_route_commits}")
    print(f"mean_d1_finite={mean_d1:.3f} sample_count={len(d1_values)}")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "proxy-train":
        run_proxy_train(args)
    elif args.mode == "rollout":
        run_debug_rollout(args)
    elif args.mode == "obs-evidence":
        run_obs_evidence(args)
    elif args.mode == "obs-episodes":
        run_obs_episodes(args)
    else:
        run_metrics(args)
