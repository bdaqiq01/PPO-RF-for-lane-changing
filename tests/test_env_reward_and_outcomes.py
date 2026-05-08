"""Integration tests: PPO reward wiring and episode_outcome (requires SUMO)."""

from __future__ import annotations

import unittest

from envs.sumo_lanechange_env import SumoLaneChangeEnv
from utils.state_extraction import OBS21_SCHEMA


def _make_short_env(*, max_steps: int = 50):
    return SumoLaneChangeEnv(
        sumo_cfg_path="SUMO_sim/base2_compl/2lane_oneOnOff.sumocfg",
        step_length=0.2,
        max_steps=max_steps,
        ego_flow_id="f_2",
        control_zone_edge="E0.212",
        start_lane=1,
        target_lane=0,
        idm_params=dict(v0=30.0, T=1.5, a_max=2.5, b_comf=4.5, s0=2.0),
        lateral_params=dict(lane_change_duration=3, lane_change_detection_distance=10),
        exit_edge_id="E2",
        use_gui=False,
        debug_mode=False,
        scenarios=None,
        lc_success_bonus=10.0,
        lc_timeout_penalty=-5.0,
        truncated_incomplete_penalty=-3.0,
    )


class TestEnvRewardWiring(unittest.TestCase):
    """Item 1–2: dense reward + shaping + episode_outcome on last step."""

    def test_steps_expose_reward_fields_and_nonzero_variation(self):
        env = _make_short_env(max_steps=30)
        try:
            env.reset(seed=42)
            rewards = []
            bases = []
            for _ in range(15):
                _obs, r, term, trunc, info = env.step(0)
                rewards.append(r)
                bases.append(info["reward_base"])
                self.assertIn("reward_base", info)
                self.assertIn("reward_shaping", info)
                self.assertIn("dense_reward_applied", info)
                if not (term or trunc):
                    self.assertIsNone(
                        info["episode_outcome"],
                        "episode_outcome should be None until episode ends",
                    )
                if term or trunc:
                    self.assertIsNotNone(info["episode_outcome"])
                    break
            env.close()
        finally:
            try:
                env.close()
            except Exception:
                pass

        self.assertTrue(
            any(abs(x) > 1e-6 for x in rewards),
            f"expected some non-zero total rewards, got {rewards[:5]}...",
        )
        self.assertTrue(
            any(abs(x) > 1e-6 for x in bases),
            f"expected some non-zero reward_base, got {bases[:5]}...",
        )

    def test_truncated_incomplete_lc_sets_outcome_and_penalty(self):
        env = _make_short_env(max_steps=4)
        try:
            env.reset(seed=0)
            last_info = None
            for _ in range(4):
                _obs, _r, term, trunc, last_info = env.step(0)
                if term or trunc:
                    break
            self.assertTrue(last_info is not None)
            self.assertTrue(trunc or term)
            # Still in scenario start lane -> lane_error != 0 typically
            le = float(_obs[OBS21_SCHEMA["lane_error"]])
            if abs(le) > 1e-6:
                self.assertEqual(last_info["episode_outcome"], "truncated_incomplete_lc")
                self.assertLess(last_info["reward_shaping"], 0.0)
        finally:
            try:
                env.close()
            except Exception:
                pass

    def test_lc_success_bonus_applied_once(self):
        env = _make_short_env(max_steps=80)
        try:
            env.reset(seed=12345)
            saw_bonus = False
            for t in range(60):
                _obs, r, _term, trunc, info = env.step(1)
                if info.get("lc_success"):
                    self.assertGreaterEqual(info["reward_shaping"], env._lc_success_bonus)
                    saw_bonus = True
                    break
                if trunc:
                    break
            self.assertTrue(saw_bonus, "expected an lc_success within 60 forced-LC steps")
        finally:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
