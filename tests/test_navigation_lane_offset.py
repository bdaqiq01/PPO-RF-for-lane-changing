"""Unit tests for navigation lane offset → absolute target (no SUMO process)."""

from __future__ import annotations

import unittest

from envs.sumo_lanechange_env import SumoLaneChangeEnv


def _bare_env() -> SumoLaneChangeEnv:
    return SumoLaneChangeEnv(
        sumo_cfg_path="SUMO_sim/base2_compl/2lane_oneOnOff.sumocfg",
        step_length=0.2,
        max_steps=10,
        ego_flow_id="f_2",
        control_zone_edge="E0.212",
        start_lane=1,
        target_lane=0,
        use_gui=False,
        debug_mode=False,
    )


class TestNavigationLaneOffset(unittest.TestCase):
    def test_plus_one_left(self):
        env = _bare_env()
        d, t = env._navigation_lane_delta_to_target(0, 3, 1)
        self.assertEqual(d, "left")
        self.assertEqual(t, 1)

    def test_minus_one_right(self):
        env = _bare_env()
        d, t = env._navigation_lane_delta_to_target(1, 3, -1)
        self.assertEqual(d, "right")
        self.assertEqual(t, 0)

    def test_zero(self):
        env = _bare_env()
        d, t = env._navigation_lane_delta_to_target(1, 2, 0)
        self.assertIsNone(d)
        self.assertEqual(t, 1)

    def test_out_of_bounds_raises(self):
        env = _bare_env()
        with self.assertRaises(RuntimeError):
            env._navigation_lane_delta_to_target(0, 2, -1)


if __name__ == "__main__":
    unittest.main()
