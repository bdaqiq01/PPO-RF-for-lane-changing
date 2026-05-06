import unittest

import numpy as np

from utils.safety_intervention import apply_safety_intervention, compute_action_distances


def _obs(c0: float = 1000.0, c1: float = 1000.0, c2: float = 1000.0, c3: float = 1000.0) -> np.ndarray:
    arr = np.zeros(21, dtype=np.float32)
    arr[5] = c0
    arr[9] = c1
    arr[13] = c2
    arr[17] = c3
    return arr


class TestSafetyIntervention(unittest.TestCase):
    def test_compute_action_distances(self):
        d = compute_action_distances(_obs(c0=-15, c1=8, c2=-12, c3=-6))
        self.assertEqual(d[0], 8.0)
        self.assertEqual(d[1], 6.0)
        self.assertEqual(d[2], 12.0)

    def test_outside_control_zone_forces_keep_no_penalty(self):
        out = apply_safety_intervention(obs=_obs(c1=2.0), lat_cmd_raw=1, in_control_zone=False)
        self.assertEqual(out.lat_cmd_safe, 0)
        self.assertEqual(out.rp, 0.0)
        self.assertEqual(out.reason, "outside_control_zone")

    def test_safe_action_passes_through(self):
        out = apply_safety_intervention(obs=_obs(c1=20.0, c3=-20.0), lat_cmd_raw=1, in_control_zone=True)
        self.assertEqual(out.lat_cmd_safe, 1)
        self.assertEqual(out.rp, 0.0)
        self.assertEqual(out.reason, "none")

    def test_unsafe_change_switches_to_wait(self):
        out = apply_safety_intervention(obs=_obs(c1=4.0, c3=-30.0), lat_cmd_raw=1, in_control_zone=True, d_s=10.0)
        self.assertEqual(out.lat_cmd_safe, 0)
        self.assertLess(out.rp, 0.0)
        self.assertEqual(out.reason, "unsafe_change_wait")

    def test_unsafe_wait_stays_wait_no_override_penalty(self):
        out = apply_safety_intervention(obs=_obs(c1=4.0), lat_cmd_raw=0, in_control_zone=True, d_s=10.0)
        self.assertEqual(out.lat_cmd_safe, 0)
        self.assertEqual(out.rp, 0.0)
        self.assertEqual(out.reason, "unsafe_wait_keep_wait")

    def test_unsafe_abort_prefers_change_when_change_safer(self):
        # change uses c1/c3 = 20, 25 => safer than abort using c0/c2 = 3, 4
        out = apply_safety_intervention(
            obs=_obs(c0=3.0, c1=20.0, c2=-4.0, c3=-25.0),
            lat_cmd_raw=2,
            in_control_zone=True,
            d_s=10.0,
        )
        self.assertEqual(out.lat_cmd_safe, 1)
        self.assertLess(out.rp, 0.0)
        self.assertEqual(out.reason, "unsafe_abort_switch_change")

    def test_all_unsafe_choose_largest_d(self):
        # D0=7, D1=min(7,9)=7, D2=min(5,8)=5 => choose action 0
        out = apply_safety_intervention(
            obs=_obs(c0=5.0, c1=7.0, c2=-8.0, c3=-9.0),
            lat_cmd_raw=2,
            in_control_zone=True,
            d_s=10.0,
        )
        self.assertEqual(out.lat_cmd_safe, 0)
        self.assertLess(out.rp, 0.0)
        self.assertEqual(out.reason, "all_actions_unsafe_choose_largest_d")

    def test_missing_neighbors_not_unsafe(self):
        out = apply_safety_intervention(obs=_obs(), lat_cmd_raw=1, in_control_zone=True, d_s=10.0)
        self.assertEqual(out.lat_cmd_safe, 1)
        self.assertEqual(out.rp, 0.0)
        self.assertTrue(np.isinf(out.d0))
        self.assertTrue(np.isinf(out.d1))
        self.assertTrue(np.isinf(out.d2))


if __name__ == "__main__":
    unittest.main()
