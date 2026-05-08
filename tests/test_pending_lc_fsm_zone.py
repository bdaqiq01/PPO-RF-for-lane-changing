"""Layer B: zone-correct lc_success and PPO-arm metadata in the pending LC FSM (no SUMO)."""

import unittest

from envs.sumo_lanechange_env import compute_pending_lc_fsm_outcome

CZ = "E_control"
INTERNAL = ":junction_internal"


class TestPendingLcFsmZone(unittest.TestCase):
    def test_lane_complete_on_control_edge_success(self):
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=True,
            curr_lane=0,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=1,
            steps=5,
            lc_max_steps=40,
            control_zone_edge=CZ,
            curr_edge=CZ,
            lc_armed_by_ppo=True,
        )
        self.assertEqual((s, r, c), (True, None, True))

    def test_lane_complete_off_control_edge_clears_with_reason(self):
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=True,
            curr_lane=0,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=1,
            steps=5,
            lc_max_steps=40,
            control_zone_edge=CZ,
            curr_edge=INTERNAL,
            lc_armed_by_ppo=True,
        )
        self.assertEqual(s, False)
        self.assertEqual(r, "lc_completed_outside_control_zone")
        self.assertTrue(c)

    def test_lane_complete_unknown_edge_treated_as_off_zone(self):
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=True,
            curr_lane=0,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=1,
            steps=5,
            lc_max_steps=40,
            control_zone_edge=CZ,
            curr_edge=None,
            lc_armed_by_ppo=True,
        )
        self.assertEqual(r, "lc_completed_outside_control_zone")
        self.assertTrue(c)
        self.assertFalse(s)

    def test_timeout_unchanged_when_still_on_start_lane(self):
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=True,
            curr_lane=1,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=5,
            steps=5 + 21,
            lc_max_steps=20,
            control_zone_edge=CZ,
            curr_edge=INTERNAL,
        )
        self.assertEqual((s, r, c), (False, "lc_timeout", True))

    def test_not_pending_unchanged(self):
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=False,
            curr_lane=0,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=1,
            steps=9,
            lc_max_steps=40,
            control_zone_edge=CZ,
            curr_edge=CZ,
        )
        self.assertEqual((s, r, c), (False, None, False))

    def test_lane_complete_without_ppo_arm_clears(self):
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=True,
            curr_lane=0,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=1,
            steps=5,
            lc_max_steps=40,
            control_zone_edge=CZ,
            curr_edge=CZ,
            lc_armed_by_ppo=False,
        )
        self.assertFalse(s)
        self.assertEqual(r, "lc_pending_without_ppo_arm")
        self.assertTrue(c)


if __name__ == "__main__":
    unittest.main()
