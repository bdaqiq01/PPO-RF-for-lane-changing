"""Unit tests for Layer-3 pending lane-change FSM (no SUMO required)."""

import unittest

from envs.sumo_lanechange_env import compute_pending_lc_fsm_outcome

_CZ = "control_zone_edge_1"


class TestPendingLcFsm(unittest.TestCase):
    def test_not_pending_noop(self):
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=False,
            curr_lane=1,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=5,
            steps=10,
            lc_max_steps=20,
            control_zone_edge=_CZ,
        )
        self.assertEqual((s, r, c), (False, None, False))

    def test_pending_unknown_lane_no_clear(self):
        """If lane index is unknown, do not clear or mark failure."""
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=True,
            curr_lane=None,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=5,
            steps=10,
            lc_max_steps=20,
            control_zone_edge=_CZ,
        )
        self.assertEqual((s, r, c), (False, None, False))

    def test_merging_no_fail_reason(self):
        """Mid-merge: still on start lane, before timeout — no failure, no clear."""
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=True,
            curr_lane=1,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=5,
            steps=10,
            lc_max_steps=20,
            control_zone_edge=_CZ,
        )
        self.assertEqual((s, r, c), (False, None, False))

    def test_merging_partial_lane_change_not_at_target(self):
        """Lane index changed but not yet target — still merging."""
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=True,
            curr_lane=2,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=5,
            steps=10,
            lc_max_steps=20,
            control_zone_edge=_CZ,
        )
        self.assertEqual((s, r, c), (False, None, False))

    def test_success_reaches_target(self):
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=True,
            curr_lane=0,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=5,
            steps=12,
            lc_max_steps=20,
            control_zone_edge=_CZ,
            curr_edge=_CZ,
        )
        self.assertEqual((s, r, c), (True, None, True))

    def test_timeout_clears(self):
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=True,
            curr_lane=1,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=5,
            steps=5 + 21,
            lc_max_steps=20,
            control_zone_edge=_CZ,
        )
        self.assertEqual((s, r, c), (False, "lc_timeout", True))

    def test_timeout_boundary_not_triggered(self):
        s, r, c = compute_pending_lc_fsm_outcome(
            pending_lc=True,
            curr_lane=1,
            lc_start_lane=1,
            lc_target_lane=0,
            lc_start_step=5,
            steps=5 + 20,
            lc_max_steps=20,
            control_zone_edge=_CZ,
        )
        self.assertEqual((s, r, c), (False, None, False))


if __name__ == "__main__":
    unittest.main()
