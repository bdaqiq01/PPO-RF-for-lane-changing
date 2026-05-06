"""Pure safety intervention logic for lane-change decisions.

This module is intentionally TraCI-free so it can be unit-tested with synthetic
observations before wiring into the environment step loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from utils.state_extraction import OBS21_SCHEMA


RP_INTERVENTION = -2.0
DS_THRESHOLD = 10.0
MISSING_GAP_THRESHOLD = 900.0


@dataclass(frozen=True)
class SafetyDecision:
    lat_cmd_safe: int
    rp: float
    reason: str
    d0: float
    d1: float
    d2: float


def _gap(obs: np.ndarray, key: str) -> float:
    return float(obs[OBS21_SCHEMA[key]])


def _valid_gap(abs_gap: float, missing_gap_threshold: float) -> bool:
    return np.isfinite(abs_gap) and abs_gap < missing_gap_threshold


def _distance_from_gaps(gaps_abs: list[float], missing_gap_threshold: float) -> float:
    valid = [g for g in gaps_abs if _valid_gap(g, missing_gap_threshold)]
    return min(valid) if valid else float("inf")


def _action_risk_f(gaps_abs: list[float], missing_gap_threshold: float) -> float:
    """Risk score for action comparisons (higher/less negative is safer)."""
    vals = []
    for g in gaps_abs:
        if _valid_gap(g, missing_gap_threshold):
            vals.append(-1.0 / (abs(g) + 0.1))
    return min(vals) if vals else 0.0


def compute_action_distances(
    obs: np.ndarray,
    *,
    missing_gap_threshold: float = MISSING_GAP_THRESHOLD,
) -> Dict[int, float]:
    """Return D0/D1/D2 by action from longitudinal gaps in observation."""
    c0 = abs(_gap(obs, "c0.dx"))
    c1 = abs(_gap(obs, "c1.dx"))
    c2 = abs(_gap(obs, "c2.dx"))
    c3 = abs(_gap(obs, "c3.dx"))
    return {
        0: _distance_from_gaps([c1], missing_gap_threshold),
        1: _distance_from_gaps([c1, c3], missing_gap_threshold),
        2: _distance_from_gaps([c0, c2], missing_gap_threshold),
    }


def apply_safety_intervention(
    *,
    obs: np.ndarray,
    lat_cmd_raw: int,
    in_control_zone: bool,
    d_s: float = DS_THRESHOLD,
    rp_intervention: float = RP_INTERVENTION,
    missing_gap_threshold: float = MISSING_GAP_THRESHOLD,
) -> SafetyDecision:
    """
    Apply rule-based safety replacement and return safe action + intervention rp.

    Outside control zone:
      - force keep lane (0)
      - no intervention penalty (rp=0)
    """
    lat_raw = int(lat_cmd_raw)
    if lat_raw not in (0, 1, 2):
        lat_raw = 0

    d_by_action = compute_action_distances(obs, missing_gap_threshold=missing_gap_threshold)
    d0, d1, d2 = d_by_action[0], d_by_action[1], d_by_action[2]

    if not in_control_zone:
        return SafetyDecision(
            lat_cmd_safe=0,
            rp=0.0,
            reason="outside_control_zone",
            d0=d0,
            d1=d1,
            d2=d2,
        )

    unsafe = {a: (d_by_action[a] < d_s) for a in (0, 1, 2)}
    if not unsafe[lat_raw]:
        return SafetyDecision(lat_cmd_safe=lat_raw, rp=0.0, reason="none", d0=d0, d1=d1, d2=d2)

    # If all actions are unsafe, choose the least bad action by largest D.
    if unsafe[0] and unsafe[1] and unsafe[2]:
        best_action = max((0, 1, 2), key=lambda a: d_by_action[a])
        rp = rp_intervention if best_action != lat_raw else 0.0
        return SafetyDecision(
            lat_cmd_safe=int(best_action),
            rp=float(rp),
            reason="all_actions_unsafe_choose_largest_d",
            d0=d0,
            d1=d1,
            d2=d2,
        )

    if lat_raw == 0:
        lat_safe = 0
        reason = "unsafe_wait_keep_wait"
    elif lat_raw == 1:
        lat_safe = 0
        reason = "unsafe_change_wait"
    else:
        # raw abort=2: choose action with smaller penalty (larger F is safer).
        c0 = abs(_gap(obs, "c0.dx"))
        c1 = abs(_gap(obs, "c1.dx"))
        c2 = abs(_gap(obs, "c2.dx"))
        c3 = abs(_gap(obs, "c3.dx"))
        f_change = _action_risk_f([c1, c3], missing_gap_threshold)
        f_abort = _action_risk_f([c0, c2], missing_gap_threshold)
        if f_change > f_abort:
            lat_safe = 1
            reason = "unsafe_abort_switch_change"
        else:
            lat_safe = 0
            reason = "unsafe_abort_switch_wait"

    rp = rp_intervention if lat_safe != lat_raw else 0.0
    return SafetyDecision(lat_cmd_safe=int(lat_safe), rp=float(rp), reason=reason, d0=d0, d1=d1, d2=d2)
