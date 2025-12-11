# utils/reward_functions.py

from dataclasses import dataclass
import numpy as np
import traci

from utils.state_extraction import (
    IDX_PY_EGO,
    IDX_VY_EGO,
    IDX_AY_EGO,
    IDX_PX_EGO,
    IDX_VX_EGO,
    START_C0,
    START_C1,
    START_C2,
    START_C3,
)


@dataclass
class RewardWeights:
    # Comfort weights (Eq. 5)
    alpha: float = 1.0   # weight on longitudinal jerk^2
    beta: float = 1.0    # weight on lateral jerk^2
    # Efficiency weights (Eq. 6)
    wt: float = 1.0      # time
    wl: float = 1.0      # lateral lane offset
    ws: float = 1.0      # speed deviation


def compute_reward(
    obs_t: np.ndarray,
    obs_tp1: np.ndarray,
    ego_id: str,
    lat_cmd: int,
    dt: float,
    ds: float = 10.0,          # near-collision threshold d_s
    rp: float = 0.0,           # R_p(t) from safety intervention
    v_desired: float = 25.0,   # desired longitudinal speed (~90 km/h)
    weights: RewardWeights = RewardWeights(),
):
    """
    Reward consistent with the structure in Ye et al.:

      R_total = R_comfort + R_efficiency + R_safety
              = R_comfort + R_efficiency + (R_collision + R_p)

    Inputs:
      obs_t, obs_tp1 : 21-d states at t and t+1
      ego_id         : SUMO ego vehicle id (for lane info)
      lat_cmd        : lateral action actually executed (0/1/2)
      dt             : simulation step length
      ds             : near-collision distance threshold (10 m)
      rp             : penalty added when safety module overrides action
      v_desired      : desired speed for efficiency term
      weights        : RewardWeights(alpha, beta, wt, wl, ws)

    Returns:
      total_reward (float), components (dict of terms for debugging)
    """

    # ---------------- 1) COMFORT (Eq. 5): jerk penalties ----------------
    ay_t = float(obs_t[IDX_AY_EGO])
    ay_next = float(obs_tp1[IDX_AY_EGO])

    vx_t = float(obs_t[IDX_VX_EGO])
    vx_next = float(obs_tp1[IDX_VX_EGO])

    dt_safe = max(dt, 1e-6)

    j_lon = (ay_next - ay_t) / dt_safe
    j_lat = (vx_next - vx_t) / dt_safe

    R_comfort = -weights.alpha * (j_lon ** 2) - weights.beta * (j_lat ** 2)

    # ---------------- 2) EFFICIENCY (Eq. 6) ----------------
    # R_time = -Δt
    R_time = -dt

    # R_lane = -|P_x - P_x*|
    px_next = float(obs_tp1[IDX_PX_EGO])

    # Approximate target lane center as lateral center of lane to the right (if exists)
    try:
        curr_idx = traci.vehicle.getLaneIndex(ego_id)
        road_id = traci.vehicle.getRoadID(ego_id)
        target_idx = curr_idx if curr_idx == 0 else curr_idx - 1
        target_lane_id = f"{road_id}_{target_idx}"
        _ = traci.lane.getLength(target_lane_id)  # just to assert it exists
        px_star = 0.0   # center of lane in SUMO's lateral coords
    except traci.TraCIException:
        px_star = 0.0

    R_lane = -abs(px_next - px_star)

    # R_speed = -|V_y - V_desired|
    vy_next = float(obs_tp1[IDX_VY_EGO])
    R_speed = -abs(vy_next - v_desired)

    R_eff = (
        weights.wt * R_time +
        weights.wl * R_lane +
        weights.ws * R_speed
    )

    # ---------------- 3) SAFETY (near-collision + collision + R_p) ----------------
    # F(Ce, Ci) = -1 / (|Δy_i| + 0.1), where Δy_i = Py_i - Py_e
    def F(dy: float) -> float:
        if abs(dy) > 900.0:  # Placeholder value means no vehicle
            return 0.0  # No penalty if vehicle doesn't exist
        return -1.0 / (abs(dy) + 0.1)

    # Δy_i at t+1 (post-action state)
    dy_c0 = float(obs_tp1[START_C0 + 0])
    dy_c1 = float(obs_tp1[START_C1 + 0])
    dy_c2 = float(obs_tp1[START_C2 + 0])
    dy_c3 = float(obs_tp1[START_C3 + 0])

    # Overall min longitudinal distance (Eq. 8) - filter out placeholders
    gaps = []
    for dy in [dy_c0, dy_c1, dy_c2, dy_c3]:
        if abs(dy) < 900.0:  # Only consider real vehicles
            gaps.append(abs(dy))
    D = min(gaps) if gaps else float('inf')  # If no vehicles, D = infinity

    # Check collision: ego removed from simulation
    try:
        collided = ego_id not in traci.vehicle.getIDList()
    except traci.TraCIException:
        collided = False

    # Near-collision contribution (Table II):
    R_near = 0.0
    if (not collided) and (D < ds):  # Now D correctly ignores placeholders
        if lat_cmd == 0:  # keep lane -> use target leader C1
            R_near = F(dy_c1)  # F() now returns 0.0 if no vehicle
        elif lat_cmd == 1:  # change lane -> min(F(C1), F(C3))
            vals = []
            if abs(dy_c1) < 900.0:
                vals.append(F(dy_c1))
            if abs(dy_c3) < 900.0:
                vals.append(F(dy_c3))
            R_near = min(vals) if vals else 0.0
        elif lat_cmd == 2:  # abort -> min(F(C0), F(C2))
            vals = []
            if abs(dy_c0) < 900.0:
                vals.append(F(dy_c0))
            if abs(dy_c2) < 900.0:
                vals.append(F(dy_c2))
            R_near = min(vals) if vals else 0.0


    # Collision penalty:
    #  -100 on collision, else near-collision term if D < ds, otherwise 0
    if collided:
        R_collision = -100.0
    else:
        R_collision = R_near

    # Full safety reward: R_safety = R_collision + R_p
    R_safety = R_collision + rp

    # ---------------- 4) TOTAL ----------------
    R_total = R_comfort + R_eff + R_safety

    components = {
        "R_total": R_total,
        "R_comfort": R_comfort,
        "R_efficiency": R_eff,
        "R_time": R_time,
        "R_lane": R_lane,
        "R_speed": R_speed,
        "R_collision": R_collision,
        "R_safety": R_safety,
        "R_route": R_route,
        "jerk_long": j_lon,
        "jerk_lat": j_lat,
        "near_gap_D": D,
        "collided": collided,
    }

    return float(R_total), components
