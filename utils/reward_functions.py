# utils/reward_functions.py
#
# Intended reward semantics (see SumoLaneChangeEnv._reward_for_transition):
#   - Dense Ye-style terms (comfort, efficiency, safety gaps, R_p, collision)
#     apply only when the policy controls the ego **inside the control zone**.
#   - Outside that, dense terms are zero; sparse LC shaping (success / timeout /
#     truncated-incomplete) may still apply.
#   - Terminal collision still runs full compute_reward once (large penalty).
#
from dataclasses import dataclass
import numpy as np
import traci

from utils.state_extraction import (
    IDX_PY_EGO,
    IDX_VY_EGO,
    IDX_AX_EGO,
    IDX_VX_EGO,
    START_C0,
    START_C1,
    START_C2,
    START_C3,
)


@dataclass
class RewardWeights:
    # Comfort weights (Eq. 5) - REDUCED to prevent reward instability
    alpha: float = 0.1   # weight on longitudinal jerk^2 (reduced from 1.0)
    beta: float = 0.1    # weight on lateral jerk^2 (reduced from 1.0)
    # Efficiency weights (Eq. 6) - REDUCED to prevent reward instability
    wt: float = 0.1      # time (reduced from 1.0)
    wl: float = 0.1      # lateral lane offset (reduced from 1.0)
    ws: float = 0.1      # speed deviation (reduced from 1.0 - this was the biggest penalty)


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
      obs_t, obs_tp1 : observation vectors at t and t+1 (22-d goal-conditioned
      layout from state_extraction, or any array where indices 0..20 match the
      Ye et al. ego+neighbors block; index 21 ``lane_error`` is ignored here).
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
    ax_t = float(obs_t[IDX_AX_EGO])
    ax_next = float(obs_tp1[IDX_AX_EGO])

    vx_t = float(obs_t[IDX_VX_EGO])
    vx_next = float(obs_tp1[IDX_VX_EGO])
    vy_t = float(obs_t[IDX_VY_EGO])
    vy_next = float(obs_tp1[IDX_VY_EGO])

    dt_safe = max(dt, 1e-6)

    # Longitudinal jerk ~ d(ax)/dt; lateral jerk ~ d(vy)/dt (lateral speed rate).
    j_lon = (ax_next - ax_t) / dt_safe
    j_lat = (vy_next - vy_t) / dt_safe

    R_comfort = -weights.alpha * (j_lon ** 2) - weights.beta * (j_lat ** 2)

    # ---------------- 2) EFFICIENCY (Eq. 6) ----------------
    # R_time = -Δt
    R_time = -dt

    # Lateral offset from lane center: use ego lateral lane position (Py), not Px.
    # Py* ≈ 0 at lane center in SUMO lane coordinates.
    py_next = float(obs_tp1[IDX_PY_EGO])
    R_lane = -abs(py_next)

    # Longitudinal speed: desired speed should compare to Vx, not Vy.
    R_speed = -abs(vx_next - v_desired)

    R_eff = (
        weights.wt * R_time +
        weights.wl * R_lane +
        weights.ws * R_speed
    )

    # ---------------- 3) SAFETY (near-collision + collision + R_p) ----------------
    # Neighbor blocks are [Dx, Vx, Ax, Py]. Longitudinal gaps use Dx (index +0);
    # lateral separation for F uses Py delta vs ego (index +3).
    # F(Ce, Ci) = -1 / (|Py_i - Py_e| + 0.1) using neighbor lateral positions.
    def F_lateral_py(py_neighbor: float, py_ego: float) -> float:
        d = float(py_neighbor) - float(py_ego)
        return -1.0 / (abs(d) + 0.1)

    py_ego_n = float(obs_tp1[IDX_PY_EGO])

    def neighbor_py(start: int) -> float:
        return float(obs_tp1[start + 3])

    def neighbor_dx(start: int) -> float:
        return float(obs_tp1[start + 0])

    py_c0 = neighbor_py(START_C0)
    py_c1 = neighbor_py(START_C1)
    py_c2 = neighbor_py(START_C2)
    py_c3 = neighbor_py(START_C3)

    # Overall min longitudinal distance (Eq. 8) - filter out placeholders
    gaps = []
    for start in (START_C0, START_C1, START_C2, START_C3):
        dx = neighbor_dx(start)
        if abs(dx) < 900.0:  # Only consider real vehicles
            gaps.append(abs(dx))
    D = min(gaps) if gaps else float('inf')  # If no vehicles, D = infinity

    # Check collision: ego removed from simulation
    try:
        collided = ego_id not in traci.vehicle.getIDList()
    except traci.TraCIException:
        collided = False

    # Near-collision contribution (Table II) — lateral F on Py deltas; only if gap D < ds
    R_near = 0.0
    if (not collided) and (D < ds):
        if lat_cmd == 0:  # keep lane -> target-lane leader C1
            if abs(neighbor_dx(START_C1)) < 900.0:
                R_near = F_lateral_py(py_c1, py_ego_n)
        elif lat_cmd == 1:  # change lane -> min(F(C1), F(C3))
            vals = []
            if abs(neighbor_dx(START_C1)) < 900.0:
                vals.append(F_lateral_py(py_c1, py_ego_n))
            if abs(neighbor_dx(START_C3)) < 900.0:
                vals.append(F_lateral_py(py_c3, py_ego_n))
            R_near = min(vals) if vals else 0.0
        elif lat_cmd == 2:  # abort -> min(F(C0), F(C2))
            vals = []
            if abs(neighbor_dx(START_C0)) < 900.0:
                vals.append(F_lateral_py(py_c0, py_ego_n))
            if abs(neighbor_dx(START_C2)) < 900.0:
                vals.append(F_lateral_py(py_c2, py_ego_n))
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
        # "R_route": R_route,
        "jerk_long": j_lon,
        "jerk_lat": j_lat,
        "near_gap_D": D,
        "collided": collided,
    }

    return float(R_total), components
