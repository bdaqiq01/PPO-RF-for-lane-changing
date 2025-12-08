from dataclasses import dataclass
import math
from typing import Optional

@dataclass
class EgoState:
    # longitudinal = along the road; lateral = across lanes
    lon_pos: float       # Py_e in paper (longitudinal position)
    lat_pos: float       # Px in paper (lateral position)
    lon_speed: float
    lat_speed: float
    lon_accel: float
    lat_accel: float

@dataclass
class OtherVehState:
    lon_pos: float       # Py_i in paper
    lon_speed: float
    lon_accel: float
    lat_pos: float

@dataclass
class RewardWeights:
    # Comfort weights (Eq. 5)
    alpha: float = 1.0   # weight on longitudinal jerk^2
    beta: float  = 1.0   # weight on lateral jerk^2
    # Efficiency weights (Eq. 6)
    wt: float = 1.0      # time
    wl: float = 1.0      # lateral lane offset
    ws: float = 1.0      # speed deviation

def compute_reward(
    ego_t: EgoState,
    ego_tm1: EgoState,                 # previous step for jerk calc
    dt: float,                         # step length
    # Surrounding vehicles C0..C3 (Fig. 2 in paper)
    C0_current_leader: Optional[OtherVehState],
    C1_target_leader:  Optional[OtherVehState],
    C2_current_follower: Optional[OtherVehState],
    C3_target_follower: Optional[OtherVehState],
    # Task/Scenario targets for efficiency (Eq. 6)
    target_lat_pos: float,             # P_x* (center of target lane)
    desired_lon_speed: float,          # V_desired (speed limit or target)
    # Chosen high-level action (Table I): lateral ∈ {0,1,2}, longitudinal ∈ {0,1}
    lateral_action: int,               # 0: keep lane, 1: change, 2: abort
    longitudinal_action: int,          # 0: follow current-lane leader, 1: follow target-lane leader
    # Safety params (Eqs. 7–9)
    near_collision_thresh_m: float = 10.0,  # d_s
    collided: bool = False,                 # true if collision occurred this step
    safety_intervention: bool = False,      # true if catastrophic action overridden
    safety_penalty: float = 0.0,            # R_p(t)
    # Weights
    w: RewardWeights = RewardWeights()
):
    """
    Returns:
        total_reward, dict_of_components
    """
    # ------------------------------
    # Comfort (Eq. 5): R_comfort = -α * a_dot_long^2 - β * a_dot_lat^2
    # ------------------------------
    # numeric jerk ≈ (a_t - a_{t-1}) / dt
    j_lon = (ego_t.lon_accel - ego_tm1.lon_accel) / max(dt, 1e-6)
    j_lat = (ego_t.lat_accel - ego_tm1.lat_accel) / max(dt, 1e-6)
    R_comfort = - w.alpha * (j_lon ** 2) - w.beta * (j_lat ** 2)  # :contentReference[oaicite:0]{index=0}

    # ------------------------------
    # Efficiency (Eq. 6)
    # R_time = -Δt
    # R_lane = -|P_x - P_x*|
    # R_speed = -|V_y - V_desired|
    # R_eff  = w_t * R_time + w_l * R_lane + w_s * R_speed
    # ------------------------------
    R_time  = -dt
    R_lane  = -abs(ego_t.lat_pos - target_lat_pos)
    R_speed = -abs(ego_t.lon_speed - desired_lon_speed)
    R_eff   = w.wt * R_time + w.wl * R_lane + w.ws * R_speed      # :contentReference[oaicite:1]{index=1}

    # ------------------------------
    # Safety (Eqs. 7–9) with Table II logic
    # F(Ce, Ci) = -1 / (|Py_e - Py_i| + 0.1)
    # R_collision = near-collision term if D < d_s, else 0; if collision then -100
    # R_safety = R_collision + R_p
    # ------------------------------
    def F(ego_lon_pos: float, other: Optional[OtherVehState]) -> float:
        if other is None:
            return 0.0
        return -1.0 / (abs(ego_lon_pos - other.lon_pos) + 0.1)     # :contentReference[oaicite:2]{index=2}

    # pick which neighbors contribute, per Table II (depends on lateral action)
    # lateral 0 (keep): near-collision only from C1 (target leader)
    # lateral 1 (change): min(F(Ce,C1), F(Ce,C3))
    # lateral 2 (abort):  min(F(Ce,C0), F(Ce,C2))
    if lateral_action == 0:
        near_term = F(ego_t.lon_pos, C1_target_leader)
    elif lateral_action == 1:
        vals = []
        if C1_target_leader is not None: vals.append(F(ego_t.lon_pos, C1_target_leader))
        if C3_target_follower is not None: vals.append(F(ego_t.lon_pos, C3_target_follower))
        near_term = min(vals) if vals else 0.0
    elif lateral_action == 2:
        vals = []
        if C0_current_leader is not None: vals.append(F(ego_t.lon_pos, C0_current_leader))
        if C2_current_follower is not None: vals.append(F(ego_t.lon_pos, C2_current_follower))
        near_term = min(vals) if vals else 0.0
    else:
        near_term = 0.0  # invalid action safeguard

    # Determine longitudinal distance to the most relevant vehicle to test near-collision threshold
    # A simple choice: use the same vehicle(s) we used for F and compute their min |Δlon|
    def min_longitudinal_gap():
        gaps = []
        def add_gap(other):
            if other is not None:
                gaps.append(abs(ego_t.lon_pos - other.lon_pos))
        if lateral_action == 0:
            add_gap(C1_target_leader)
        elif lateral_action == 1:
            add_gap(C1_target_leader); add_gap(C3_target_follower)
        elif lateral_action == 2:
            add_gap(C0_current_leader); add_gap(C2_current_follower)
        return min(gaps) if gaps else float('inf')

    D = min_longitudinal_gap()
    if collided:
        R_collision = -100.0                                           # :contentReference[oaicite:3]{index=3}
    elif D < near_collision_thresh_m:
        R_collision = near_term                                        # near-collision penalty table + F()  :contentReference[oaicite:4]{index=4}
    else:
        R_collision = 0.0

    R_safety = R_collision + ( -abs(safety_penalty) if safety_intervention else 0.0 )  # Eq. (9)  :contentReference[oaicite:5]{index=5}

    # ------------------------------
    # Total reward (sum of components used by the paper)
    # ------------------------------
    R_total = R_comfort + R_eff + R_safety

    components = {
        "R_comfort": R_comfort,
        "R_efficiency": R_eff,
        "R_time": R_time,
        "R_lane": R_lane,
        "R_speed": R_speed,
        "R_collision": R_collision,
        "R_safety": R_safety,
        "jerk_long": j_lon,
        "jerk_lat": j_lat,
        "near_gap_D": D
    }
    return R_total, components
