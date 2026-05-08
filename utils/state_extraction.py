# utils/state_extraction.py
import numpy as np
import traci

# ----------------- STATE INDEXING -----------------
# ego: (Px, Vx, Ax, Py, Vy)  — X = forward (longitudinal), Y = lateral
IDX_PX_EGO = 0  # forward position (world X for east-west road)
IDX_VX_EGO = 1  # forward speed
IDX_AX_EGO = 2  # forward acceleration
IDX_PY_EGO = 3  # lateral position (lane coordinates)
IDX_VY_EGO = 4  # lateral speed

# C0..C3 blocks of 4 for each neighbor (relative distance, speed, acceleration, lateral position)
START_C0 = 5   # current-lane leader
START_C1 = 9   # target-lane leader
START_C2 = 13  # current-lane follower
START_C3 = 17  # target-lane follower

# ----------------- LOCKED 22-D SCHEMA -----------------
# Keep this map as the single source of truth for observation indices.
# The environment and any reward/safety logic should reference this layout.
# Name kept as OBS21_SCHEMA for backward compatibility with existing imports;
# dimension is now 22 with an appended goal-conditioning feature `lane_error`.
OBS21_SCHEMA = {
    # Ego block (5)
    "ego.px": 0,  # forward position (X)
    "ego.vx": 1,  # forward speed
    "ego.ax": 2,  # forward acceleration
    "ego.py": 3,  # lateral position (Y/lane coordinates)
    "ego.vy": 4,  # lateral speed
    # Current-lane leader C0 (4)
    "c0.dx": 5,
    "c0.vx": 6,
    "c0.ax": 7,
    "c0.py": 8,
    # Target-lane leader C1 (4)
    "c1.dx": 9,
    "c1.vx": 10,
    "c1.ax": 11,
    "c1.py": 12,
    # Current-lane follower C2 (4)
    "c2.dx": 13,
    "c2.vx": 14,
    "c2.ax": 15,
    "c2.py": 16,
    # Target-lane follower C3 (4)
    "c3.dx": 17,
    "c3.vx": 18,
    "c3.ax": 19,
    "c3.py": 20,
    # Goal-conditioning feature (1): signed distance to target lane.
    # lane_error = target_lane_index - current_lane_index
    #   negative -> target is to the right (SUMO lane idx decreases right)
    #   zero     -> already in target lane
    #   positive -> target is to the left
    "lane_error": 21,
}
OBS_DIM = 22
IDX_LANE_ERROR = OBS21_SCHEMA["lane_error"]
MISSING_NEIGHBOR_BLOCK = np.array([1000.0, 0.0, 0.0, 0.0], dtype=np.float32)
MISSING_EGO_OBS = np.zeros(OBS_DIM, dtype=np.float32)


def _validate_obs(obs: np.ndarray) -> np.ndarray:
    """
    Enforce the locked observation contract:
      - exact shape (OBS_DIM,)
      - float32 dtype
      - finite values only (no NaN/Inf)
    """
    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape != (OBS_DIM,):
        raise ValueError(f"Expected observation shape ({OBS_DIM},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Observation contains non-finite values (NaN/Inf)")
    return arr


# Back-compat alias for any external callers that imported _validate_obs21.
_validate_obs21 = _validate_obs


# ----------------- HELPERS -----------------
def _fill_neighbor_block(obs: np.ndarray,
                         start_idx: int,
                         ego_Px: float,
                         veh_id: str | None,
                         ego_id: str = None) -> None:
    """
    For each neighbor, fill obs[start_idx : start_idx+4] with:
        [Dx, Vx, Ax, Py]  — X = forward (longitudinal), Y = lateral
    If vehicle/neighbor doesn't exist, use a large positive gap and zeros:
        [1000.0, 0.0, 0.0, 0.0]

    Use lane position for Dx when possible — vehicles in different lanes at
    the same forward position share the same world X, making them indistinguishable
    without lane-relative coordinates.
    """
    if veh_id is None:
        obs[start_idx:start_idx + 4] = MISSING_NEIGHBOR_BLOCK
        return

    try:
        x, y = traci.vehicle.getPosition(veh_id)      # world coords
        v = traci.vehicle.getSpeed(veh_id)            # forward speed Vx
        a = traci.vehicle.getAcceleration(veh_id)     # forward acceleration Ax
        try:
            Py = traci.vehicle.getLateralLanePosition(veh_id)  # lateral position in lane coords
        except traci.TraCIException:
            Py = y  # fallback: world Y is lateral for east-west road

        # Use lane position for Dx — consistent coordinate system required.
        # If lane position fails for either vehicle, fall back to world X for both.
        if ego_id is not None:
            try:
                ego_lane_pos = traci.vehicle.getLanePosition(ego_id)
                veh_lane_pos = traci.vehicle.getLanePosition(veh_id)
                Dx = veh_lane_pos - ego_lane_pos
            except traci.TraCIException:
                try:
                    x_e_world, _ = traci.vehicle.getPosition(ego_id)
                    Dx = x - x_e_world  # world X for both (forward on east-west road)
                except traci.TraCIException:
                    Dx = x - ego_Px  # final fallback
        else:
            Dx = x - ego_Px  # world X if ego_id not provided

        obs[start_idx:start_idx + 4] = [Dx, v, a, Py]
    except traci.TraCIException:
        obs[start_idx:start_idx + 4] = MISSING_NEIGHBOR_BLOCK


def _find_leader_follower_in_lane(ego_id: str, lane_id: str):
    """
    Find nearest leader and follower of ego in a specific lane using
    lane position (not world coordinates). Returns (leader_id, follower_id),
    each possibly None.
    
    CRITICAL FIX: Use lane position instead of world y-coordinates because
    vehicles in different lanes at the same longitudinal position have the
    same world y-coordinate (dy=0.0), making them undetectable.
    """
    # CRITICAL: Use consistent coordinate system (lane position preferred, world as fallback)
    # If ego's lane position fails, use world coordinates for ALL vehicles to avoid mixing systems
    use_lane_position = True
    try:
        pos_e = traci.vehicle.getLanePosition(ego_id)
    except traci.TraCIException:
        x_e, _ = traci.vehicle.getPosition(ego_id)  # world X = forward on east-west road
        pos_e = x_e
        use_lane_position = False

    leader_id = None
    follower_id = None
    min_front_dx = float("inf")
    max_back_dx = -float("inf")

    for vid in traci.lane.getLastStepVehicleIDs(lane_id):
        if vid == ego_id:
            continue

        if use_lane_position:
            try:
                pos = traci.vehicle.getLanePosition(vid)
                dx = pos - pos_e
            except traci.TraCIException:
                try:
                    x_e_world, _ = traci.vehicle.getPosition(ego_id)
                    x, _ = traci.vehicle.getPosition(vid)
                    dx = x - x_e_world  # world X for both (forward on east-west road)
                except traci.TraCIException:
                    continue
        else:
            try:
                x, _ = traci.vehicle.getPosition(vid)
                dx = x - pos_e  # pos_e is already world X
            except traci.TraCIException:
                continue

        if dx > 0.0 and dx < min_front_dx:
            min_front_dx = dx
            leader_id = vid
        if dx < 0.0 and dx > max_back_dx:
            max_back_dx = dx
            follower_id = vid

    return leader_id, follower_id


def get_state_with_info(
    ego_id: str,
    target_lane_index: int,
    *,
    control_zone_edge: str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Return the 22-dim state vector:
        [ego(5), C0(4), C1(4), C2(4), C3(4), lane_error(1)]
    ego:        [Px, Vx, Ax, Py, Vy]  — X = forward (longitudinal), Y = lateral
    Ci:         [Dx_i, Vx_i, Ax_i, Py_i]
    lane_error: target_lane_index - current_lane_index   (signed scalar)

    The target lane is supplied by the caller (env) rather than inferred from
    geometry. This makes the policy goal-conditioned and works for both
    left and right lane changes.

    If ``control_zone_edge`` is set and the ego's current edge differs from it,
    the **effective** target lane index for this observation is the current lane
    index (lane_error = 0, C1/C3 unused). The maneuver goal from ``target_lane_index``
    only applies on the control-zone edge — not on exit ramps, internal junction
    lanes, or downstream edges.

    Returns:
      - obs: np.ndarray with shape (OBS_DIM,)
      - state_info: dict with extraction metadata for debugging/sanity checks
    """
    if ego_id not in traci.vehicle.getIDList():
        return _validate_obs(MISSING_EGO_OBS.copy()), {
            "ego_present": False,
            "target_lane_exists": False,
            "missing_neighbors": 4,
            "state_valid": False,
            "target_lane_index": int(target_lane_index),
            "effective_target_lane_index": None,
            "lc_goal_in_obs": False,
            "current_lane_index": None,
            "lane_error": 0.0,
        }

    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # --------- EGO FEATURES (5) ---------
    x_e, y_e = traci.vehicle.getPosition(ego_id)  # world X = forward, world Y = lateral
    Vx_e = traci.vehicle.getSpeed(ego_id)         # forward speed
    Ax_e = traci.vehicle.getAcceleration(ego_id)  # forward acceleration

    try:
        Py_e = traci.vehicle.getLateralLanePosition(ego_id)  # lateral position in lane coords
    except traci.TraCIException:
        Py_e = y_e  # fallback: world Y is lateral for east-west road

    try:
        Vy_e = traci.vehicle.getLateralSpeed(ego_id)  # lateral speed
    except (AttributeError, traci.TraCIException):
        Vy_e = 0.0

    obs[IDX_PX_EGO] = x_e
    obs[IDX_VX_EGO] = Vx_e
    obs[IDX_AX_EGO] = Ax_e
    obs[IDX_PY_EGO] = Py_e
    obs[IDX_VY_EGO] = Vy_e

    # --------- SURROUNDING VEHICLES (4 x 4) ---------
    curr_lane_id = traci.vehicle.getLaneID(ego_id)
    curr_edge_id = traci.vehicle.getRoadID(ego_id)
    curr_lane_index = traci.vehicle.getLaneIndex(ego_id)

    # Lane ids are "{edge}_{laneIndex}". Target lane index is defined for the control-zone
    # maneuver; on edges with fewer lanes (e.g. E2 has only E2_0) or on internal junction
    # edges, indices like "_1" do not exist — do not call TraCI with invalid lane ids.
    n_lanes_here = int(traci.edge.getLaneNumber(curr_edge_id))
    cl = int(curr_lane_index)
    tl_raw = int(target_lane_index)
    if control_zone_edge is not None and curr_edge_id != control_zone_edge:
        # Outside the control zone: no LC goal in the observation (same as "already in target").
        tl = cl
    else:
        tl = tl_raw
    target_lane_id = None
    if 0 <= tl < n_lanes_here:
        try:
            candidate_id = f"{curr_edge_id}_{tl}"
            _ = traci.lane.getLength(candidate_id)
            target_lane_id = candidate_id
        except traci.TraCIException:
            target_lane_id = None

    # C0 / C2: current-lane leader & follower
    c0, c2 = _find_leader_follower_in_lane(ego_id, curr_lane_id)

    # C1 / C3: target-lane leader & follower (if target lane exists on this edge)
    if target_lane_id is not None and tl != cl:
        c1, c3 = _find_leader_follower_in_lane(ego_id, target_lane_id)
    else:
        # Already in target lane (or target lane not present on this edge):
        # C1/C3 collapse to "missing neighbor" so safety logic treats the
        # change-lane action as free (it's a no-op anyway).
        c1 = c3 = None

    # Fill each 4-d block [Dx, Vx, Ax, Py] — pass x_e as world-X forward fallback
    _fill_neighbor_block(obs, START_C0, x_e, c0, ego_id)
    _fill_neighbor_block(obs, START_C1, x_e, c1, ego_id)
    _fill_neighbor_block(obs, START_C2, x_e, c2, ego_id)
    _fill_neighbor_block(obs, START_C3, x_e, c3, ego_id)

    # --------- GOAL FEATURE: lane_error ---------
    # Only meaningful while the target lane index exists on the current edge; otherwise
    # treat as satisfied on this edge (e.g. post-commit single-lane exit E2).
    if 0 <= tl < n_lanes_here:
        obs[IDX_LANE_ERROR] = float(tl - cl)
    else:
        obs[IDX_LANE_ERROR] = 0.0

    validated = _validate_obs(obs)
    missing_neighbors = sum(v is None for v in (c0, c1, c2, c3))
    lc_goal_in_obs = not (
        control_zone_edge is not None and curr_edge_id != control_zone_edge
    )
    return validated, {
        "ego_present": True,
        "target_lane_exists": target_lane_id is not None,
        "missing_neighbors": int(missing_neighbors),
        "state_valid": True,
        "target_lane_index": int(tl_raw),
        "effective_target_lane_index": int(tl),
        "lc_goal_in_obs": bool(lc_goal_in_obs),
        "current_lane_index": int(curr_lane_index),
        "lane_error": float(obs[IDX_LANE_ERROR]),
        "n_lanes_on_edge": int(n_lanes_here),
    }


# ----------------- MAIN STATE EXTRACTION -----------------
def get_state(
    ego_id: str,
    target_lane_index: int,
    *,
    control_zone_edge: str | None = None,
) -> np.ndarray:
    """
    Backward-compatible wrapper returning only the observation vector.
    """
    obs, _ = get_state_with_info(
        ego_id, target_lane_index, control_zone_edge=control_zone_edge
    )
    return obs


def collect_neighbor_lane_evidence(
    ego_id: str,
    target_lane_index: int,
    *,
    control_zone_edge: str | None = None,
) -> dict:
    """
    TraCI-backed facts for tests and reports: which lanes C0–C3 are resolved on.

    SUMO convention (driving direction): lane index **increases to the left** of the
    vehicle; index 0 is the rightmost lane on the edge. With
    ``lane_error = target_lane_index - current_lane_index``, a **left** adjacent
    target has ``lane_error == +1`` and a **right** adjacent target has ``lane_error == -1``.

    Returns:
        Dict with lane ids, neighbor vehicle ids, and booleans that C1/C3 vehicles
        lie on the resolved target lane id (when those neighbors exist).
    """
    out: dict = {
        "ego_present": False,
        "curr_edge_id": None,
        "curr_lane_idx": None,
        "curr_lane_id": None,
        "target_lane_idx": int(target_lane_index),
        "target_lane_id": None,
        "target_lane_exists": False,
        "n_lanes_on_edge": None,
        "c0_leader_id": None,
        "c1_leader_id": None,
        "c2_follower_id": None,
        "c3_follower_id": None,
        "c1_on_target_lane": None,
        "c3_on_target_lane": None,
    }
    if ego_id not in traci.vehicle.getIDList():
        return out
    out["ego_present"] = True
    try:
        curr_lane_id = traci.vehicle.getLaneID(ego_id)
        curr_edge_id = traci.vehicle.getRoadID(ego_id)
        curr_lane_idx = int(traci.vehicle.getLaneIndex(ego_id))
        n_lanes = int(traci.edge.getLaneNumber(curr_edge_id))
    except traci.TraCIException:
        return out

    out["curr_lane_id"] = str(curr_lane_id)
    out["curr_edge_id"] = str(curr_edge_id)
    out["curr_lane_idx"] = curr_lane_idx
    out["n_lanes_on_edge"] = n_lanes

    tl_raw = int(target_lane_index)
    if control_zone_edge is not None and curr_edge_id != control_zone_edge:
        tl = int(curr_lane_idx)
    else:
        tl = tl_raw
    out["target_lane_idx_effective"] = tl

    target_lane_id = None
    if 0 <= tl < n_lanes:
        try:
            candidate = f"{curr_edge_id}_{tl}"
            _ = traci.lane.getLength(candidate)
            target_lane_id = candidate
        except traci.TraCIException:
            target_lane_id = None
    out["target_lane_id"] = target_lane_id
    out["target_lane_exists"] = target_lane_id is not None

    c0, c2 = _find_leader_follower_in_lane(ego_id, curr_lane_id)
    if target_lane_id is not None and tl != curr_lane_idx:
        c1, c3 = _find_leader_follower_in_lane(ego_id, target_lane_id)
    else:
        c1 = c3 = None

    out["c0_leader_id"] = c0
    out["c1_leader_id"] = c1
    out["c2_follower_id"] = c2
    out["c3_follower_id"] = c3

    def _vid_on_lane(vid: str | None, lane_id: str | None) -> bool | None:
        if vid is None or lane_id is None:
            return None
        try:
            return str(traci.vehicle.getLaneID(vid)) == str(lane_id)
        except traci.TraCIException:
            return None

    out["c1_on_target_lane"] = _vid_on_lane(c1, target_lane_id)
    out["c3_on_target_lane"] = _vid_on_lane(c3, target_lane_id)
    return out
