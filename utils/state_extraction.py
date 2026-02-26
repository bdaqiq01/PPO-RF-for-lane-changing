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
        obs[start_idx:start_idx + 4] = [1000.0, 0.0, 0.0, 0.0]
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
        obs[start_idx:start_idx + 4] = [1000.0, 0.0, 0.0, 0.0]


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


def _get_target_lane_index(ego_id: str) -> int:
    """
    Mandatory route: main -> off-ramp.
    In the paper's network the off-ramp is on the right, so the target
    lane is always the lane immediately to the RIGHT of the ego
    (if such a lane exists).
    """
    curr_idx = traci.vehicle.getLaneIndex(ego_id)
    # SUMO uses 0 = right-most lane, larger index = more to the left.
    if curr_idx > 0:
        return curr_idx - 1
    else:
        return curr_idx  # already in right-most / target lane


# ----------------- MAIN STATE EXTRACTION -----------------
def get_state(ego_id: str) -> np.ndarray:
    """
    Return the 21-dim state vector used in Ye et al.:
        [ego(5), C0(4), C1(4), C2(4), C3(4)]
    ego:  [Px, Vx, Ax, Py, Vy]  — X = forward (longitudinal), Y = lateral
    Ci:   [Dx_i, Vx_i, Ax_i, Py_i]

    Returns zeros if ego_id is not in the simulation.
    """
    if ego_id not in traci.vehicle.getIDList():
        return np.zeros(21, dtype=np.float32)

    obs = np.zeros(21, dtype=np.float32)

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
    target_lane_index = _get_target_lane_index(ego_id)

    target_lane_id = None
    try:
        target_lane_id = f"{curr_edge_id}_{target_lane_index}"
        _ = traci.lane.getLength(target_lane_id)
    except traci.TraCIException:
        target_lane_id = None

    # C0 / C2: current-lane leader & follower
    c0, c2 = _find_leader_follower_in_lane(ego_id, curr_lane_id)

    # C1 / C3: target-lane leader & follower (if target lane exists)
    if target_lane_id is not None:
        c1, c3 = _find_leader_follower_in_lane(ego_id, target_lane_id)
    else:
        c1 = c3 = None

    # Fill each 4-d block [Dx, Vx, Ax, Py] — pass x_e as world-X forward fallback
    _fill_neighbor_block(obs, START_C0, x_e, c0, ego_id)
    _fill_neighbor_block(obs, START_C1, x_e, c1, ego_id)
    _fill_neighbor_block(obs, START_C2, x_e, c2, ego_id)
    _fill_neighbor_block(obs, START_C3, x_e, c3, ego_id)

    return obs
