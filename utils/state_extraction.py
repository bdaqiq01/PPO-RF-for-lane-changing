# utils/state_extraction.py
import numpy as np
import traci

# ----------------- STATE INDEXING -----------------
# ego: (Py, Vy, Ay, Px, Vx)
IDX_PY_EGO = 0  # longitudinal position
IDX_VY_EGO = 1  # longitudinal speed
IDX_AY_EGO = 2  # longitudinal acceleration
IDX_PX_EGO = 3  # lateral position
IDX_VX_EGO = 4  # lateral speed

# C0..C3 blocks of 4 for each neighbor (relative distance, speed, acceleration, lateral position)
START_C0 = 5   # current-lane leader
START_C1 = 9   # target-lane leader
START_C2 = 13  # current-lane follower
START_C3 = 17  # target-lane follower


# ----------------- HELPERS -----------------
def _fill_neighbor_block(obs: np.ndarray,
                         start_idx: int,
                         ego_Py: float,
                         veh_id: str | None,
                         ego_id: str = None) -> None:
    """
    For each neighbor, fill obs[start_idx : start_idx+4] with:
        [Δy, Vy, Ay, Px]
    If vehicle/neighbor doesn't exist, use a large positive gap and zeros:
        [1000.0, 0.0, 0.0, 0.0]
    
    CRITICAL FIX: Use lane position instead of world y-coordinates for Dy when possible.
    Vehicles in different lanes at the same longitudinal position have the same
    world y-coordinate (dy=0.0), making them undetectable.
    """
    if veh_id is None:
        obs[start_idx:start_idx + 4] = [1000.0, 0.0, 0.0, 0.0]
        return

    try:
        x, y = traci.vehicle.getPosition(veh_id)      # world coords
        v = traci.vehicle.getSpeed(veh_id)            # longitudinal speed Vy
        a = traci.vehicle.getAcceleration(veh_id)     # longitudinal acceleration Ay
        try:
            Px = traci.vehicle.getLateralLanePosition(veh_id)  # lateral position in lane coords
        except traci.TraCIException:
            Px = x  # fallback: world x as lateral

        # CRITICAL FIX: Use lane position for Dy calculation when ego_id is provided
        # This correctly handles vehicles in different lanes at the same world y-coordinate
        # IMPORTANT: Must use consistent coordinate system - if lane position fails for either,
        # fall back to world coordinates for BOTH to avoid mixing systems
        if ego_id is not None:
            try:
                ego_lane_pos = traci.vehicle.getLanePosition(ego_id)
                veh_lane_pos = traci.vehicle.getLanePosition(veh_id)
                Dy = veh_lane_pos - ego_lane_pos  # Use lane position (correct for different lanes)
            except traci.TraCIException:
                # Lane position failed for at least one vehicle, fall back to world coordinates
                # Get world coordinates for both to ensure consistency
                try:
                    _, y_e_world = traci.vehicle.getPosition(ego_id)
                    Dy = y - y_e_world  # Use world coordinates for both
                except traci.TraCIException:
                    Dy = y - ego_Py  # Final fallback (ego_Py might be world y)
        else:
            Dy = y - ego_Py  # Use world coordinates if ego_id not provided

        obs[start_idx:start_idx + 4] = [Dy, v, a, Px]
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
        pos_e = traci.vehicle.getLanePosition(ego_id)  # ego's position along the lane
    except traci.TraCIException:
        # Fallback to world coordinates if lane position fails
        _, y_e = traci.vehicle.getPosition(ego_id)
        pos_e = y_e
        use_lane_position = False  # Must use world coordinates for all vehicles

    leader_id = None  # leader_id is the id of the nearest vehicle in front of ego
    follower_id = None  # follower_id is the id of the nearest vehicle behind ego
    min_front_dy = float("inf")  # min_front_dy is the minimum distance between ego and the nearest vehicle in front of ego
    max_back_dy = -float("inf")  # max_back_dy is the maximum distance between ego and the nearest vehicle behind ego

    for vid in traci.lane.getLastStepVehicleIDs(lane_id):  # get the ids of all vehicles in the lane
        if vid == ego_id:
            continue
        
        # CRITICAL FIX: Use same coordinate system for both ego and vehicle
        # If we're using lane position, try lane position for vehicle; if it fails, fall back to world
        # If we're using world coordinates (because ego's lane position failed), use world for vehicle too
        if use_lane_position:
            try:
                pos = traci.vehicle.getLanePosition(vid)  # get the position of the vehicle along the lane
                dy = pos - pos_e  # calculate the distance between ego and the vehicle along the lane
            except traci.TraCIException:
                # Vehicle's lane position failed, fall back to world coordinates
                # But pos_e is lane position, so we need to get ego's world position too
                try:
                    _, y_e_world = traci.vehicle.getPosition(ego_id)
                    _, y = traci.vehicle.getPosition(vid)
                    dy = y - y_e_world  # Use world coordinates for both
                except traci.TraCIException:
                    continue  # Skip this vehicle if we can't get positions
        else:
            # Using world coordinates (ego's lane position failed)
            try:
                _, y = traci.vehicle.getPosition(vid)
                dy = y - pos_e  # pos_e is already world y-coordinate
            except traci.TraCIException:
                continue  # Skip this vehicle if we can't get position
        
        if dy > 0.0 and dy < min_front_dy:
            min_front_dy = dy  # update the minimum distance between ego and the nearest vehicle in front of ego
            leader_id = vid  # update the leader_id
        if dy < 0.0 and dy > max_back_dy:
            max_back_dy = dy  # update the maximum distance between ego and the nearest vehicle behind ego
            follower_id = vid  # update the follower_id

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
    ego:  [Py, Vy, Ay, Px, Vx]
    Ci:   [Δy_i, Vy_i, Ay_i, Px_i]
    
    Returns zeros if ego_id is not in the simulation.
    """
    # Check if ego vehicle exists
    if ego_id not in traci.vehicle.getIDList():
        return np.zeros(21, dtype=np.float32)
    
    obs = np.zeros(21, dtype=np.float32)

    # --------- EGO FEATURES (5) ---------
    x_e, y_e = traci.vehicle.getPosition(ego_id)     # Py_e = y_e
    Vy_e = traci.vehicle.getSpeed(ego_id)            # longitudinal speed
    Ay_e = traci.vehicle.getAcceleration(ego_id)     # longitudinal accel

    try:
        Px_e = traci.vehicle.getLateralLanePosition(ego_id)  # lateral position
    except traci.TraCIException:
        Px_e = x_e  # fallback

    try:
        Vx_e = traci.vehicle.getLateralSpeed(ego_id)         # lateral speed
    except (AttributeError, traci.TraCIException):
        Vx_e = 0.0

    obs[IDX_PY_EGO] = y_e
    obs[IDX_VY_EGO] = Vy_e
    obs[IDX_AY_EGO] = Ay_e
    obs[IDX_PX_EGO] = Px_e
    obs[IDX_VX_EGO] = Vx_e

    # --------- SURROUNDING VEHICLES (4 x 4) ---------
    curr_lane_id = traci.vehicle.getLaneID(ego_id)
    curr_edge_id = traci.vehicle.getRoadID(ego_id)
    target_lane_index = _get_target_lane_index(ego_id)

    # try to construct the target lane ID on the same edge
    target_lane_id = None
    try:
        target_lane_id = f"{curr_edge_id}_{target_lane_index}"
        # make sure this lane actually exists
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

    # Fill each 4-d block [Δy, Vy, Ay, Px]
    # Pass ego_id so _fill_neighbor_block can use lane position for Dy calculation
    _fill_neighbor_block(obs, START_C0, y_e, c0, ego_id)
    _fill_neighbor_block(obs, START_C1, y_e, c1, ego_id)
    _fill_neighbor_block(obs, START_C2, y_e, c2, ego_id)
    _fill_neighbor_block(obs, START_C3, y_e, c3, ego_id)

    return obs
