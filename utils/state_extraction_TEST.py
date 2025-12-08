import os
import sys
import numpy as np
import traci

# ----------------- SUMO SETUP -----------------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

SUMO_CONFIG = [
    "sumo-gui",
    "-c", "SUMO_sim/base2_compl/2lane_oneOnOff.sumocfg",
    "--step-length", "0.1",
    "--delay", "1000",
    "--lateral-resolution", "0.1",
]

# ----------------- STATE INDEXING -----------------
# ego: (Py, Vy, Ay, Px, Vx)
IDX_PY_EGO = 0  #longitudinal position
IDX_VY_EGO = 1 #longitudinal speed
IDX_AY_EGO = 2 #longitudinal acceleration
IDX_PX_EGO = 3 #lateral position
IDX_VX_EGO = 4

# C0..C3 blocks of 4 for each neighbor ( relative distance, speed, acceleration, lateral position)
START_C0 = 5   # current-lane leader
START_C1 = 9   # target-lane leader
START_C2 = 13  # current-lane follower
START_C3 = 17  # target-lane follower


# ----------------- HELPERS -----------------
def _fill_neighbor_block(obs: np.ndarray,
                         start_idx: int,
                         ego_Py: float,
                         veh_id: str | None) -> None:
    """
    for each neighbor, fill obs[start_idx : start_idx+4] with:
        [Δy, Vy, Ay, Px]
    If vehicle/nigbor doesn't exist, use a large positive gap and zeros:
        [1000.0, 0.0, 0.0, 0.0]
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

        Dy = y - ego_Py  # relative longitudinal distance to ego

        obs[start_idx:start_idx + 4] = [Dy, v, a, Px]
    except traci.TraCIException:
        obs[start_idx:start_idx + 4] = [1000.0, 0.0, 0.0, 0.0]


def _find_leader_follower_in_lane(ego_id: str, lane_id: str):
    """
    Find nearest leader and follower of ego in a specific lane using
    world-longitudinal coordinate (y). Returns (leader_id, follower_id),
    each possibly None.
    """
    x_e, y_e = traci.vehicle.getPosition(ego_id) # ego's position in world coordinates

    leader_id = None # leader_id is the id of the nearest vehicle in front of ego
    follower_id = None # follower_id is the id of the nearest vehicle behind ego
    min_front_dy = float("inf") # min_front_dy is the minimum distance between ego and the nearest vehicle in front of ego
    max_back_dy = -float("inf") # max_back_dy is the maximum distance between ego and the nearest vehicle behind ego

    for vid in traci.lane.getLastStepVehicleIDs(lane_id): # get the ids of all vehicles in the lane
        if vid == ego_id:
            continue
        _, y = traci.vehicle.getPosition(vid) # get the position of the vehicle in world coordinates
        dy = y - y_e # calculate the distance between ego and the vehicle
        if dy > 0.0 and dy < min_front_dy:
            min_front_dy = dy # update the minimum distance between ego and the nearest vehicle in front of ego
            leader_id = vid # update the leader_id
        if dy < 0.0 and dy > max_back_dy:
            max_back_dy = dy # update the maximum distance between ego and the nearest vehicle behind ego
            follower_id = vid # update the follower_id      

    return leader_id, follower_id


def _get_target_lane_index(ego_id: str) -> int: # get the index of the right lane
    """
    Mandatory route: main -> off-ramp.
    In the paper’s network the off-ramp is on the right, so the target
    lane is always the lane immediately to the RIGHT of the ego
    (if such a lane exists). :contentReference[oaicite:1]{index=1}
    """
    curr_idx = traci.vehicle.getLaneIndex(ego_id)
    # SUMO uses 0 = right-most lane, larger index = more to the left.
    if curr_idx > 0:
        return curr_idx - 1
    else:
        return curr_idx  # already in right-most / target lane


def select_ego_from_flow(flow_id_prefix: str = "f_2") -> str | None:
    """
    Choose one ego vehicle from the mandatory-lane-change flow f_2.
    We check both routeID == 'f_2' and ID starting with 'f_2'.
    """
    for vid in traci.vehicle.getIDList():
        try:
            if traci.vehicle.getRouteID(vid) == flow_id_prefix:
                return vid
        except traci.TraCIException:
            pass

        if vid.startswith(flow_id_prefix):
            return vid

    return None


# ----------------- MAIN STATE EXTRACTION -----------------
def get_state_21d(ego_id: str) -> np.ndarray:
    """
    Return the 21-dim state vector used in Ye et al.:
        [ego(5), C0(4), C1(4), C2(4), C3(4)]
    ego:  [Py, Vy, Ay, Px, Vx]
    Ci:   [Δy_i, Vy_i, Ay_i, Px_i]
    """
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
    _fill_neighbor_block(obs, START_C0, y_e, c0)
    _fill_neighbor_block(obs, START_C1, y_e, c1)
    _fill_neighbor_block(obs, START_C2, y_e, c2)
    _fill_neighbor_block(obs, START_C3, y_e, c3)

    return obs


# ----------------- EXAMPLE APPLICATION LOOP -----------------
if __name__ == "__main__":
    traci.start(SUMO_CONFIG)

    ego_id = None

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # lazily pick ego when an f_2 vehicle appears
            if ego_id is None or ego_id not in traci.vehicle.getIDList():
                ego_id = select_ego_from_flow("f_2")

            if ego_id is None:
                continue  # no ego yet in the network

            state = get_state_21d(ego_id)

            # For verification / debugging before wiring into Gym
            sim_time = traci.simulation.getTime()
            print(f"t={sim_time:.2f}  ego={ego_id}  state={state}")

    finally:
        traci.close()
