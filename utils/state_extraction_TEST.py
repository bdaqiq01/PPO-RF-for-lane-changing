# utils/state_extraction.py

# Step 1: Add modules to provide access to specific libraries and functions
import os # Module provides functions to handle file paths, directories, environment variables
import sys # Module provides access to Python-specific system parameters and functions
import numpy as np
import traci # Step 3: Add Traci module to provide access to specific libraries and functions
import time
# Step 2: Establish path to SUMO (SUMO_HOME)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

TOOLS = os.path.join(os.environ["SUMO_HOME"], "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)



#below this needs to be updated to extract state
#indexes for state vector
# Step 6: Define Variables
IDX_PY_EGO = 0
IDX_VY_EGO = 1
IDX_AY_EGO = 2
IDX_PX_EGO = 3
IDX_VX_EGO = 4

# Surrounding vehicles start at index 5
START_C0 = 5     # C0: current-lane leader
START_C1 = 9     # C1: target-lane leader
START_C2 = 13    # C2: current-lane follower
START_C3 = 17    # C3: target-lane follower


# Step 7: Define Functions
def _fill_vehicle_block(obs: np.ndarray, start_idx: int, ego_pos, vid: str | None):
    """
    Fills obs[start_idx : start_idx+4] with:
       [Δy,  v_y,  a_y,  x_position]
    If vehicle doesn't exist, fill with large gap + zeros.
    """
    if vid is None:
        obs[start_idx:start_idx+4] = [1000.0, 0.0, 0.0, 0.0]
        return

    try:
        # Longitudinal
        x, y = traci.vehicle.getPosition(vid)
        v = traci.vehicle.getSpeed(vid)
        a = traci.vehicle.getAcceleration(vid)

        # ego_pos = (x_ego, y_ego)
        Dy = y - ego_pos[1]

        obs[start_idx:start_idx+4] = [Dy, v, a, x]
    except traci.TraCIException:
        obs[start_idx:start_idx+4] = [1000.0, 0.0, 0.0, 0.0]


def get_state(ego_id: str) -> np.ndarray:
    """
    Returns a 21-dim np.array representing:
    [ego(5), C0(4), C1(4), C2(4), C3(4)]
    """

    obs = np.zeros(21, dtype=np.float32)

    # ---------- EGO ----------
    x_e, y_e = traci.vehicle.getPosition(ego_id)
    v_e = traci.vehicle.getSpeed(ego_id)
    a_e = traci.vehicle.getAcceleration(ego_id)
    lane = traci.vehicle.getLaneIndex(ego_id)

    # lateral speed needs to be approximated (SUMO gives only longitudinal v)
    # simple estimate: difference in x positions over lane width change
    vx_e = traci.vehicle.getLateralSpeed(ego_id) if hasattr(traci.vehicle, "getLateralSpeed") else 0.0

    obs[IDX_PY_EGO] = y_e
    obs[IDX_VY_EGO] = v_e
    obs[IDX_AY_EGO] = a_e
    obs[IDX_PX_EGO] = x_e
    obs[IDX_VX_EGO] = vx_e

    ego_pos = (x_e, y_e)

    # ---------- SURROUNDING VEHICLES ----------
    # SUMO gives leaders and followers in the SAME lane
    try:
        c0 = traci.vehicle.getLeader(ego_id, dist=500)[0] if traci.vehicle.getLeader(ego_id) else None
    except:
        c0 = None

    # To get C2 (follower)
    try:
        follower_info = traci.vehicle.getFollower(ego_id)
        c2 = follower_info[0] if follower_info else None
    except:
        c2 = None

    # To get target lane’s vehicles:
    lane_target = lane + 1  # example: lane to left (adjust for your network)

    c1 = None
    c3 = None

    if lane_target >= 0:
        try:
            c1 = traci.vehicle.getLeader(ego_id, dist=500, laneIndex=lane_target)[0]
        except:
            pass
        try:
            follower_info = traci.vehicle.getFollower(ego_id, laneIndex=lane_target)
            c3 = follower_info[0] if follower_info else None
        except:
            pass

    # Fill blocks
    _fill_vehicle_block(obs, START_C0, ego_pos, c0)
    _fill_vehicle_block(obs, START_C1, ego_pos, c1)
    _fill_vehicle_block(obs, START_C2, ego_pos, c2)
    _fill_vehicle_block(obs, START_C3, ego_pos, c3)

    return obs

# Step 8: Take simulation steps until there are no more vehicles in the network
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep() # Move simulation forward 1 step
    # Here you can decide what to do with simulation data at each step
    process_vehicles()
    
# Step 9: Close connection between SUMO and Traci
traci.close()
