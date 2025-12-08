# utils/state_extraction.py
import numpy as np
import traci

def get_state(ego_id: str) -> np.ndarray:
    """
    Extract a 21-dimensional observation vector for the RL agent.

    Structure example (you can adjust it to match your paper):
      0-2 : ego speed, accel, lane index
      3-5 : rel. dist, rel. speed to lead vehicle in same lane
      6-8 : rel. dist, rel. speed to following vehicle in same lane
      9-20: info about two neighbors in adjacent lanes (4 vehicles)
    """
    if ego_id not in traci.vehicle.getIDList():
        # if ego disappeared (arrived or teleported), return zeros
        return np.zeros(21, dtype=np.float32)

    # 1️⃣ Ego state
    ego_speed = traci.vehicle.getSpeed(ego_id)
    ego_accel = traci.vehicle.getAcceleration(ego_id)
    ego_lane = traci.vehicle.getLaneIndex(ego_id)
    ego_pos = traci.vehicle.getPosition(ego_id)[0]  # x-coordinate along the road

    # 2️⃣ Neighboring vehicles (simplified example)
    # Find the leader and follower in the same lane
    try:
        leader_id, dist_to_leader = traci.vehicle.getLeader(ego_id)
        leader_speed = traci.vehicle.getSpeed(leader_id)
    except TypeError:
        leader_id, dist_to_leader, leader_speed = None, 0.0, 0.0

    try:
        follower_id, dist_to_follower = traci.vehicle.getFollower(ego_id)
        follower_speed = traci.vehicle.getSpeed(follower_id)
    except TypeError:
        follower_id, dist_to_follower, follower_speed = None, 0.0, 0.0

    # Relative features
    rel_speed_lead = leader_speed - ego_speed
    rel_speed_follow = follower_speed - ego_speed

    # 3️⃣ Simple example of neighbors from adjacent lanes
    # (You can expand this with traci.vehicle.getLeftLeaders() etc.)
    neighbor_feats = np.zeros(12)  # placeholder for 4 neighbors * 3 features each

    # 4️⃣ Build final 21-dim vector
    state = np.array([
        ego_speed, ego_accel, ego_lane,
        dist_to_leader, rel_speed_lead,
        dist_to_follower, rel_speed_follow,
        *neighbor_feats
    ], dtype=np.float32)

    # pad or trim to exactly 21
    if state.shape[0] < 21:
        state = np.pad(state, (0, 21 - state.shape[0]))
    elif state.shape[0] > 21:
        state = state[:21]

    return state
