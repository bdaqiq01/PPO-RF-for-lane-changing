# controllers/lateral_controller.py

import traci


class LateralController:
    """
    Low-level lateral controller.

    High-level lateral commands:
      lat_cmd = 0: lane keeping (stay in current lane)
      lat_cmd = 1: change to the target lane (right lane)
      lat_cmd = 2: abort lane change (go back left / original lane)
    """

    def __init__(self,
                 lane_change_duration: float = 3.0,
                 lane_change_detection_distance: float = 10.0):
        """
        lane_change_duration:
            How many seconds SUMO should keep the requested target lane.

        lane_change_detection_distance:
            Used by the safety intervention logic in the env to decide
            whether a lane change is "too close" to neighbors.
        """
        self.duration = lane_change_duration
        self.detect_dist = lane_change_detection_distance

    def execute(self, ego_id: str, lat_cmd: int) -> None:
        """Execute the requested lateral action for ego_id."""
        # If ego is gone, do nothing
        if ego_id not in traci.vehicle.getIDList():
            return

        try:
            curr_idx = traci.vehicle.getLaneIndex(ego_id)
            edge_id = traci.vehicle.getRoadID(ego_id)
            n_lanes = traci.edge.getLaneNumber(edge_id)
        except traci.TraCIException:
            return

        # SUMO: lane 0 = right-most; higher index = further left.
        target_idx_right = max(curr_idx - 1, 0)

        # 0: lane keeping
        if lat_cmd == 0:
            traci.vehicle.changeLane(ego_id, curr_idx, self.duration)

        # 1: change to target lane (right lane)
        elif lat_cmd == 1:
            if curr_idx > 0:
                traci.vehicle.changeLane(ego_id, target_idx_right, self.duration)
            # else already in right-most lane -> effectively lane keeping

        # 2: abort lane change (go back left/original lane)
        elif lat_cmd == 2:
            # "abort" = go one lane to the left (back toward original lane)
            if curr_idx < n_lanes - 1:
                back_idx = curr_idx + 1
                traci.vehicle.changeLane(ego_id, back_idx, self.duration)
            # if already left-most, just stay

        # other lat_cmd values are silently ignored
