# controllers/lateral_controller.py

import traci


class LateralController:
    """
    Low-level lateral controller.

    High-level lateral commands (target-lane-relative, NOT direction-hardcoded):
      lat_cmd = 0: lane keeping (stay in current lane)
      lat_cmd = 1: change to the configured target lane (left or right,
                   determined by the caller-supplied target_lane_index)
      lat_cmd = 2: abort lane change (stay in current lane)
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

    def execute(self, ego_id: str, lat_cmd: int, target_lane_index: int) -> None:
        """Execute the requested lateral action for ego_id.

        target_lane_index is the absolute SUMO lane index the ego should head
        toward when lat_cmd == 1. The controller clamps it to the valid lane
        range of the current edge.
        """
        # If ego is gone, do nothing
        if ego_id not in traci.vehicle.getIDList():
            return

        try:
            curr_idx = traci.vehicle.getLaneIndex(ego_id)
            edge_id = traci.vehicle.getRoadID(ego_id)
            n_lanes = traci.edge.getLaneNumber(edge_id)
        except traci.TraCIException:
            return

        # Clamp the requested target into [0, n_lanes - 1] so an externally
        # supplied target lane that doesn't exist on the current edge degrades
        # safely (treated as "stay in nearest valid lane").
        target_idx = max(0, min(int(target_lane_index), n_lanes - 1))

        # 0: lane keeping
        if lat_cmd == 0:
            traci.vehicle.changeLane(ego_id, curr_idx, self.duration)

        # 1: change to the configured target lane (works for both directions)
        elif lat_cmd == 1:
            if target_idx != curr_idx:
                traci.vehicle.changeLane(ego_id, target_idx, self.duration)
            # else already in target lane -> effectively lane keeping

        # 2: abort lane change (keep current lane)
        elif lat_cmd == 2:
            traci.vehicle.changeLane(ego_id, curr_idx, self.duration)

        # other lat_cmd values are silently ignored
