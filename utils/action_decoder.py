# utils/action_decoder.py
def decode_action(action: int) -> tuple[int, int]:
    """
    Decode a discrete action in [0, 5] into:
      - lon_cmd in {0, 1}  # 0: follow current-lane leader, 1: follow target-lane leader
      - lat_cmd in {0, 1, 2}  # 0: lane keeping, 1: change to target lane (right lane), 2: abort lane change

    Canonical mapping:
      action = lon_cmd ( 0 pace with current-lane leader, 1 pace with target-lane leader) * 3, lat_cmd (lane munever intent)
      0 = (0,0) follow current-lane leader, keep lane  
      1 = (0,1) follow current-lane leader, lane change
      2 = (0,2) follow current-lane leader, abort
      
      3 = (1,0) follow target-lane leader, keep lane  
      4 = (1,1) follow target-lane leader, lane change, 
      5 = (1,2) follow target-lane leader, abort, 
    """
    a = int(action)
    if a < 0 or a > 5:
        raise ValueError(f"Action {a} out of bounds for Discrete(6)")
    lon_cmd = a // 3
    lat_cmd = a % 3
    return lon_cmd, lat_cmd