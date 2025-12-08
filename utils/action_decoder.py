# utils/action_decoder.py
def decode_action(a: int):
    # 6 discrete actions -> (lat_cmd, lon_cmd)
    mapping = {
        0: (0, 0),  # keep lane, follow current-lane leader
        1: (0, 1),  # keep lane, follow target-lane leader
        2: (1, 0),  # change lane, follow current-lane leader
        3: (1, 1),  # change lane, follow target-lane leader
        4: (2, 0),  # abort, follow current-lane leader
        5: (2, 1),  # abort, follow target-lane leader
    }
    return mapping[a]
# envs/sumo_lanechange_env.py