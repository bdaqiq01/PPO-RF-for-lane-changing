# utils/action_decoder.py
def decode_action(action: int) -> tuple[int, int]:
    """
    Decode a discrete action in [0, 5] into:
      - lon_cmd in {0, 1}
      - lat_cmd in {0, 1, 2}

    Canonical mapping:
      action = lon_cmd * 3 + lat_cmd
    """
    a = int(action)
    if a < 0 or a > 5:
        raise ValueError(f"Action {a} out of bounds for Discrete(6)")
    lon_cmd = a // 3
    lat_cmd = a % 3
    return lon_cmd, lat_cmd