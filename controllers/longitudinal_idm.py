# controllers/longitudinal_idm.py

import numpy as np

from utils.state_extraction import (
    IDX_VY_EGO,
    START_C0,
    START_C1,
)


class IDMController:
    """
    Low-level longitudinal controller using the Intelligent Driver Model (IDM).

    It takes the 21-d observation and a high-level longitudinal command
    (which leader to follow) and outputs a speed command v_cmd for the
    next SUMO step.
    """

    def __init__(self,
                 v0: float = 30.0,   # desired free-flow speed [m/s] (~108 km/h)
                 T: float = 1.5,     # desired time headway [s]
                 a_max: float = 2.5, # max acceleration [m/s^2]
                 b_comf: float = 4.5,# comfortable deceleration (positive) [m/s^2]
                 s0: float = 2.0,    # minimum gap [m]
                 dt: float = 0.5):   # simulation step length [s] (will be set from env)
        self.v0 = v0
        self.T = T
        self.a_max = a_max
        self.b_comf = b_comf
        self.s0 = s0
        self.dt = dt

    def compute(self, obs: np.ndarray, lon_cmd: int) -> float:
        """
        Compute a target speed v_cmd for the ego vehicle.

        obs:     21-d state vector
        lon_cmd: 0 = follow current-lane leader (C0)
                 1 = follow target-lane leader (C1)
        """
        v_ego = float(obs[IDX_VY_EGO])

        # --- pick which leader block to use ---
        if lon_cmd == 0:
            base = START_C0  # current-lane leader
        else:
            base = START_C1  # target-lane leader

        Dy, v_lead, _, _ = obs[base:base + 4]

        # --- IDM acceleration ---
        # If no leader exists, we filled Dy ~ 1000, v_lead = 0 in get_state()
        if Dy > 900.0:
            # free road: accelerate towards v0
            acc = self.a_max * (1.0 - (v_ego / max(self.v0, 1e-3)) ** 4)
        else:
            s = max(float(Dy), 0.1)                # actual gap [m]
            delta_v = v_ego - float(v_lead)        # v_ego - v_lead
            s_star = (self.s0
                      + v_ego * self.T
                      + v_ego * delta_v / (2.0 * np.sqrt(self.a_max * self.b_comf)))

            acc = self.a_max * (
                1.0
                - (v_ego / max(self.v0, 1e-3)) ** 4
                - (s_star / s) ** 2
            )

        # Clip acceleration to physical limits from the paper
        acc = float(np.clip(acc, -self.b_comf, self.a_max))

        # Simple Euler step to get next speed command
        v_next = max(0.0, v_ego + acc * self.dt)

        return float(v_next)
