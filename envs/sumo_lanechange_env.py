# envs/sumo_lanechange_env.py

import gymnasium as gym                        # Gym API for RL environments
import numpy as np                # numerical arrays
import traci                     # SUMO's TraCI Python client
from gymnasium import spaces            # space definitions for obs/actions
import random

# your modules (paths unchanged)
from controllers.longitudinal_idm import IDMController   # low-level longitudinal controller (IDM)
from controllers.lateral_controller import LateralController  # low-level lateral controller
from utils.state_extraction import get_state             # builds the 21-d observation



class SumoLaneChangeEnv(gym.Env):
    """
    A Gym environment that wraps a SUMO simulation for lane-change RL.
    - Uses your SUMO project files: SUMO_sim/base_compl/base.sumocfg
    - Chooses an ego vehicle from a given flow id (e.g., 'f_0') once the sim starts.
    """

    def __init__(self,
                 sumo_cfg_path,      # sumo config file path
                 step_length, # step length in seconds for SUMO and environment step 
                 max_steps,  #max steps per episode
                 ego_flow_id, #flow id to choose ego from 
                 control_zone_edge, #edge where the ego vehicle is controlled
                 debug_mode: bool = False, 
                 start_lane: int = 1,  #ego car lane in the control zone
                 target_lane: int = 0,  #target lane in the control zone (offramp lane)
                 idm_params = None,
                 lateral_params = None, 
                 exit_edge_id = None #the edge the ego vehicle is routed in the after commiting to at the GATE):
    ):
        
        super().__init__()       
        self.sumo_cfg_path = sumo_cfg_path     # path to .sumocfg   (loads base.net.xml, base.rou.xml, etc.)
        self.step_length = step_length         # SUMO simulation step in seconds
        self.dt = float(step_length)          # time delta for the reward function and jerk calc
        self._max_steps = max_steps  # maximum steps per episode

        self.ego_flow_id = ego_flow_id                
        self.ego_id = None                   # will hold the chosen SUMO vehicle in the select_ego_from_flow() helper during reset()
        
        self.start_lane = start_lane # the state lane of the ego vehicle in the control zone
        self.control_zone_edge_ID = control_zone_edge # the edge where the ego vehicle is controlled
        
        # --- Gym spaces  ---
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)  # 21 continuous features
        self.action_space = spaces.Discrete(6)                                      # 6 discrete actions

        self._steps = 0

    

        self.debug_mode = debug_mode

    def reset(self, seed=None, options=None):
        #1. gym seed handling
        """Start (or restart) SUMO, choose an ego from the desired flow, and return the initial observation for a new episode."""
        super().reset(seed=seed)                       # inform Gym we've reset
        if seed is None:
            sumo_seed = int(self.np_random.integers(0, 2**31 - 1))
        else:
            sumo_seed = int(seed)

        #2.  if an old TraCI session is still open, close it
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass
        #3. reset the episode-level variables
        self._steps = 0

        warmup_steps = 20 
        #4. start a new TraCI session
        sumo_binary = "sumo"                           # change to "sumo-gui" while debugging if you want
        try:   
            traci.start([
                sumo_binary,
                "-c", self.sumo_cfg_path,
                "--step-length", str(self.step_length),
                "--seed", str(sumo_seed),
                "--no-step-log", "true",
                "--time-to-teleport", "-1",  # Disable teleporting (vehicles removed on collision)
                "--collision.action", "remove",  # Remove vehicles on collision instead of teleporting
            ], numRetries=3)  # Limit retries to avoid infinite loop
        except Exception as e:
            raise RuntimeError(
                f"Failed to start SUMO with config: {self.sumo_cfg_path}\n"
                f"Error: {e}\n"
                f"Please check:\n"
                f"  1. SUMO is installed and in PATH\n"
                f"  2. Config file exists and is valid\n"
                f"  3. All referenced files (net.xml, rou.xml) exist\n"
                f"  4. No other SUMO instance is running on the same port"
            ) from e
        
        for _ in range(warmup_steps):
            traci.simulationStep()
        
        #5. choose an ego from the desired flow
        self._choose_ego_from_flow(
            self.ego_flow_id,
            warmup_steps=0,  # Prevent double warmup - already did warmup above
            spawn_edge_id=self.control_zone_edge_ID, 
            spawn_lane_idx=self.start_lane
        )

        
        info = {
            "ego_id": self.ego_id,
            "step": self._steps
        }
        
        obs = self._get_state().astype(np.float32)
        if self.debug_mode:
            print(f"[RESET] Chosen ego_id={self.ego_id} from flow='{self.ego_flow_id}'")
            print(f"[RESET] Observation shape: {obs.shape}")
            print(f"[RESET] Observation: {obs}")
        assert obs.shape == self.observation_space.shape

        return obs, info

    def step(self, action: int):
        """
        Apply an RL action, advance SUMO one step, and return:
        (next_obs, reward, terminated, truncated, info)
        """

        self._steps += 1 

        # --- check if ego exists ---
        ego_exists = self.ego_id in traci.vehicle.getIDList()
        if not ego_exists:
            # Ego vanished before we even act -> terminate
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"ego_id": self.ego_id, "step": self._steps, "reason": "ego_missing_pre"}
            return obs, 0.0, True, False, info #current observation, reward, terminated, truncated, info
        
        
        
        traci.simulationStep()
        obs = self._get_state().astype(np.float32)

        info = {
            "ego_id": self.ego_id,
            "step": self._steps
        }   

        terminated = self._steps >= self._max_steps
        truncated = False
        return obs, 0.0, terminated, truncated, info #current observation, reward, terminated, truncated, info


    def close(self):
        """Close the TraCI connection cleanly and call the parent close()."""
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass

        # Call Gym's built-in close (important for some wrappers)
        try:
            super().close()
        except Exception:
            pass


    # ------------------------- helpers -------------------------

    def _get_state(self):
        return get_state(self.ego_id)

    def _choose_ego_from_flow(
        self,
        flow_prefix: str,
        warmup_steps: int = 20,
        timeout_steps: int = 2000,
        spawn_edge_id: str | None = None,
        spawn_lane_idx: int | None = None,
    ):
        # Warmup so vehicles spawn
        for _ in range(warmup_steps):
            traci.simulationStep()

        for _ in range(timeout_steps):
            traci.simulationStep()

            # Vehicles that started teleporting this step (good to avoid selecting)
            try:
                teleporting = set(traci.simulation.getStartingTeleportIDList()) | set(traci.simulation.getTeleportingVehiclesIDList())
            except Exception:
                teleporting = set()

            # Candidates from the requested flow
            candidates = [
                vid for vid in traci.vehicle.getIDList()
                if vid.startswith(flow_prefix + ".") and vid not in teleporting
            ]

            # Optional filters: edge + lane at selection time
            if spawn_edge_id is not None:
                candidates = [
                    vid for vid in candidates
                    if traci.vehicle.getRoadID(vid) == spawn_edge_id
                ]
            if spawn_lane_idx is not None:
                candidates = [
                    vid for vid in candidates
                    if traci.vehicle.getLaneIndex(vid) == spawn_lane_idx
                ]

            if candidates:
                # Deterministic ego selection to reduce eval variance:
                # always pick the first candidate in sorted order.
                candidates = sorted(candidates)
                self.ego_id = candidates[0]
                traci.vehicle.setSpeedMode(self.ego_id, 0)
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)
                return

        raise RuntimeError(
            f"Could not find a vehicle from flow '{flow_prefix}' "
            f"on edge={spawn_edge_id} lane={spawn_lane_idx} within {timeout_steps} steps."
        )



