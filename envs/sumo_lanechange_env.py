# envs/sumo_lanechange_env.py

import gymnasium as gym                        # Gym API for RL environments
import numpy as np                # numerical arrays
import traci                     # SUMO's TraCI Python client
from gymnasium import spaces            # space definitions for obs/actions
import random

# your modules (paths unchanged)
from controllers.longitudinal_idm import IDMController   # low-level longitudinal controller (IDM)
from controllers.lateral_controller import LateralController  # low-level lateral controller
from utils.reward_functions import compute_reward        # step reward function
from utils.state_extraction import get_state             # builds the 21-d observation
from utils.action_decoder import decode_action           # maps discrete action -> (lat_cmd, lon_cmd)

from utils.state_extraction import (
    START_C1,
    START_C3,
)


#gym has 4 core methods: __init__ (define the observation space action space any simuation setting or constance),
# reset - called at the begining of an episode should restart, 
# step - applies the agent chosen action to the world, advances the simulation by one step, and returns the new observation, calculates the reward, done
#close cleans up the environment when done



class SumoLaneChangeEnv(gym.Env):
    """
    A Gym environment that wraps a SUMO simulation for lane-change RL.
    - Uses your SUMO project files: SUMO_sim/base_compl/base.sumocfg
    - Chooses an ego vehicle from a given flow id (e.g., 'f_0') once the sim starts.
    """

    def __init__(self,
                 sumo_cfg_path,      # allow caller to override, but default to your file tree
                 step_length, # seconds
                 max_steps,  #max steps per episode
                 ego_flow_id, #flow id to choose ego from 
                 gate_pos: float,  #distance in metter from the start the region we commit to route decision
                 control_zone_edge,
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
        
        self.gate_pos = gate_pos #the distance in meters from the start the region we commit to route decision
        self.start_lane = start_lane # the state lane of the ego vehicle in the control zone
        self.target_lane = target_lane # the target lane of the ego vehicle in the control zone
        self.control_zone_edge_ID = control_zone_edge # the edge where the ego vehicle is controlled
        self.exit_edge_id = exit_edge_id #the edge the ego vehicle is routed in the after commiting to at the GATE
        # --- Gym spaces  ---
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)  # 21 continuous features
        self.action_space = spaces.Discrete(6)                                      # 6 discrete actions
        # --- low-level controllers ---
        #I am not sure how these work yet
        if idm_params is None:
            idm_params = {}
        if lateral_params is None:
            lateral_params = {}
        self.longi_ctrl = IDMController(dt= self.dt, **idm_params)          # translates longitudinal intent -> speed command
        self.lat_ctrl = LateralController(**lateral_params)        # translates lateral intent -> lane-change TraCI calls
    
        self._steps = 0

        self._committed = False
        self._took_exit = False
        self._ppo_lane_change = False
        self._pending_lc = False
        self._pending_lc_steps = 0

    def reset(self, seed=None, options=None):
        """Start (or restart) SUMO, choose an ego from the desired flow, and return the initial observation."""
        super().reset(seed=seed)                       # inform Gym we've reset
        if seed is None:
            sumo_seed = int(self.np_random.integers(0, 2**31 - 1))
        else:
            sumo_seed = int(seed)

        # if an old TraCI session is still open, close it
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass
        
        self._steps = 0
        self._committed = False
        self._took_exit = False
        self._ppo_lane_change = False
        self._pending_lc = False
        self._pending_lc_steps = 0

        #start a new TraCI session
        # choose binary: "sumo" for headless (fast), "sumo-gui" to watch
        sumo_binary = "sumo"                           # change to "sumo-gui" while debugging if you want
        traci.start([                                   # launch SUMO with your config and step length
            sumo_binary,
            "-c", self.sumo_cfg_path,
            "--step-length", str(self.step_length) ,
            "--seed", str(sumo_seed) #,
        ])

        # pick the ego vehicle from the specified flow
        #option 1- of picking vehicle from a flow
        self._choose_ego_from_flow(self.ego_flow_id, spawn_edge_id= self.control_zone_edge_ID, spawn_lane_idx= self.start_lane)  #need returns self.ego_id self.ego_id is defined in _choose_ego_from_flow()    
        try:
            self._initial_lane_idx = traci.vehicle.getLaneIndex(self.ego_id)
            self._initial_edge_id = traci.vehicle.getRoadID(self.ego_id)
        except Exception:
            self._initial_lane_idx = None
            self._initial_edge_id = None
        
        obs = self._get_state() #this has to be produce the 21 dims that goes to PPO agent 

        info = {   
            "ego_id": self.ego_id,
            "step_length": float(self.step_length),
            "seed": seed,
            "sumo_seed": sumo_seed,
            "lane_index": self._initial_lane_idx,
            "episode_start_time": traci.simulation.getTime()
        }

        return obs, info

    def step(self, action: int):
        """
        Apply an RL action, advance SUMO one step, and return:
        (next_obs, reward, terminated, truncated, info)
        """

        # --- (A) per-episode flags MUST exist (safe even if already set in reset) ---
        if not hasattr(self, "_committed"):
            self._committed = False
        if not hasattr(self, "_took_exit"):
            self._took_exit = False
        if not hasattr(self, "_ppo_lane_change"):
            self._ppo_lane_change = False
        if not hasattr(self, "_pending_lc"):
            self._pending_lc = False
        if not hasattr(self, "_pending_lc_steps"):
            self._pending_lc_steps = 0

        self._steps += 1

        # --- (B) snapshot BEFORE action (guarded) ---
        ego_exists = self.ego_id in traci.vehicle.getIDList()
        if not ego_exists:
            # Ego vanished before we even act -> terminate
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"ego_id": self.ego_id, "step": self._steps, "collision": True, "success": False, "reason": "ego_missing_pre"}
            return obs, 0.0, True, False, info
        
        obs_t = self._get_state().astype(np.float32)
        prev_edge = prev_lane_idx = prev_pos = None
       
        try:
            prev_edge = traci.vehicle.getRoadID(self.ego_id)
            prev_lane_idx = traci.vehicle.getLaneIndex(self.ego_id)
        except Exception:
            pass

        # Decode discrete action -> (lat_cmd, lon_cmd)
        lat_cmd, lon_cmd = decode_action(int(action))

        # --- (C) control-zone gating + safety intervention ---
        in_control_zone = (self.control_zone_edge_ID is None)
        if self.control_zone_edge_ID is not None and prev_edge is not None:
            in_control_zone = (prev_edge == self.control_zone_edge_ID)

        rp = 0.0
        if not in_control_zone:
            lat_cmd = 0  # force keep-lane outside control zone
        else:
            lat_cmd, rp = self._apply_safety_intervention(obs_t, lat_cmd)

        # --- (D) apply low-level controls + mark pending LC when PPO asks for LC in zone ---
        # lateral controller issues TraCI lane-change command (if lat_cmd==1, etc.)
        v_cmd = self.longi_ctrl.compute(obs_t, lon_cmd)
        self.lat_ctrl.execute(self.ego_id, lat_cmd)
        traci.vehicle.setSpeed(self.ego_id, float(v_cmd))

        # PPO "attempted lane change" flag (pending until we see actual lane index change)
        if in_control_zone and lat_cmd == 1:
            self._pending_lc = True
            self._pending_lc_steps = 0

        # --- (E) advance SUMO ---
        traci.simulationStep()

        # --- (F) snapshot AFTER step (guarded) ---
        ego_exists = self.ego_id in traci.vehicle.getIDList()
        curr_edge = curr_lane_idx = curr_pos = None
        if ego_exists:
            try:
                curr_edge = traci.vehicle.getRoadID(self.ego_id)
                curr_lane_idx = traci.vehicle.getLaneIndex(self.ego_id)
                curr_pos = traci.vehicle.getLanePosition(self.ego_id)
            except Exception:
                pass

        # --- (G) PPO lane-change tracking using _pending_lc ---
        # Mark success once we observe a real lane-index transition start_lane -> target_lane
        if self._pending_lc:
            self._pending_lc_steps += 1

            # Confirm transition start_lane -> target_lane (best effort)
            if (
                prev_lane_idx is not None
                and curr_lane_idx is not None
                and prev_lane_idx == self.start_lane
                and curr_lane_idx == self.target_lane
            ):
                self._ppo_lane_change = True
                self._pending_lc = False
                self._pending_lc_steps = 0

            elif (
                (curr_lane_idx is not None)
                and (curr_lane_idx == self.target_lane)
                and (self.control_zone_edge_ID is None or curr_edge == self.control_zone_edge_ID)
            ):

                self._ppo_lane_change = True
                self._pending_lc = False
                self._pending_lc_steps = 0

            # Clear pending if it’s taking too long or ego left zone
            if self._pending_lc and self._pending_lc_steps >= 6:
                self._pending_lc = False
                self._pending_lc_steps = 0

            if (
                self._pending_lc
                and self.control_zone_edge_ID is not None
                and curr_edge is not None
                and curr_edge != self.control_zone_edge_ID
            ):
                self._pending_lc = False
                self._pending_lc_steps = 0

    

        # --- (H) commit route ONLY ONCE, and only if PPO already made it into target lane ---
        # This prevents SUMO from "choosing back" before you commit.
        if (
            (not self._committed)
            and ego_exists
            and (self.control_zone_edge_ID is None or curr_edge == self.control_zone_edge_ID)
            and (curr_pos is not None)
            and (curr_pos >= self.gate_pos)
        ):
            if self._ppo_lane_change:
                self._took_exit = bool(self._commit_route_at_gate())
            else:
                self._took_exit = False
            self._committed = True

        # --- (I) next obs + reward ---
        if ego_exists:
            try:
                next_obs = self._get_state()
            except Exception:
                next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        reward, reward_components = compute_reward(
            obs_t=obs_t,
            obs_tp1=next_obs,
            ego_id=self.ego_id,
            lat_cmd=lat_cmd,
            dt=self.dt,
            ds=getattr(self.lat_ctrl, "detect_dist", 10.0),
            rp=rp,
            v_desired=25.0,
        )

        # --- (J) termination ---
        terminated, truncated, reason, collision, success = self._check_done()

        # --- (K) end-of-episode log ---
        if terminated or truncated:
            try:
                edge_id = traci.vehicle.getRoadID(self.ego_id) if ego_exists else None
                lane_idx = traci.vehicle.getLaneIndex(self.ego_id) if ego_exists else None
                pos = traci.vehicle.getLanePosition(self.ego_id) if ego_exists else None
                pos_str = f"{pos:.2f}" if pos is not None else "None"
            except Exception:
                edge_id, lane_idx, pos_str = None, None, "None"

            print(
                f"[EP END] steps={self._steps}, reason={reason}, collision={collision}, success={success}, "
                f"edge={edge_id}, lane={lane_idx}, pos={pos_str}, "
                f"_pending_lc={bool(self._pending_lc)}, _ppo_lane_change={bool(self._ppo_lane_change)}, "
                f"_committed={bool(self._committed)}, _took_exit={bool(self._took_exit)}"
            )

        # --- (L) info dict ---
        info = {
            "ego_id": self.ego_id,
            "step": self._steps,
            "collision": bool(collision),
            "success": bool(success),
            "is_success": bool(success),
            "reason": reason,
            "rp": float(rp),
            "_pending_lc": bool(self._pending_lc),
            "_ppo_lane_change": bool(self._ppo_lane_change),
            "_committed": bool(self._committed),
            "_took_exit": bool(self._took_exit),
        }
        info.update(reward_components)

        if ego_exists:
            try:
                info["lane_index"] = traci.vehicle.getLaneIndex(self.ego_id)
                info["v_ego"] = traci.vehicle.getSpeed(self.ego_id)
                info["edge_id"] = traci.vehicle.getRoadID(self.ego_id)
                info["lane_pos"] = traci.vehicle.getLanePosition(self.ego_id)
            except Exception:
                pass

        return next_obs, float(reward), bool(terminated), bool(truncated), info


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
        """Wrapper so your existing utils/state_extraction.get_state() can start by only needing ego_id internally."""
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
                self.ego_id = random.choice(candidates)
                traci.vehicle.setSpeedMode(self.ego_id, 0)
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)
                return

        raise RuntimeError(
            f"Could not find a vehicle from flow '{flow_prefix}' "
            f"on edge={spawn_edge_id} lane={spawn_lane_idx} within {timeout_steps} steps."
        )


    def _apply_safety_intervention(self, obs: np.ndarray, lat_cmd: int):
        """
        Safety intervention module (paper-style):

        - Only intervenes when the agent wants to change lane (lat_cmd == 1).
        - Uses distances to target-lane leader (C1) and follower (C3).
        - If either is closer than detect_dist, treat as "catastrophic" and
          override with an abort (lat_cmd = 2) and return a penalty Rp.
        """
        rp = 0.0

        # If not trying to change lane, nothing to do
        if lat_cmd != 1:
            return lat_cmd, rp

        # Distances in target lane:
        Dy_c1 = float(obs[START_C1 + 0])  # Δy to target-lane leader
        Dy_c3 = float(obs[START_C3 + 0])  # Δy to target-lane follower

        # distance threshold ds (10 m by your macro)
        ds = getattr(self.lat_ctrl, "detect_dist", 10.0)

        # Remember: Δy = y_i - y_e
        unsafe_front = False
        unsafe_back = False
        
        if abs(Dy_c1) < 900.0:  # Real vehicle exists
            unsafe_front = 0.0 < Dy_c1 < ds
        
        if abs(Dy_c3) < 900.0:  # Real vehicle exists
            unsafe_back = -ds < Dy_c3 < 0.0
        
        if unsafe_front or unsafe_back:
            lat_cmd = 2
            rp = -1.0
        
        return lat_cmd, rp


    def _check_done(self):
        terminated = False
        truncated = False
        collision = False
        success = False
        reason = None

        ego_exists = self.ego_id in traci.vehicle.getIDList()

        # --- 1) Check collisions & teleports first ---
        try:
            colliding = set(traci.simulation.getCollidingVehiclesIDList())
            starting_teleports = set(traci.simulation.getStartingTeleportIDList())
            teleporting = set(traci.simulation.getTeleportingVehiclesIDList())
            all_teleporting = starting_teleports | teleporting
        except Exception:
            colliding = set()
            all_teleporting = set()

        if ego_exists and (self.ego_id in colliding or self.ego_id in all_teleporting):
            return True, False, "collision_or_teleport", True, False

        # --- 2) Ego removed from simulation (not found anymore) ---
        if not ego_exists:
            return True, False, "ego_removed", True, False

        # --- 3) Ego still exists: check for goal ---
        if self._reached_goal():
            return True, False, "goal", False, True

        # committed but didn't take exit = failure
        if self._committed and (not self._took_exit):
            return True, False, "chose_mainline", False, False

        if self._steps >= self._max_steps:
            truncated = True
            reason = "timeout"

        return terminated, truncated, reason, collision, success

    
    def _reached_goal(self):
        return bool(self._committed and self._took_exit and self._ppo_lane_change)


    def _commit_route_at_gate(self) -> bool:
        """
        Commit decision at gate:
        - "take exit" iff ego is in target_lane at/after gate on control zone edge
        - If exit_edge_id is provided and we are taking exit: set a short route to force SUMO not to flip back.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return False

        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            lane = traci.vehicle.getLaneIndex(self.ego_id)
            pos = traci.vehicle.getLanePosition(self.ego_id)
        except Exception:
            return False

        # Must be on (or effectively at) control-zone edge when committing, if one is defined
        if self.control_zone_edge_ID is not None and edge != self.control_zone_edge_ID:
            return False

        if pos is None or pos < self.gate_pos:
            return False

        took_exit = (lane == self.target_lane)

        # If we decided exit, optionally force route to the exit edge
        if took_exit and self.exit_edge_id:
            # Include current edge so SUMO accepts the change
            try:
                traci.vehicle.setRoute(self.ego_id, [edge, self.exit_edge_id])
            except Exception:
                # If setRoute fails, still return decision based on lane
                pass

        return took_exit



