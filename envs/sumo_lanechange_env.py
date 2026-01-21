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
        self._lc_reward_given = False  # Track if lane change reward was given this episode
        self._attempt_reward_given = False  # Track if attempt reward was given this episode
        self._last_pos_in_zone = 0.0  # Track position in control zone for progress reward
        self._entered_control_zone = False  # Track if agent entered control zone this episode
        self._steps_in_control_zone = 0  # Track steps in control zone without lane change (for delay penalty)

        #start a new TraCI session
        # choose binary: "sumo" for headless (fast), "sumo-gui" to watch
        sumo_binary = "sumo"                           # change to "sumo-gui" while debugging if you want
        # Launch SUMO with your config and step length
        # IMPORTANT: Disable per-step logging to avoid console spam and speed up training.
        traci.start([
            sumo_binary,
            "-c", self.sumo_cfg_path,
            "--step-length", str(self.step_length),
            "--seed", str(sumo_seed),
            "--no-step-log", "true",
        ])
        
        # CRITICAL: Advance simulation to let traffic build up before selecting ego
        # This ensures vehicles from E1 (on-ramp) arrive in lane 0 before ego needs them
        # Without this, target lane is often empty, making the task too easy
        warmup_steps = 50  # ~10 seconds at 0.2s step length - increased to allow more vehicles to arrive
        for _ in range(warmup_steps):
            traci.simulationStep()
        
        # Verify that target lane has traffic after warmup (silent check)

        # Select ego from start_lane on control_zone_edge and disable automatic lane changing
        # Set warmup_steps=0 to prevent double warmup (already warmed up above)
        self._choose_ego_from_flow(
            self.ego_flow_id,
            warmup_steps=0,  # Prevent double warmup - already did warmup above
            spawn_edge_id=self.control_zone_edge_ID, 
            spawn_lane_idx=self.start_lane
        )
        
        try:
            self._initial_lane_idx = traci.vehicle.getLaneIndex(self.ego_id)
            self._initial_edge_id = traci.vehicle.getRoadID(self.ego_id)
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)
            traci.vehicle.setSpeedMode(self.ego_id, 0)
            
            # CRITICAL: Reject ego if not in start_lane - this prevents false success
            if self._initial_lane_idx != self.start_lane:
                print(f"[ERROR] Ego {self.ego_id} in lane {self._initial_lane_idx}, expected {self.start_lane}. Reselecting...")
                # Try to find another vehicle
                self._choose_ego_from_flow(
                    self.ego_flow_id, 
                    spawn_edge_id=self.control_zone_edge_ID, 
                    spawn_lane_idx=self.start_lane
                )
                self._initial_lane_idx = traci.vehicle.getLaneIndex(self.ego_id)
                self._initial_edge_id = traci.vehicle.getRoadID(self.ego_id)
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)
                traci.vehicle.setSpeedMode(self.ego_id, 0)
                
            if self._initial_lane_idx == self.target_lane:
                print(f"[CRITICAL ERROR] Ego spawned in target lane {self.target_lane}! This will cause false success!")
                # Force failure by setting initial_lane_idx to None
                self._initial_lane_idx = None
        except Exception as e:
            print(f"[ERROR] Failed to verify ego selection: {e}")
            self._initial_lane_idx = None
            self._initial_edge_id = None
        
        # CRITICAL: Ensure target lane (lane 0) has traffic for challenging scenario
        if self.control_zone_edge_ID:
            try:
                target_lane_id = f"{self.control_zone_edge_ID}_{self.target_lane}"
                vehicles_in_target_lane = traci.lane.getLastStepVehicleIDs(target_lane_id)
                
                # Disable automatic lane changing for vehicles in target lane
                vehicles_in_target_lane = [v for v in vehicles_in_target_lane if v != self.ego_id]
                for veh_id in vehicles_in_target_lane:
                    try:
                        traci.vehicle.setLaneChangeMode(veh_id, 0)
                    except Exception:
                        pass
            except Exception:
                pass
        
        obs = self._get_state()

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

        self._steps += 1

        # --- (B) snapshot BEFORE action ---
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

        lat_cmd, lon_cmd = decode_action(int(action))

        # --- (C) control-zone gating + safety intervention ---
        in_control_zone = (self.control_zone_edge_ID is None or 
                          (prev_edge is not None and prev_edge == self.control_zone_edge_ID))

        rp = 0.0
        if not in_control_zone:
            lat_cmd = 0
        else:
            lat_cmd, rp = self._apply_safety_intervention(obs_t, lat_cmd)

        # --- (D) apply controls ---
        if ego_exists:
            try:
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)
            except Exception:
                pass
        
        # CRITICAL: Ensure vehicles in target lane (lane 0) stay there
        # Only disable automatic lane changing - don't force lane changes (causes errors)
        # Network issue: Vehicles from E0 enter E0.212 in lane 1, not lane 0
        # Only vehicles from E1 (on-ramp) enter E0.212 in lane 0
        if in_control_zone and self.control_zone_edge_ID:
            try:
                target_lane_id = f"{self.control_zone_edge_ID}_{self.target_lane}"
                vehicles_in_target_lane = traci.lane.getLastStepVehicleIDs(target_lane_id)
                vehicles_in_target_lane = [v for v in vehicles_in_target_lane if v != self.ego_id]
                
                # Disable automatic lane changing for vehicles already in target lane
                for veh_id in vehicles_in_target_lane:
                    try:
                        traci.vehicle.setLaneChangeMode(veh_id, 0)
                    except Exception:
                        pass
            except Exception:
                pass
        
        v_cmd = self.longi_ctrl.compute(obs_t, lon_cmd)
        self.lat_ctrl.execute(self.ego_id, lat_cmd)
        traci.vehicle.setSpeed(self.ego_id, float(v_cmd))

        if in_control_zone and lat_cmd == 1:
            self._pending_lc = True
            self._pending_lc_steps = 0

        traci.simulationStep()

        # --- (F) CRITICAL: Early teleport/collision detection (before any other processing) ---
        # This prevents computing rewards/states for invalid episodes
        # IMPORTANT: Only terminate on ACTUAL problems, not on exceptions during checking
        has_teleport_or_collision = False
        ego_exists = True  # Default to True, only set False if we can confirm ego is missing
        
        # Check for teleports/collisions (only if we can successfully check)
        try:
            colliding = set(traci.simulation.getCollidingVehiclesIDList())
            starting_teleports = set(traci.simulation.getStartingTeleportIDList())
            teleporting = set(traci.simulation.getTeleportingVehiclesIDList())
            all_teleporting = starting_teleports | teleporting
            has_teleport_or_collision = (self.ego_id in colliding or self.ego_id in all_teleporting)
        except Exception:
            # If we can't check, assume no teleport/collision (don't terminate on check failure)
            pass
        
        # Check if ego still exists (only terminate if we can confirm it's missing)
        try:
            ego_exists = self.ego_id in traci.vehicle.getIDList()
        except Exception:
            # If we can't check, assume ego exists (don't terminate on check failure)
            ego_exists = True
        
        # CRITICAL: Only terminate if we have CONFIRMED problems (not just check failures)
        # This prevents false positives from TraCI timing issues
        if has_teleport_or_collision or not ego_exists:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {
                "ego_id": self.ego_id,
                "step": self._steps,
                "collision": True,
                "success": False,
                "reason": "teleport_or_collision" if has_teleport_or_collision else "ego_removed",
                "teleport": has_teleport_or_collision
            }
            return obs, 0.0, True, False, info

        # --- (F) snapshot AFTER step (only if ego still exists) ---
        curr_edge = curr_lane_idx = curr_pos = None
        try:
            curr_edge = traci.vehicle.getRoadID(self.ego_id)
            curr_lane_idx = traci.vehicle.getLaneIndex(self.ego_id)
            curr_pos = traci.vehicle.getLanePosition(self.ego_id)
        except traci.TraCIException as e:
            # TraCI error accessing ego - check if it's a "vehicle not known" error
            # Only terminate if it's a critical error, not just a timing issue
            error_msg = str(e).lower()
            if "not known" in error_msg or "does not exist" in error_msg:
                # Ego actually doesn't exist - terminate
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                info = {
                    "ego_id": self.ego_id,
                    "step": self._steps,
                    "collision": True,
                    "success": False,
                    "reason": "ego_not_known",
                    "traci_error": True
                }
                return obs, 0.0, True, False, info
            # Otherwise, just continue with None values (will be handled later)
        except Exception:
            # Non-TraCI exceptions - just continue
            pass

        # --- (G) detect lane change ---
        # CRITICAL: Only detect lane change if we explicitly requested it (lat_cmd == 1)
        if self._pending_lc:
            self._pending_lc_steps += 1

            # Only mark success if we see transition from start_lane -> target_lane
            # AND we're still in the control zone
            if (prev_lane_idx is not None and curr_lane_idx is not None and
                prev_lane_idx == self.start_lane and curr_lane_idx == self.target_lane and
                (self.control_zone_edge_ID is None or curr_edge == self.control_zone_edge_ID)):
                self._ppo_lane_change = True
                self._pending_lc = False
                self._pending_lc_steps = 0
                # Lane change successfully detected

            # Clear if timeout or left control zone
            if (self._pending_lc_steps >= 6 or 
                (self.control_zone_edge_ID is not None and curr_edge is not None and 
                 curr_edge != self.control_zone_edge_ID)):
                self._pending_lc = False
                self._pending_lc_steps = 0

    

        # --- (H) commit route at gate (only if lane change occurred) ---
        if (not self._committed and ego_exists and curr_pos is not None and 
            curr_pos >= self.gate_pos and
            (self.control_zone_edge_ID is None or curr_edge == self.control_zone_edge_ID)):
            # CRITICAL: Only commit if PPO actually changed lanes
            if self._ppo_lane_change:
                self._took_exit = bool(self._commit_route_at_gate(self.exit_edge_id))
            else:
                self._took_exit = False
            self._committed = True

        # --- (I) compute reward ---
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

        # CRITICAL: Scale down all rewards to prevent value function instability
        # Original rewards were too large (hundreds), causing value_loss in millions
        # Scale factor: 0.1 to bring rewards into reasonable range (-1 to +1 per step)
        REWARD_SCALE = 0.1
        
        # Reward for entering control zone (encourages agent to reach the task area)
        if in_control_zone and not getattr(self, '_entered_control_zone', False):
            reward += 1.0  # Will be scaled at end
            reward_components['R_enter_zone'] = 1.0
            self._entered_control_zone = True
        
        # Reward for attempting lane change (encourages exploration)
        # ONE-TIME reward per episode to prevent reward spam and reduce variance
        if in_control_zone and lat_cmd == 1 and not self._attempt_reward_given:
            reward += 2.0  # Will be scaled at end
            reward_components['R_attempt_lc'] = 2.0
            self._attempt_reward_given = True
        
        # Additional reward for making progress in control zone (encourages forward movement)
        if in_control_zone:
            try:
                pos = traci.vehicle.getLanePosition(self.ego_id) if ego_exists else None
                if pos is not None and pos > getattr(self, '_last_pos_in_zone', 0.0):
                    progress = pos - getattr(self, '_last_pos_in_zone', 0.0)
                    reward += progress * 0.01  # Will be scaled at end
                    reward_components['R_progress'] = progress * 0.01
                    self._last_pos_in_zone = pos
                elif pos is not None:
                    self._last_pos_in_zone = pos
            except Exception:
                pass
        
        # PENALTY for taking too long to change lanes (encourages faster decisions)
        # BOUNDED penalty to prevent unbounded growth and reduce variance
        if in_control_zone and not self._ppo_lane_change:
            # Count steps since entering control zone without lane change
            steps_in_zone = getattr(self, '_steps_in_control_zone', 0) + 1
            self._steps_in_control_zone = steps_in_zone
            
            # BOUNDED delay penalty: small, stable, capped to prevent explosion
            # Penalty: -0.2 per step after 5 steps, capped at -5.0 total
            if steps_in_zone > 5:
                delay_penalty = -min(5.0, 0.2 * (steps_in_zone - 5))
                reward += delay_penalty
                reward_components['R_delay'] = delay_penalty
        elif in_control_zone and self._ppo_lane_change:
            # Reset counter once lane change occurs
            self._steps_in_control_zone = 0
        
        # Reward for successfully changing lanes (one-time reward when detected)
        if self._ppo_lane_change and not getattr(self, '_lc_reward_given', False):
            reward += 10.0  # Will be scaled at end
            self._lc_reward_given = True
            reward_components['R_lane_change'] = 10.0

        terminated, truncated, reason, collision, success = self._check_done()
        
        # Terminal reward for success (reaching exit after lane change)
        if success:
            reward += 20.0  # Will be scaled at end
            reward_components['R_success'] = 20.0
        elif collision:
            reward -= 5.0  # Will be scaled at end
            reward_components['R_collision_penalty'] = -5.0
        
        # CRITICAL: Scale final reward to prevent value function instability
        # This brings rewards from range [-100, +300] to range [-10, +30] per step
        reward = reward * REWARD_SCALE
        
        # CRITICAL: Clip reward to reduce variance and stabilize value function learning
        # Prevents extreme outliers from destabilizing the critic
        reward = float(np.clip(reward, -5.0, 5.0))

        if terminated or truncated:
            # Episode ended - no debug output for performance
            pass

        # --- (J) info dict ---
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


    def _apply_safety_intervention(self, obs: np.ndarray, lat_cmd: int):
        """Safety intervention: abort lane change if unsafe."""
        rp = 0.0
        if lat_cmd != 1:
            return lat_cmd, rp

        Dy_c1 = float(obs[START_C1 + 0])
        Dy_c3 = float(obs[START_C3 + 0])
        ds = getattr(self.lat_ctrl, "detect_dist", 10.0)
        
        # INCREASED SAFETY THRESHOLD: Require larger gaps for safe lane changes
        # Original: 10m is too small - gaps of 20-30m are always available, making task trivial
        # Increased to 50m to require agent to find very large gaps or make decisions in tight traffic
        # This makes the task significantly more challenging
        safe_gap_threshold = 50.0  # Require at least 50m gap (was 10m, then 30m)

        unsafe_front = (abs(Dy_c1) < 900.0 and 0.0 < Dy_c1 < safe_gap_threshold)
        unsafe_back = (abs(Dy_c3) < 900.0 and -safe_gap_threshold < Dy_c3 < 0.0)
        
        if unsafe_front or unsafe_back:
            lat_cmd = 2
            rp = -1.0
        
        return lat_cmd, rp


    def _check_done(self):
        """
        Check termination conditions according to paper:
        - Success: ego changed lanes AND took exit (entered exit_edge_id)
        - Failure: collision, timeout, or committed but didn't take exit
        """
        terminated = False
        truncated = False
        collision = False
        success = False
        reason = None

        ego_exists = self.ego_id in traci.vehicle.getIDList()

        # --- 1) Check collisions & teleports first (CRITICAL: must prevent success) ---
        has_collision_or_teleport = False
        try:
            colliding = set(traci.simulation.getCollidingVehiclesIDList())
            starting_teleports = set(traci.simulation.getStartingTeleportIDList())
            teleporting = set(traci.simulation.getTeleportingVehiclesIDList())
            all_teleporting = starting_teleports | teleporting
            
            # Check if ego collided or teleported
            has_collision_or_teleport = (self.ego_id in colliding or self.ego_id in all_teleporting)
        except Exception:
            colliding = set()
            all_teleporting = set()

        # CRITICAL: If collision/teleport occurred, ALWAYS return failure (success=False)
        if has_collision_or_teleport:
            return True, False, "collision_or_teleport", True, False

        # --- 2) Ego removed from simulation ---
        if not ego_exists:
            return True, False, "ego_removed", True, False

        # --- 3) Check if ego successfully took exit (entered exit_edge_id) ---
        # According to paper: success = changed lanes AND took exit
        # NOTE: We only reach here if NO collision/teleport occurred
        if ego_exists and self.exit_edge_id:
            try:
                curr_edge = traci.vehicle.getRoadID(self.ego_id)
                # Success: ego entered exit edge after successful lane change
                if curr_edge == self.exit_edge_id and self._ppo_lane_change and self._took_exit:
                    # Final safety check: verify no collision occurred
                    try:
                        final_colliding = set(traci.simulation.getCollidingVehiclesIDList())
                        if self.ego_id in final_colliding:
                            return True, False, "collision_or_teleport", True, False
                    except Exception:
                        pass
                    return True, False, "success_exit", False, True
            except Exception:
                pass

        # --- 4) Committed but didn't take exit = failure ---
        if self._committed and (not self._took_exit):
            return True, False, "chose_mainline", False, False

        # --- 5) Timeout ---
        if self._steps >= self._max_steps:
            truncated = True
            reason = "timeout"
            # If timeout but ego changed lanes, it's partial success (but still failure)
            # This encourages faster lane changes

        return terminated, truncated, reason, collision, success

    
    def _reached_goal(self):
        """
        DEPRECATED: Use _check_done() directly instead.
        This method is kept for backward compatibility but success is now checked in _check_done().
        """
        return False  # Always return False - success is now checked in _check_done()


    def _commit_route_at_gate(self, exit_edge_id: str | None) -> bool:
        """Route ego to exit_edge_id after successful lane change."""
        if self.ego_id not in traci.vehicle.getIDList():
            return False

        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            lane = traci.vehicle.getLaneIndex(self.ego_id)
            pos = traci.vehicle.getLanePosition(self.ego_id)
        except Exception:
            return False

        if (self.control_zone_edge_ID is not None and edge != self.control_zone_edge_ID) or pos is None or pos < self.gate_pos:
            return False

        took_exit = (lane == self.target_lane)

        if took_exit and exit_edge_id:
            try:
                traci.vehicle.setRoute(self.ego_id, [edge, exit_edge_id])
                if exit_edge_id not in traci.vehicle.getRoute(self.ego_id):
                    print(f"[WARNING] Failed to route to {exit_edge_id}")
            except Exception as e:
                print(f"[WARNING] Route error: {e}")

        return took_exit



