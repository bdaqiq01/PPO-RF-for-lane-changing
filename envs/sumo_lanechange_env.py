# envs/sumo_lanechange_env.py

import os                         # stdlib: path handling for config file
import gymnasium as gym                        # Gym API for RL environments
import numpy as np                # numerical arrays
import traci                     # SUMO's TraCI Python client
from gymnasium import spaces            # space definitions for obs/actions

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
                 ego_flow_id, 
                 idm_params = None,
                 lateral_params = None):      # <— choose which flow to pull ego from
        super().__init__()                     # Gym boilerplate

        self.sumo_cfg_path = sumo_cfg_path     # path to .sumocfg (loads base.net.xml, base.rou.xml, etc.)
        self.step_length = step_length         # SUMO simulation step in seconds
        self.ego_flow_id = ego_flow_id         # we will pick ego from this flow (f_0, f_1, or f_2)
        #i am not what this is used for
        self.dt = float(step_length)          # time delta for the reward function and jerk calc
        #self.prev_ego_state = None                     # will hold the chosen SUMO vehicle id, e.g., "f_0.3"
        self.ego_id = None                   # will hold the chosen SUMO vehicle id, e.g., "f_0.3"

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

        # --- optional: simple episode step counter cap (helps termination) ---
        self._max_steps = max_steps  # maximum steps per episode
        self._steps = 0

    def reset(self, seed=None, options=None): #TODO REDO THIS 
        """Start (or restart) SUMO, choose an ego from the desired flow, and return the initial observation."""
        super().reset(seed=seed)                       # inform Gym we've reset

        # if an old TraCI session is still open, close it
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass
        
        #start a new TraCI session
        # choose binary: "sumo" for headless (fast), "sumo-gui" to watch
        sumo_binary = "sumo"                           # change to "sumo-gui" while debugging if you want
        traci.start([                                   # launch SUMO with your config and step length
            sumo_binary,
            "-c", self.sumo_cfg_path,
            "--step-length", str(self.step_length),
            "--no-warnings", "true",
            "--error-log", "NUL"  
        ])

        # pick the ego vehicle from the specified flow
        #option 1- of picking vehicle from a flow
        self._choose_ego_from_flow(self.ego_flow_id)  #need returns self.ego_id self.ego_id is defined in _choose_ego_from_flow()    
        self._initial_lane_idx = traci.vehicle.getLaneIndex(self.ego_id)
        route = traci.vehicle.getRoute(self.ego_id)
        self._target_edge = route[-1]   # final destination edge for ego - last edge in the route

        # reset step counter
        self._steps = 0
        obs = self._get_state() #this has to be produce the 21 dims that goes to PPO agent 

        # Build info
        lane_idx = None
        try:
            lane_idx = traci.vehicle.getLaneIndex(self.ego_id)
        except Exception:
            pass

        info = {   
            "ego_id": self.ego_id,
            "step_length": float(self.step_length),
            "seed": seed,
            "lane_index": lane_idx,
            "episode_start_time": traci.simulation.getTime()
        }

        return obs, info

    def step(self, action: int): #SUMO and environment step 
        
        """
        Apply an RL action (high-level lane-change decision), advance SUMO by one step,
        and return (next_obs, reward, terminated, truncated, info) in Gymnasium format.
        """
        self._steps += 1 # increment step counter

        # read current obs (21-d) for controllers/reward
        obs_t = self._get_state().astype(np.float32) #not sure about the astype part
        # decode discrete action -> (lateral_command, longitudinal_command)
        
        #   lateral: 0=keep lane, 1=change now, 2=abort lane change
        #   longitudinal: 0=follow current-lane leader, 1=follow target-lane leader
        lat_cmd, lon_cmd = decode_action(action)

        lat_cmd, rp = self._apply_safety_intervention(obs_t, lat_cmd)

        # longitudinal: compute a speed command from current obs + longitudinal intent
        v_cmd = self.longi_ctrl.compute(obs_t, lon_cmd)

        # lateral: execute lane command for the ego vehicle (controller will issue TraCI lane-change)
        self.lat_ctrl.execute(self.ego_id, lat_cmd) #Should handle the lane change command 

        # send speed to SUMO for the ego vehicle
        traci.vehicle.setSpeed(self.ego_id, float(v_cmd)) 
        #what about the lane change command?

        # advance the simulation one step
        traci.simulationStep() #advance the simulation by one step based on the configuraion in the sumo config  in the reset function

        # next observation and reward
        next_obs = self._get_state()

        reward, reward_components = compute_reward(
            obs_t=obs_t,
            obs_tp1=next_obs,
            ego_id=self.ego_id,
            lat_cmd=lat_cmd,
            dt=self.dt,
            ds=self.lat_ctrl.detect_dist,  # 10 m from your lateral macro
            rp=rp,
            v_desired=25.0,
        )
    
        # termination condition
        terminated, truncated, reason, collision, success = self._check_done()

        if terminated or truncated:
            try:
                edge_id = traci.vehicle.getRoadID(self.ego_id)
            except Exception:
                edge_id = None
            #print(f"[EP END] steps={self._steps}, reason={reason}, " f"collision={collision}, success={success}, edge={edge_id}")

        # optional info dict (empty is fine)
        info = {
            "ego_id": self.ego_id,
            "step": self._steps,
            "collision": bool(collision),
            "success": bool(success),
            "is_success": bool(success),   # SB3’s EvalCallback can use this
            "reason": reason,
            "rp": rp
            }
        info.update(reward_components)

        if self.ego_id in traci.vehicle.getIDList():
            try:
                info["lane_index"] = traci.vehicle.getLaneIndex(self.ego_id)
                info["v_ego"] = traci.vehicle.getSpeed(self.ego_id)
            except Exception:
                pass

        return next_obs, reward, bool(terminated), bool(truncated), info

    def close(self):
        """Close the TraCI connection cleanly and call the parent close()."""
        # Try closing TraCI
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
        # If your current get_state() signature takes no args, you can:
        #   - modify utils/state_extraction.py to read self.ego_id via a global or singleton, or
        #   - change get_state to accept ego_id and then: return get_state(self.ego_id)
        # Below assumes you've updated get_state to accept ego_id:
        return get_state(self.ego_id)

    def _choose_ego_from_flow(self, flow_prefix: str, warmup_steps: int = 200, timeout_steps: int = 2000):
        """
        Wait for vehicles to spawn, then pick one whose id starts with e.g. 'f_0.'.
        Your base.rou.xml defines flows f_0, f_1, f_2, so SUMO makes ids like 'f_0.0', 'f_0.1', ...
        """
        # optional warmup: let the flows inject some vehicles first
        for _ in range(warmup_steps):
            traci.simulationStep()

        steps = 0
        while steps < timeout_steps:
            ids = traci.vehicle.getIDList()
            # prefer vehicles from the requested flow
            candidates = [vid for vid in ids if vid.startswith(flow_prefix + ".")]
            if not candidates:
                # fallback: any flow-type vehicle (contains a dot, typical SUMO flow naming)
                candidates = [vid for vid in ids if "." in vid]

            if candidates:
                self.ego_id = candidates[0]                 # pick the first or randomize
                # put ego fully under RL control (disable SUMO’s built-in models for this vehicle)
                traci.vehicle.setSpeedMode(self.ego_id, 0)        # you control speed
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)   # you control lane changes
                return

            traci.simulationStep()
            steps += 1

        raise RuntimeError(
            f"Could not find a vehicle from flow '{flow_prefix}' within {timeout_steps} steps."
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
            colliding = traci.simulation.getCollidingVehiclesIDList()
            starting_teleports = traci.simulation.getStartingTeleportIDList()
            #teleporting = traci.simulation.getTeleportingVehiclesIDList()
        except Exception:
            colliding = []
            starting_teleports = []

        if ego_exists and (self.ego_id in colliding or self.ego_id in starting_teleports):
            # Treat any collision/teleport as episode failure
            terminated = True
            collision = True
            success = False
            reason = "collision_or_starting_teleport"
            return terminated, truncated, reason, collision, success

        # --- 2) Ego removed from simulation (not found anymore) ---
        if not ego_exists:
            # If it's gone and we didn't catch it as a collision/teleport above,
            # treat this as a failure (you can refine this later if needed).
            terminated = True
            collision = True
            success = False
            reason = "ego_removed"
            return terminated, truncated, reason, collision, success

        # --- 3) Ego still exists: check for goal ---
        if self._reached_goal():
            terminated = True
            success = True
            collision = False
            reason = "goal"
            return terminated, truncated, reason, collision, success

        # --- 4) Time/step limit reached (episode cap) ---
        if self._steps >= self._max_steps:
            truncated = True
            reason = "timeout"

        return terminated, truncated, reason, collision, success

    
    def _reached_goal(self):
        """
        Check if ego successfully reached target lane (right lane) before exit.
        Returns True if:
        - Ego is in the right-most lane (lane index 0)
        - Ego hasn't exited the network yet
        - Optionally: ego is past a certain position/edge
        """
        try:
            if self.ego_id not in traci.vehicle.getIDList():
                return False
            
            curr_lane_idx = traci.vehicle.getLaneIndex(self.ego_id)
            edge_id = traci.vehicle.getRoadID(self.ego_id)
            x, y = traci.vehicle.getPosition(self.ego_id)

            if edge_id == self._target_edge:
                return True
            return False
        except Exception:
            return False   

