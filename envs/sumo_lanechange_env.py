# envs/sumo_lanechange_env.py

import gymnasium as gym                        # Gym API for RL environments
import numpy as np                # numerical arrays
import traci                     # SUMO's TraCI Python client
from gymnasium import spaces            # space definitions for obs/actions

# your modules (paths unchanged)
from controllers.longitudinal_idm import IDMController   # low-level longitudinal controller (IDM)
from controllers.lateral_controller import LateralController  # low-level lateral controller
from utils.action_decoder import decode_action
from utils.safety_intervention import DS_THRESHOLD, apply_safety_intervention
from utils.state_extraction import (
    get_state_with_info,   # builds the goal-conditioned observation + metadata
    OBS21_SCHEMA,
    OBS_DIM,
)


def compute_pending_lc_fsm_outcome(
    *,
    pending_lc: bool,
    curr_lane: int | None,
    lc_start_lane: int | None,
    lc_target_lane: int | None,
    lc_start_step: int | None,
    steps: int,
    lc_max_steps: int,
) -> tuple[bool, str | None, bool]:
    """
    Layer-3 pending lane-change FSM transition after a SUMO step.

    Returns:
        lc_success: True when ego reached ``lc_target_lane`` from ``lc_start_lane``.
        lc_fail_reason: "lc_timeout" on timeout, otherwise None while merging.
        clear_pending: If True, caller should clear pending LC state.

    While the vehicle is still merging, returns (False, None, False) so pending
    stays armed (avoids clearing every timestep with a spurious failure reason).
    """
    if not pending_lc:
        return False, None, False
    if curr_lane is None:
        return False, None, False

    if (
        lc_start_lane is not None
        and lc_target_lane is not None
        and curr_lane != lc_start_lane
        and curr_lane == lc_target_lane
    ):
        return True, None, True

    if lc_start_step is not None and (steps - lc_start_step) > lc_max_steps:
        return False, "lc_timeout", True

    return False, None, False


# layer 0 of training implementation

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
                 lc_step = 20, 
                 debug_mode: bool = False,
                 use_gui: bool = False,  # launch sumo-gui instead of sumo (for debugging)
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
        self.target_lane = target_lane # configured target lane in the control zone
        self.control_zone_edge_ID = control_zone_edge # the edge where the ego vehicle is controlled
        self.exit_edge_id = exit_edge_id  # optional route target after successful LC in control zone
        
        # --- Gym spaces  ---
        # Observation layout (OBS_DIM features) defined in utils/state_extraction.py:
        #   [ego(5), C0(4), C1(4), C2(4), C3(4), lane_error(1)]
        # The trailing `lane_error` makes the policy goal-conditioned on target_lane.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)                                      # 6 discrete actions

        self._steps = 0

        self.debug_mode = debug_mode
        self.use_gui = use_gui
        idm_params = idm_params or {}
        lateral_params = lateral_params or {}
        self.longitudinal_ctrl = IDMController(dt=self.dt, **idm_params)
        self.lateral_ctrl = LateralController(**lateral_params)
        self._obs_debug_steps = 5
        self._obs_min = None
        self._obs_max = None
        # Unique TraCI connection label for this env instance so that multiple
        # simultaneous envs (e.g. training + eval) never share the same SUMO
        # process.  traci.switch(label) is called at the top of every method
        # that uses TraCI, redirecting all global traci.xxx calls to the right
        # connection without requiring changes to state_extraction.py.
        self._traci_label = f"sumo_{id(self)}"

        #tracking lane change variables
        self._pending_lc = False #whether a lane change is pending
        self._lc_start_lane = None # runtime lane where current LC attempt started
        self._lc_target_lane = None # runtime target lane for current LC attempt
        self._lc_start_step = None #the step the ego vehicle starts lane changing
        self._lc_max_steps = lc_step #the maximum number of steps a lane change can take
        self._controller_owner = None  # "sumo" or "ppo"
        self._last_known_edge = None
        self._last_known_lane = None
        self._prev_lane_for_debug = None

    def reset(self, seed=None, options=None):
        #1. gym seed handling
        """Start (or restart) SUMO, choose an ego from the desired flow, and return the initial observation for a new episode."""
        super().reset(seed=seed)                       # inform Gym we've reset
        if seed is None:
            sumo_seed = int(self.np_random.integers(0, 2**31 - 1))
        else:
            sumo_seed = int(seed)
        if self.debug_mode:
            print("[RESET] before traci.start()")
        #2.  if THIS env's TraCI session is still open from a previous episode, close it
        try:
            traci.getConnection(self._traci_label).close()
            if self.debug_mode:
                print("[RESET] closing old TraCI session")
        except traci.TraCIException:
            pass  # no prior session for this label
        #3. reset the episode-level variables
        self._steps = 0
        self._obs_min = None # reset the observation min and max to None
        self._obs_max = None
        self._clear_pending_lc()
        self._controller_owner = None
        self._last_known_edge = None
        self._last_known_lane = None
        self._prev_lane_for_debug = None

        warmup_steps = 40  # enough for multiple flow vehicles to reach the control zone edge
        #4. start a new TraCI session
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        try:   
            traci.start([
                sumo_binary,
                "-c", self.sumo_cfg_path,
                "--step-length", str(self.step_length),
                "--seed", str(sumo_seed),
                "--no-step-log", "true",
                "--time-to-teleport", "-1",  # Disable teleporting (vehicles removed on collision)
                "--collision.action", "remove",  # Remove vehicles on collision instead of teleporting
            ], label=self._traci_label, numRetries=3)
            traci.switch(self._traci_label)  # route all global traci.xxx calls to this env
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
        
        if self.debug_mode:
            print("[RESET] traci.start() succeeded")
        for _ in range(warmup_steps):
            traci.simulationStep()
        
        #5. choose an ego from the desired flow
        self._choose_ego_from_flow(
            self.ego_flow_id,
            warmup_steps=0,  # Prevent double warmup - already did warmup above
            spawn_edge_id=self.control_zone_edge_ID, 
            spawn_lane_idx=self.start_lane
        )
        self._update_controller_handoff(in_control_zone=self._is_in_control_zone())
        if self.debug_mode:
            print(f"[RESET] ego_id found: {self.ego_id}")

        obs, state_info = self._get_state()
        if self.debug_mode:
            self._prev_lane_for_debug = self._safe_get_lane_index()
        info = {
            "ego_id": self.ego_id,
            "step": self._steps,
            **state_info,
        }
        obs = obs.astype(np.float32)
        if self.debug_mode:
            print(f"[RESET] Chosen ego_id={self.ego_id} from flow='{self.ego_flow_id}'")
            self._update_obs_debug_stats(obs)
            self._debug_print_key_fields(obs, state_info, prefix="[RESET]")
        assert obs.shape == self.observation_space.shape

        return obs, info

    def step(self, action: int):
        """
        Apply an RL action, advance SUMO one step, and return:
        (next_obs, reward, terminated, truncated, info)
        """
        traci.switch(self._traci_label)  # ensure global traci points to this env's connection
        self._steps += 1 

        # --- check if ego exists ---
        ego_exists = self.ego_id in traci.vehicle.getIDList()
        if self.debug_mode:
            print(f"[STEP {self._steps}] ego_exists={ego_exists}")
        if not ego_exists:
            # Ego vanished before action application.
            reason = self._classify_ego_disappearance(default_reason="ego_missing_pre")
            if self.debug_mode:
                print(
                    f"[STEP {self._steps}] ego gone — reason={reason} "
                    f"last_edge={self._last_known_edge} last_lane={self._last_known_lane}"
                )
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            if self._pending_lc:
                lc_fail_reason = reason
                self._clear_pending_lc()
            else:
                lc_fail_reason = None
            lc_info = self._build_lc_info(
                curr_lane=None,
                lc_success=False,
                lc_fail_reason=lc_fail_reason,
            )
            info = {
                "ego_id": self.ego_id,
                "step": self._steps,
                "reason": reason,
                "curr_edge": self._last_known_edge,
                "curr_lane": self._last_known_lane,
                "controller_owner": self._controller_owner or "sumo",
                "in_control_zone": False,
                "lat_cmd_raw": None,
                "lat_cmd_applied": None,
                "lat_cmd_safe": None,
                "lat_cmd_blocked_outside_control_zone": False,
                "rp": 0.0,
                "safety_intervened": False,
                "safety_reason": "none",
                "d_s": None,
                "d0": None,
                "d1": None,
                "d2": None,
                "route_committed": False,
                **lc_info,
            }
            return obs, 0.0, True, False, info

        # Layer 2: decode action, then apply only if PPO owns control in control zone.
        lon_cmd, lat_cmd = decode_action(int(action))
        lat_cmd_raw = int(lat_cmd)
        in_control_zone = self._is_in_control_zone()
        controller_owner = self._update_controller_handoff(in_control_zone=in_control_zone)
        rp = 0.0  # penalty semantics: intervention penalty only applies in PPO control zone
        safety_intervened = False
        safety_reason = "none"
        d0 = d1 = d2 = float("inf")

        lat_cmd_applied = 0
        if controller_owner == "ppo":
            pre_obs, _ = self._get_state()
            safety = apply_safety_intervention(
                obs=pre_obs,
                lat_cmd_raw=lat_cmd_raw,
                in_control_zone=bool(in_control_zone),
            )
            lat_cmd_applied = int(safety.lat_cmd_safe)
            rp = float(safety.rp)
            safety_reason = str(safety.reason)
            d0, d1, d2 = float(safety.d0), float(safety.d1), float(safety.d2)
            safety_intervened = lat_cmd_applied != lat_cmd_raw
            pre_lane = self._safe_get_lane_index()
            self._maybe_start_pending_lc(lat_cmd=lat_cmd_applied, curr_lane=pre_lane)
            self._apply_action(pre_obs, lon_cmd, lat_cmd_applied)
        elif self._pending_lc:
            # A pending PPO lane-change attempt does not persist after handoff to SUMO.
            self._clear_pending_lc()

        traci.simulationStep() 

        obs, state_info = self._get_state() 
        obs = obs.astype(np.float32)
        if not state_info.get("ego_present", True): #if the ego vehicle is not present, classify the disappearance reason
            # Ego disappeared during this simulation step (e.g., collision removal).
            reason = self._classify_ego_disappearance(default_reason="ego_missing_post")
            if self.debug_mode:
                print(
                    f"[STEP {self._steps}] ego gone after simulationStep — reason={reason} "
                    f"last_edge={self._last_known_edge} last_lane={self._last_known_lane}"
                )
            if self._pending_lc:
                lc_fail_reason = reason
                self._clear_pending_lc()
            else:
                lc_fail_reason = None
            lc_info = self._build_lc_info(
                curr_lane=None,
                lc_success=False,
                lc_fail_reason=lc_fail_reason,
            )
            info = {
                "ego_id": self.ego_id,
                "step": self._steps,
                "reason": reason,
                "curr_edge": self._last_known_edge,
                "curr_lane": self._last_known_lane,
                "controller_owner": controller_owner,
                "in_control_zone": bool(in_control_zone),
                "lat_cmd_raw": int(lat_cmd_raw),
                "lat_cmd_applied": int(lat_cmd_applied),
                "lat_cmd_safe": int(lat_cmd_applied),
                "lat_cmd_blocked_outside_control_zone": bool((controller_owner == "sumo") and (lat_cmd_raw != 0)),
                "rp": float(rp),
                "safety_intervened": bool(safety_intervened),
                "safety_reason": safety_reason,
                "d_s": None if controller_owner == "sumo" else float(DS_THRESHOLD),
                "d0": None if controller_owner == "sumo" else d0,
                "d1": None if controller_owner == "sumo" else d1,
                "d2": None if controller_owner == "sumo" else d2,
                "route_committed": False,
                **state_info,
                **lc_info,
            }
            return obs, 0.0, True, False, info


        #if the ego vehicle is present, check if a lane change is pending
        curr_edge = self._safe_get_edge_id()
        curr_lane = self._safe_get_lane_index()
        self._last_known_edge = curr_edge
        self._last_known_lane = curr_lane
        if self.debug_mode:
            prev_lane = self._prev_lane_for_debug
            if prev_lane is not None and curr_lane is not None and int(curr_lane) != int(prev_lane):
                ego_px_now = float(obs[OBS21_SCHEMA["ego.px"]])
                print(
                    f"[STEP {self._steps}] LANE-TRANSITION "
                    f"{prev_lane} -> {curr_lane} edge={curr_edge} ego_px={ego_px_now:.3f}"
                )
            self._prev_lane_for_debug = curr_lane

        # Layer 3 strict lane-change detection:
        # success only on an actual lane-index transition to configured target lane
        # while an LC attempt is pending.
        lc_success, lc_fail_reason, clear_pending = compute_pending_lc_fsm_outcome(
            pending_lc=self._pending_lc,
            curr_lane=curr_lane,
            lc_start_lane=self._lc_start_lane,
            lc_target_lane=self._lc_target_lane,
            lc_start_step=self._lc_start_step,
            steps=self._steps,
            lc_max_steps=self._lc_max_steps,
        )
        if clear_pending:
            self._clear_pending_lc()
        route_committed = self._maybe_commit_route_to_exit(
            lc_success=lc_success,
            curr_lane=curr_lane,
            in_control_zone=in_control_zone,
        )

        if self.debug_mode:
            self._update_obs_debug_stats(obs)
            if self._steps <= self._obs_debug_steps:
                self._debug_print_key_fields(obs, state_info, prefix=f"[STEP {self._steps}]")
                lane_idx = self._safe_get_lane_index()
                print(
                    f"[STEP {self._steps}] action={int(action)} "
                    f"lon_cmd={lon_cmd} lat_cmd_raw={lat_cmd_raw} "
                    f"lat_cmd_applied={lat_cmd_applied} in_control_zone={in_control_zone} "
                    f"controller_owner={controller_owner} "
                    f"safety_intervened={safety_intervened} rp={rp:.3f} "
                    f"safety_reason={safety_reason} d0={d0:.3f} d1={d1:.3f} d2={d2:.3f} "
                    f"lane_idx={lane_idx}"
                )

        info = {
            "ego_id": self.ego_id,
            "step": self._steps,
            "controller_owner": controller_owner,
            "in_control_zone": bool(in_control_zone),
            "curr_edge": curr_edge,
            "lat_cmd_raw": int(lat_cmd_raw),
            "lat_cmd_applied": int(lat_cmd_applied),
            "lat_cmd_safe": int(lat_cmd_applied),
            "lat_cmd_blocked_outside_control_zone": bool((controller_owner == "sumo") and (lat_cmd_raw != 0)),
            "rp": float(rp),
            "safety_intervened": bool(safety_intervened),
            "safety_reason": safety_reason,
            "d_s": None if controller_owner == "sumo" else float(DS_THRESHOLD),
            "d0": None if controller_owner == "sumo" else d0,
            "d1": None if controller_owner == "sumo" else d1,
            "d2": None if controller_owner == "sumo" else d2,
            "route_committed": bool(route_committed),
            **state_info,
            **self._build_lc_info(curr_lane=curr_lane, lc_success=lc_success, lc_fail_reason=lc_fail_reason),
        }

        terminated = False
        truncated = self._steps >= self._max_steps
        if self.debug_mode and (terminated or truncated):
            self._debug_print_obs_minmax()
        return obs, 0.0, terminated, truncated, info


    def close(self):
        """Close the TraCI connection cleanly and call the parent close()."""
        if self.debug_mode:
            print("[CLOSE] closing TraCI connection")
        try:
            traci.getConnection(self._traci_label).close()
        except traci.TraCIException:
            pass  # already closed or never opened

        # Call Gym's built-in close (important for some wrappers)
        try:
            super().close()
        except Exception:
            pass


    # ------------------------- helpers -------------------------

    def _get_state(self):
        """
        Returns the state of the ego vehicle and the surrounding vehicles
        and the state information for debugging/sanity checks.

        The state is an OBS_DIM-dimensional vector with the following features:
        - ego: [Px, Vx, Ax, Py, Vy]
        - c0:  [Dx, Vx, Ax, Py]   current-lane leader
        - c1:  [Dx, Vx, Ax, Py]   target-lane leader   (uses self.target_lane)
        - c2:  [Dx, Vx, Ax, Py]   current-lane follower
        - c3:  [Dx, Vx, Ax, Py]   target-lane follower (uses self.target_lane)
        - lane_error: target_lane - current_lane  (signed scalar, goal feature)

        The state information is a dictionary with the following keys:
        - ego_present: whether the ego vehicle is present
        - target_lane_exists: whether the target lane exists on the current edge
        - missing_neighbors: the number of missing neighbors
        - state_valid: whether the state is valid
        - target_lane_index: the configured target lane (echoed for debugging)
        - current_lane_index: the ego's current lane index (or None)
        - lane_error: float, equal to obs[-1]
        """
        return get_state_with_info(self.ego_id, target_lane_index=int(self.target_lane))

    def _clear_pending_lc(self):
        self._pending_lc = False
        self._lc_start_lane = None
        self._lc_target_lane = None
        self._lc_start_step = None

    def _maybe_start_pending_lc(self, lat_cmd: int, curr_lane: int | None):
        """
        Arm a new lane change attempt if the lateral command is to change to the target lane 
        and the current lane is known and not the target lane.
        Does not arm a new lane change attempt if a lane change is already pending,
        the lateral command is not to change to the target lane, the current lane is not known, 
        or the ego is already in the configured target lane.
        Sets _pending_lc to True, _lc_start_lane to the current lane, _lc_target_lane to the target lane, and _lc_start_step to the current step.
        """
        if self._pending_lc: #if a lane change is already pending, do not arm a new one and return
            return
        if lat_cmd != 1: #if the lateral command is not to change to the target lane, do not arm a new one and return
            return
        if curr_lane is None: #if the current lane is not known, do not arm a new one and return
            return
        # If ego is already in configured target lane, do not arm a new LC attempt.
        if int(curr_lane) == int(self.target_lane):
            return
        self._pending_lc = True #arm a new lane change attempt
        self._lc_start_lane = int(curr_lane)
        # Use configured scenario target lane (Layer 3 requirement).
        self._lc_target_lane = int(self.target_lane)
        self._lc_start_step = int(self._steps)

    def _build_lc_info(self, curr_lane: int | None, lc_success: bool, lc_fail_reason: str | None):
        """
        Builds the lane change information for the ego vehicle.
        The lane change information is a dictionary with the following keys:
        - pending_lc: whether a lane change is pending
        - lc_success: whether the lane change was successful
        - lc_fail_reason: the reason the lane change failed
        - lc_start_lane: the lane the ego vehicle started lane changing from
        - lc_target_lane: the lane the ego vehicle is lane changing to
        - curr_lane: the current lane of the ego vehicle
        - lc_start_lane_matches_config: whether the lane the ego vehicle started lane changing from matches the configured target lane
        """
        return {
            "pending_lc": self._pending_lc,
            "lc_success": bool(lc_success),
            "lc_fail_reason": lc_fail_reason,
            "lc_start_lane": self._lc_start_lane,
            "lc_target_lane": self._lc_target_lane,
            "curr_lane": curr_lane,
            # Validation signal: scenario expectation vs runtime LC start lane.
            "lc_start_lane_matches_config": (
                None if self._lc_start_lane is None else (self._lc_start_lane == self.start_lane)
            ),
        }

    def _classify_ego_disappearance(self, default_reason: str) -> str:
        """
        Best-effort disappearance reason with priority:
          collision > teleport > arrived > default.
        """
        try:
            colliding = set(traci.simulation.getCollidingVehiclesIDList())
        except Exception:
            colliding = set()
        if self.ego_id in colliding:
            return "ego_collision"

        try:
            teleporting = (
                set(traci.simulation.getStartingTeleportIDList())
                | set(traci.simulation.getTeleportingVehiclesIDList())
            )
        except Exception:
            teleporting = set()
        if self.ego_id in teleporting:
            return "ego_teleport"

        try:
            arrived = set(traci.simulation.getArrivedIDList())
        except Exception:
            arrived = set()
        if self.ego_id in arrived:
            return "ego_arrived"

        return default_reason

    def _set_controller_owner(self, owner: str) -> None:
        """
        Switch ego low-level control owner.
        - owner == "ppo": disable SUMO autonomous lane changes.
        - owner == "sumo": restore SUMO lane-change control and release speed commands.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return
        try:
            if owner == "ppo":
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)
                if self.debug_mode:
                    print(f"[CONTROL] owner=ppo lane_change_mode=0 ego_id={self.ego_id}")
            else:
                # SUMO default lane-change mode (bitset default).
                traci.vehicle.setLaneChangeMode(self.ego_id, 1621)
                # Release any prior TraCI speed command so SUMO car-following resumes.
                traci.vehicle.setSpeed(self.ego_id, -1)
                if self.debug_mode:
                    print(f"[CONTROL] owner=sumo lane_change_mode=1621 ego_id={self.ego_id}")
        except traci.TraCIException:
            if self.debug_mode:
                print(f"[CONTROL] failed to switch owner={owner} for ego_id={self.ego_id}")

    def _update_controller_handoff(self, in_control_zone: bool) -> str:
        desired_owner = "ppo" if in_control_zone else "sumo"
        if self._controller_owner != desired_owner:
            self._set_controller_owner(desired_owner)
            self._controller_owner = desired_owner
        return self._controller_owner

    def _apply_action(self, pre_obs: np.ndarray, lon_cmd: int, lat_cmd: int) -> None:
        """
        Apply lateral and longitudinal low-level controllers to ego.
        If the lat_cmd is 1=0, the ego vehicle will keep the current lane.
        If the lat_cmd is 1, the ego vehicle will change to the target lane.
        If the lat_cmd is 2, the ego vehicle will abort the lane change and return to the current lane.

        lon_cmd:
            -1: slow down
            0: maintain speed
            1: accelerate
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return
        try:
            self.lateral_ctrl.execute(self.ego_id, lat_cmd, target_lane_index=int(self.target_lane))
        except traci.TraCIException:
            # Non-fatal in Layer 2; proceed with simulation step.
            pass
        try:
            v_cmd = float(self.longitudinal_ctrl.compute(pre_obs, lon_cmd))
            traci.vehicle.setSpeed(self.ego_id, max(0.0, v_cmd))
        except traci.TraCIException:
            pass

    def _safe_get_lane_index(self):
        if self.ego_id not in traci.vehicle.getIDList():
            return None
        try:
            return int(traci.vehicle.getLaneIndex(self.ego_id))
        except traci.TraCIException:
            return None

    def _safe_get_edge_id(self):
        if self.ego_id not in traci.vehicle.getIDList():
            return None
        try:
            return str(traci.vehicle.getRoadID(self.ego_id))
        except traci.TraCIException:
            return None

    def _is_in_control_zone(self) -> bool:
        curr_edge = self._safe_get_edge_id()
        return curr_edge is not None and curr_edge == self.control_zone_edge_ID

    def _maybe_commit_route_to_exit(self, lc_success: bool, curr_lane: int | None, in_control_zone: bool) -> bool:
        """
        Commit route target only after successful LC in control zone.
        No-op when exit_edge_id is not configured.
        """
        if not lc_success:
            return False
        if not in_control_zone:
            return False
        if curr_lane is None or int(curr_lane) != int(self.target_lane):
            return False
        if not self.exit_edge_id:
            return False
        if self.ego_id not in traci.vehicle.getIDList():
            return False
        try:
            traci.vehicle.changeTarget(self.ego_id, self.exit_edge_id)
            return True
        except traci.TraCIException:
            return False

    def _update_obs_debug_stats(self, obs: np.ndarray):
        if self._obs_min is None:
            self._obs_min = obs.copy()
            self._obs_max = obs.copy()
            return
        self._obs_min = np.minimum(self._obs_min, obs)
        self._obs_max = np.maximum(self._obs_max, obs)

    def _debug_print_key_fields(self, obs: np.ndarray, state_info: dict, prefix: str):
        ego_px = float(obs[OBS21_SCHEMA["ego.px"]])
        ego_vx = float(obs[OBS21_SCHEMA["ego.vx"]])
        ego_py = float(obs[OBS21_SCHEMA["ego.py"]])
        c0_dx = float(obs[OBS21_SCHEMA["c0.dx"]])  # current-lane leader gap
        c1_dx = float(obs[OBS21_SCHEMA["c1.dx"]])  # target-lane leader gap
        c3_dx = float(obs[OBS21_SCHEMA["c3.dx"]])  # target-lane follower gap
        lane_error = float(obs[OBS21_SCHEMA["lane_error"]])
        print(
            f"{prefix} ego_px={ego_px:.3f} ego_vx={ego_vx:.3f} ego_py={ego_py:.3f} "
            f"c0_dx={c0_dx:.3f} c1_dx={c1_dx:.3f} c3_dx={c3_dx:.3f} "
            f"lane_error={lane_error:+.1f} target_lane={self.target_lane} "
            f"curr_lane={state_info.get('current_lane_index')} "
            f"ego_present={state_info.get('ego_present')} "
            f"missing_neighbors={state_info.get('missing_neighbors')}"
        )

    def _debug_print_obs_minmax(self):
        if self._obs_min is None or self._obs_max is None:
            return
        print(
            f"[EPISODE OBS RANGE] min={np.array2string(self._obs_min, precision=3)} "
            f"max={np.array2string(self._obs_max, precision=3)}"
        )

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
                #deterministic ego selection
                #candidates = sorted(candidates)
                #self.ego_id = candidates[0]
                #random ego selection
                # Pick a random candidate so the ego is not always the
                # frontrunner vehicle (lowest ID = spawned first = furthest
                # along the edge = no traffic ahead). Using np_random keeps
                # the selection deterministic when a fixed seed is passed to
                # reset(), which is important for the eval env.
                idx = int(self.np_random.integers(0, len(candidates)))
                self.ego_id = candidates[idx]
                #traci.vehicle.setSpeedMode(self.ego_id, 0) #disables for layer 0 
                #traci.vehicle.setLaneChangeMode(self.ego_id, 0)
                return

        raise RuntimeError(
            f"Could not find a vehicle from flow '{flow_prefix}' "
            f"on edge={spawn_edge_id} lane={spawn_lane_idx} within {timeout_steps} steps."
        )



