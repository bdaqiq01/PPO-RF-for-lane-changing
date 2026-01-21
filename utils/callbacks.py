import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SuccessMetricsCallback(BaseCallback):
    """
    Logs rolling episode metrics based on `info` dict keys emitted by your env.
    Writes to SB3 logger -> shows up in progress.csv + TensorBoard.
    """

    def __init__(self, window_size: int = 100, log_every_steps: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.log_every_steps = log_every_steps

        self._success_hist = []
        self._collision_hist = []
        self._lc_hist = []
        self._exit_hist = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # In DummyVecEnv, infos/dones are lists length n_envs
        for info, done in zip(infos, dones):
            if done:
                self._success_hist.append(int(info.get("success", 0) or info.get("is_success", 0)))
                self._collision_hist.append(int(info.get("collision", 0)))
                self._lc_hist.append(int(info.get("_ppo_lane_change", 0)))
                self._exit_hist.append(int(info.get("_took_exit", 0)))

        # Log rolling means periodically
        if self.num_timesteps % self.log_every_steps == 0:
            if len(self._success_hist) > 0:
                w = min(self.window_size, len(self._success_hist))
                self.logger.record("custom/success_rate", float(np.mean(self._success_hist[-w:])))
                self.logger.record("custom/collision_rate", float(np.mean(self._collision_hist[-w:])))
                self.logger.record("custom/lane_change_rate", float(np.mean(self._lc_hist[-w:])))
                self.logger.record("custom/exit_rate", float(np.mean(self._exit_hist[-w:])))
                self.logger.record("custom/episodes", len(self._success_hist))

        return True
