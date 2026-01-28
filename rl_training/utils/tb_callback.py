import numpy as np
from typing import List, Optional, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, log_action_stats: bool = True, log_gain_keys: Optional[List[str]] = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_action_stats = log_action_stats
        self.log_gain_keys = log_gain_keys or []
        self._config_logged = False
        self.ep_ret = None          # undiscounted sum per env
        self.ep_disc = None         # discounted sum per env
        self.pow_gamma = None       # current gamma^t per env

    def _on_training_start(self) -> None:
        if self._config_logged:
            return  
        cfg = None
        try:
            # Prefer VecEnv get_attr; falls back to best-effort unwrap.
            if hasattr(self.training_env, "get_attr"):
                vals = self.training_env.get_attr("config")
                cfg = vals[0] if vals else None
            else:
                env0 = getattr(self.training_env, "envs", [self.training_env])[0]
                cfg = getattr(getattr(env0, "env", env0), "config", None)
            self.gamma = self.model.gamma
        except Exception:
            pass
        if cfg is not None:
            try:
                self.logger.record_text("config/yaml", str(cfg))
            except Exception:
                pass
        self.gamma = float(getattr(self.model, "gamma"))
        self.n_envs = int(getattr(self.training_env, "num_envs", 1))
        self.ep_ret = np.zeros(self.n_envs, dtype=float)
        self.ep_disc = np.zeros(self.n_envs, dtype=float)
        self.pow_gamma = np.ones(self.n_envs, dtype=float)
        self.ep_alt_err = np.zeros(self.n_envs, dtype=float)
        self.ep_x_err = np.zeros(self.n_envs, dtype=float)
        self.ep_y_err = np.zeros(self.n_envs, dtype=float)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        # # Merge infos from all parallel envs, last one wins
        merged: Dict[str, Any] = {}
        for d in infos or []:
            if isinstance(d, dict):
                merged.update(d)
        # make sure that they are in info dict as well. !!
        # env_keys = ["max_stable_time"]
        # for k in env_keys:
        #     if k in merged:
        #         self.logger.record(f"env/{k}", merged[k])

        # gains    
        # for k in self.log_gain_keys:
        #     self.logger.record(f"gains/{k}", merged[k])

        # log errors for each episode
        alt_errors = merged["alt_err"]
        x_errors =   merged["x_err"]
        y_errors =   merged["y_err"]
        self.ep_alt_err +=   np.abs(merged["alt_err"])
        self.ep_x_err   +=   np.abs(merged["x_err"])
        self.ep_y_err   +=   np.abs(merged["y_err"])

        # log returns after each episode end
        rewards = np.asarray(self.locals["rewards"], dtype=float)     # shape: (n_envs,)
        dones   = np.asarray(self.locals["dones"], dtype=bool)        # shape: (n_envs,)
        
        
        # online update: G_t = Σ r_k * gamma^k
        self.ep_ret += rewards
        self.ep_disc += self.pow_gamma * rewards
        self.pow_gamma *= self.gamma


        # log + reset on episode end for each env
        for i, d in enumerate(dones):
            if d:
                self.logger.record("rollout/ep_ret_undisc", float(self.ep_ret[i]))
                self.logger.record("rollout/ep_ret_disc", float(self.ep_disc[i]))
                self.logger.record("rollout/ep_alt_err", float(self.ep_alt_err))
                self.logger.record("rollout/ep_x_err", float(self.ep_x_err))
                self.logger.record("rollout/ep_y_err", float(self.ep_y_err))
                
                # print(f"Done {i}")
                self.ep_ret[i] = 0.0
                self.ep_disc[i] = 0.0
                self.pow_gamma[i] = 1.0
                self.ep_alt_err = 0.0
                self.ep_x_err = 0.0
                self.ep_y_err = 0.0
        return True