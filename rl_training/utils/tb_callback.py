import numpy as np
from typing import List, Optional, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, log_action_stats: bool = True, log_gain_keys: Optional[List[str]] = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_action_stats = log_action_stats
        self.log_gain_keys = log_gain_keys or []
        self._config_logged = False

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
        except Exception:
            pass
        if cfg is not None:
            try:
                self.logger.record_text("config/yaml", str(cfg))
            except Exception:
                pass
        self._config_logged = True

    def _safe_record_scalar(self, key: str, val: Any):
        try:
            if isinstance(val, (np.generic, np.ndarray)):
                if np.isscalar(val):
                    val = float(val)
                else:
                    # best default: mean over array
                    val = float(np.mean(val))
            elif isinstance(val, (int, float)):
                val = float(val)
            else:
                return
            self.logger.record(key, val)
        except Exception:
            pass

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        # # Merge infos from all parallel envs, last one wins
        merged: Dict[str, Any] = {}
        for d in infos or []:
            if isinstance(d, dict):
                merged.update(d)
        # env scalars (guarded)
        env_keys = [
            "pos_error", "alt_error", "stable_time", "max_stable_time",
            "accumulated_huber_error"
        ]
        for k in env_keys:
            if k in merged:
                self._safe_record_scalar(f"env/{k}", merged[k])

        # gains (either flat in info or nested under 'gains')   
        for k in self.log_gain_keys:
            self._safe_record_scalar(f"gains/{k}", merged[k])

        return True
