from __future__ import annotations

import csv
import json
import logging
import os
import time
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback

from logicity.utils.centralized_joint_eval import (
    collect_centralized_joint_debug,
    evaluate_centralized_joint_model,
    load_episode_data,
)

logger = logging.getLogger(__name__)


def _jsonify_debug_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify_debug_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_debug_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify_debug_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class CentralizedJointEvalCheckpointCallback(CheckpointCallback):
    def __init__(
        self,
        exp_name: str,
        simulation_config: dict[str, Any],
        shield_config: dict[str, Any],
        validation_episode_data: str,
        eval_freq: int = 500,
        result_path: str | None = None,
        debug_shield: bool = False,
        debug_steps: int = 3,
        debug_episode: str | int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.exp_name = exp_name
        self.simulation_config = simulation_config
        self.shield_config = shield_config
        self.validation_episode_data = load_episode_data(validation_episode_data)
        self.eval_freq = int(eval_freq)
        self.best_joint_tsr = float("-inf")
        self.best_mean_reward = float("-inf")
        self.start_time = time.time()
        self.result_path = result_path or self.save_path
        self.csv_path = os.path.join(self.result_path, f"{self.exp_name}_metrics.csv")
        self.debug_shield = bool(debug_shield)
        self.debug_steps = int(debug_steps)
        self.debug_episode = debug_episode

    def _append_metrics_csv(self, row: dict[str, Any]) -> None:
        os.makedirs(self.result_path, exist_ok=True)
        write_header = not os.path.exists(self.csv_path)
        fieldnames = list(row.keys())
        with open(self.csv_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            logger.info("Saved centralized checkpoint to %s", model_path)

        if self.n_calls % self.eval_freq != 0:
            return True

        logger.info(
            "Starting centralized validation at timestep=%s on %s episodes",
            self.num_timesteps,
            len(self.validation_episode_data),
        )
        if hasattr(self.model, "reset_joint_shield_metrics"):
            self.model.reset_joint_shield_metrics()

        _, summary_row = evaluate_centralized_joint_model(
            model=self.model,
            simulation_config=self.simulation_config,
            shield_config=self.shield_config,
            episode_data=self.validation_episode_data,
            log_progress=True,
        )
        csv_row = {
            "step": int(self.num_timesteps),
            "joint_tsr": float(summary_row.get("joint_tsr", 0.0)),
            "mean_reward": float(summary_row.get("mean_reward", 0.0)),
            "fail": int(summary_row.get("fail", 0)),
            "timeout": int(summary_row.get("timeout", 0)),
            "rule_based_fail_events": int(summary_row.get("rule_based_fail_events", 0)),
            "deadzone_fail_events": int(summary_row.get("deadzone_fail_events", 0)),
            "simultaneous_entry_fail_events": int(summary_row.get("simultaneous_entry_fail_events", 0)),
            "elapsed_wall_clock_seconds": float(time.time() - self.start_time),
            "seconds_per_1k_timesteps": float((time.time() - self.start_time) / max(self.num_timesteps, 1) * 1000.0),
        }
        for key, value in summary_row.items():
            if key not in csv_row:
                csv_row[key] = value
        self._append_metrics_csv(csv_row)
        logger.info(
            "Centralized validation done at timestep=%s | joint_tsr=%.4f | mean_reward=%.4f | fail=%s | timeout=%s",
            self.num_timesteps,
            float(summary_row.get("joint_tsr", 0.0)),
            float(summary_row.get("mean_reward", 0.0)),
            int(summary_row.get("fail", 0)),
            int(summary_row.get("timeout", 0)),
        )

        if self.debug_shield and len(self.validation_episode_data) > 0:
            debug_episode = self.debug_episode
            if debug_episode is None:
                debug_episode = next(iter(self.validation_episode_data.keys()))
            elif debug_episode not in self.validation_episode_data:
                try:
                    numeric_debug_episode = int(debug_episode)
                except (TypeError, ValueError):
                    numeric_debug_episode = None
                if numeric_debug_episode in self.validation_episode_data:
                    debug_episode = numeric_debug_episode
            episode_cache = self.validation_episode_data[debug_episode]
            debug_rows = collect_centralized_joint_debug(
                model=self.model,
                simulation_config=self.simulation_config,
                shield_config=self.shield_config,
                episode_cache=episode_cache,
                episode_id=debug_episode,
                debug_steps=self.debug_steps,
            )
            debug_path = os.path.join(self.result_path, f"{self.exp_name}_shield_debug_step_{int(self.num_timesteps)}.json")
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(_jsonify_debug_value(debug_rows), f, indent=2)
            logger.info(
                "Wrote centralized shield debug dump for episode %s to %s",
                debug_episode,
                debug_path,
            )

        current_tsr = float(summary_row.get("joint_tsr", 0.0))
        current_reward = float(summary_row.get("mean_reward", 0.0))
        is_better = (current_tsr > self.best_joint_tsr) or (
            current_tsr == self.best_joint_tsr and current_reward > self.best_mean_reward
        )
        if is_better:
            self.best_joint_tsr = current_tsr
            self.best_mean_reward = current_reward
            self.model.save(os.path.join(self.result_path, "best_model"))
        return True
