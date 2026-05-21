from __future__ import annotations

import csv
import logging
import os
import time
import pickle as pkl
from typing import Any

import numpy as np

from logicity.utils.joint_two_car_vec_env import CentralizedJointTwoCarEnv

logger = logging.getLogger(__name__)


def load_episode_data(path: str) -> dict[Any, dict[str, Any]]:
    with open(path, "rb") as f:
        return pkl.load(f)


def evaluate_centralized_joint_model(
    model,
    simulation_config: dict[str, Any],
    shield_config: dict[str, Any],
    episode_data: dict[Any, dict[str, Any]],
    log_progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_start = time.time()
    episode_rows: list[dict[str, Any]] = []
    rewards = []
    successes = []
    fail_episodes = 0
    timeout_episodes = 0
    deadzone_fail_events = 0
    simultaneous_entry_fail_events = 0
    rule_based_fail_events = 0

    for episode_id, episode_cache in episode_data.items():
        if log_progress:
            logger.info("Centralized validation evaluating episode %s...", episode_id)
        env = CentralizedJointTwoCarEnv(
            simulation_config=simulation_config,
            shield_config=shield_config,
            fixed_episode_cache=episode_cache,
        )
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        last_info: dict[str, Any] = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)
            done = bool(terminated or truncated)
            last_info = info
        env.close()

        joint_success = bool(last_info.get("joint_success", False))
        any_fail = bool(last_info.get("any_fail", False))
        overtime = bool(last_info.get("overtime", False))
        rule_fail = int(last_info.get("rule_based_fail_count", 0))
        deadzone_fail = int(last_info.get("deadzone_fail_count", 0))
        simultaneous_fail = int(last_info.get("simultaneous_entry_fail_count", 0))
        termination_reason = "success" if joint_success else ("overtime" if overtime else ("fail" if any_fail else "unknown"))

        rewards.append(episode_reward)
        successes.append(1 if joint_success else 0)
        fail_episodes += int(any_fail and not joint_success)
        timeout_episodes += int(overtime and not joint_success)
        rule_based_fail_events += rule_fail
        deadzone_fail_events += deadzone_fail
        simultaneous_entry_fail_events += simultaneous_fail

        episode_rows.append(
            {
                "row_type": "episode",
                "episode": episode_id,
                "joint_tsr": float(joint_success),
                "mean_reward": episode_reward,
                "termination_reason": termination_reason,
                "fail": int(any_fail and not joint_success),
                "timeout": int(overtime and not joint_success),
                "rule_based_fail_events": rule_fail,
                "deadzone_fail_events": deadzone_fail,
                "simultaneous_entry_fail_events": simultaneous_fail,
            }
        )
        if log_progress:
            logger.info(
                "Centralized validation episode %s done | joint_success=%s | reward=%.4f | reason=%s",
                episode_id,
                joint_success,
                episode_reward,
                termination_reason,
            )

    summary_row = {
        "row_type": "summary",
        "episode": "all",
        "joint_tsr": float(np.mean(successes)) if successes else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "fail": int(fail_episodes),
        "timeout": int(timeout_episodes),
        "rule_based_fail_events": int(rule_based_fail_events),
        "deadzone_fail_events": int(deadzone_fail_events),
        "simultaneous_entry_fail_events": int(simultaneous_entry_fail_events),
        "eval_wall_clock_seconds": float(time.time() - eval_start),
    }
    if hasattr(model, "get_joint_shield_metrics"):
        summary_row.update(model.get_joint_shield_metrics())
    return episode_rows, summary_row


def collect_centralized_joint_debug(
    model,
    simulation_config: dict[str, Any],
    shield_config: dict[str, Any],
    episode_cache: dict[str, Any],
    episode_id: Any,
    debug_steps: int = 3,
) -> list[dict[str, Any]]:
    env = CentralizedJointTwoCarEnv(
        simulation_config=simulation_config,
        shield_config=shield_config,
        fixed_episode_cache=episode_cache,
    )
    obs, _ = env.reset()
    rows: list[dict[str, Any]] = []
    done = False
    step = 0
    while (not done) and (step < debug_steps):
        snapshot = model.debug_action_snapshot(obs, env_source=env)
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(int(action))
        rows.append(
            {
                "episode": episode_id,
                "step": step,
                "chosen_joint_action": int(action),
                "reward": float(reward),
                "done": bool(terminated or truncated),
                "info": info,
                "snapshot": snapshot,
            }
        )
        obs = next_obs
        done = bool(terminated or truncated)
        step += 1
    env.close()
    return rows


def write_episode_csv(csv_path: str, episode_rows: list[dict[str, Any]]) -> None:
    if not episode_rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(episode_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in episode_rows:
            writer.writerow(row)


def write_summary_csv(csv_path: str, summary_row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(summary_row.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)
