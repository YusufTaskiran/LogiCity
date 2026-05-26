import os
import copy
import time
import yaml
import torch
import argparse
import importlib
import inspect
import numpy as np
import pickle as pkl
from tqdm import trange
from logicity.core.config import *
from logicity.utils.load import CityLoader
from logicity.utils.logger import setup_logger
from logicity.utils.vis import visualize_city
# RL
from logicity.rl_agent.alg import *
from logicity.utils.gym_wrapper import GymCityWrapper
from logicity.utils.multi_agent_wrapper import SharedPolicyMultiAgentVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from logicity.utils.gym_callback import (
    ACTION_ID_TO_NAME,
    EvalCheckpointCallback,
    DreamerEvalCheckpointCallback,
    _load_task_rule_names,
    _sanitize_rule_name,
    append_eval_csv_row,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Logic-based city simulation.')
    # logger
    parser.add_argument('--log_dir', type=str, default="./log_rl")
    parser.add_argument('--checkpoint_root', type=str, default=None, help='Override the base checkpoint root used by eval callbacks.')
    parser.add_argument('--exp', type=str, default="maxsynth_debug")
    parser.add_argument('--vis', action='store_true', help='Visualize the city.')
    # seed
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=300)
    # RL
    parser.add_argument('--collect_only', action='store_true', help='Only collect expert data.')
    parser.add_argument('--collect_num_episodes', type=int, default=None, help='Override collecting_config.num_episodes for expert data collection.')
    parser.add_argument('--use_gym', action='store_true', help='In gym mode, we can use RL alg. to control certain agents.')
    parser.add_argument('--save_steps', action='store_true', help='Save step-wise decision for each trajectory.')
    parser.add_argument('--config', default='config/tasks/Nav/medium/algo/maxsynthtest.yaml', help='Configure file for this RL exp.')
    parser.add_argument('--checkpoint_path', default=None, help='Path to the trained model.')

    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def infer_difficulty_from_config_path(config_path):
    normalized = str(config_path).replace("\\", "/")
    for difficulty in ["easy", "medium", "hard"]:
        if f"/{difficulty}/" in normalized:
            return difficulty
    return "unknown"

def make_env(simulation_config, episode_cache=None, return_cache=False): 
    # Unpack arguments from simulation_config and pass them to CityLoader
    city, cached_observation = CityLoader.from_yaml(**simulation_config, episode_cache=episode_cache)
    if simulation_config.get("rl_agent", {}).get("agent_names"):
        env = SharedPolicyMultiAgentVecEnv(city)
    else:
        env = GymCityWrapper(city)
    if return_cache: 
        return env, cached_observation
    else:
        return env
    
def make_envs(simulation_config, rank):
    """
    Utility function for multiprocessed env.
    
    :param simulation_config: The configuration for the simulation.
    :param rank: Unique index for each environment to ensure different seeds.
    :return: A function that creates a single environment.
    """
    def _init():
        env = make_env(simulation_config)
        env.seed(rank + 1000)  # Optional: set a unique seed for each environment
        return env
    return _init

def main_collect(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    simulation_config = config["simulation"]
    logger.info("Simulation config: {}".format(simulation_config))
    collection_config = config['collecting_config']
    if args.collect_num_episodes is not None:
        collection_config = copy.deepcopy(collection_config)
        collection_config["num_episodes"] = args.collect_num_episodes
    logger.info("RL config: {}".format(collection_config))

    # Check if expert data collection is requested
    logger.info("Collecting expert demonstration data...")
    # Create an environment instance for collecting expert demonstrations
    expert_data_env, cached_observation = make_env(simulation_config, None, True)  # Use your existing environment setup function
    assert expert_data_env.use_expert  # Ensure the environment uses expert actions
    
    # Initialize the ExpertCollector with the environment and total timesteps
    collector = ExpertCollector(expert_data_env, **collection_config)
    _, full_world = collector.collect_data(cached_observation)
    
    # Save the collected expert demonstrations
    collector.save_data(f"{args.log_dir}/{args.exp}_expert_demonstrations.pkl")
    logger.info(f"Collected and saved expert demonstration data to {args.log_dir}/{args.exp}_expert_demonstrations.pkl")
    # Save the full world if needed
    if collection_config["return_full_world"]:
        for ts in range(len(full_world)):
            with open(os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, ts)), "wb") as f:
                pkl.dump(full_world[ts], f)

def main(args, logger):
    config = load_config(args.config)
    # simulation config
    simulation_config = config["simulation"]
    logger.info("Simulation config: {}".format(simulation_config))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Create a city instance with a predefined grid
    city, cached_observation = CityLoader.from_yaml(**simulation_config)
    visualize_city(city, 4*WORLD_SIZE, -1, "vis/init.png")
    # Main simulation loop
    steps = 0
    while steps < args.max_steps:
        logger.info("Simulating Step_{}...".format(steps))
        s = time.time()
        time_obs = city.update()
        e = time.time()
        logger.info("Time spent: {}".format(e-s))
        # Visualize the current state of the city (optional)
        if args.vis:
            visualize_city(city, 4*WORLD_SIZE, -1, "vis/step_{}.png".format(steps))
        steps += 1
        cached_observation["Time_Obs"][steps] = time_obs

    # Save the cached observation for better rendering
    with open(os.path.join(args.log_dir, "{}.pkl".format(args.exp)), "wb") as f:
        pkl.dump(cached_observation, f)

def main_gym(args, logger): 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    # simulation config
    simulation_config = config["simulation"]
    logger.info("Simulation config: {}".format(simulation_config))
    # RL config
    rl_config = config['stable_baselines']
    logger.info("RL config: {}".format(rl_config))
    # Dynamic import of the features extractor class
    if "features_extractor_module" in rl_config["policy_kwargs"]:
        features_extractor_class = dynamic_import(
            rl_config["policy_kwargs"]["features_extractor_module"],
            rl_config["policy_kwargs"]["features_extractor_class"]
        )
        # Prepare policy_kwargs with the dynamically imported class
        policy_kwargs = {
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": rl_config["policy_kwargs"]["features_extractor_kwargs"]
        }
    else:
        policy_kwargs = rl_config["policy_kwargs"]
    # Dynamic import of the RL algorithm
    algorithm_class = dynamic_import(
        "logicity.rl_agent.alg",  # Adjust the module path as needed
        rl_config["algorithm"]
    )
    # Load the entire eval_checkpoint configuration as a dictionary
    eval_checkpoint_config = config.get('eval_checkpoint')
    # Hyperparameters
    hyperparameters = rl_config["hyperparameters"]
    train = rl_config["train"]
    
    # model training
    if train: 
        num_envs = rl_config["num_envs"]
        total_timesteps = rl_config["total_timesteps"]
        if simulation_config.get("rl_agent", {}).get("agent_names"):
            if num_envs != 1:
                raise ValueError("Shared-policy multi-agent env already exposes multiple RL agents internally; set num_envs=1.")
            logger.info("Running in shared-policy multi-agent RL mode with {} RL agents.".format(len(simulation_config["rl_agent"]["agent_names"])))
            train_env = make_env(simulation_config)
        elif num_envs > 1:
            logger.info("Running in RL mode with {} parallel environments.".format(num_envs))
            train_env = SubprocVecEnv([make_envs(simulation_config, i) for i in range(num_envs)])
        else:
            train_env = make_env(simulation_config)
        train_env.reset()
        resuming_from_checkpoint = os.path.isfile(rl_config["checkpoint_path"])
        if resuming_from_checkpoint:
            logger.info("Resume training")
            logger.info("Loading the model from checkpoint: {}".format(rl_config["checkpoint_path"]))
            policy_kwargs_use = copy.deepcopy(policy_kwargs)
            model = algorithm_class.load(rl_config["checkpoint_path"], \
                                        train_env, **hyperparameters, policy_kwargs=policy_kwargs_use)
        else:
            model = algorithm_class(rl_config["policy_network"], \
                                    train_env, \
                                    **hyperparameters, \
                                    policy_kwargs=policy_kwargs)
        # RL training mode
        # Create the custom checkpoint and evaluation callback
        eval_checkpoint_callback = None
        if eval_checkpoint_config:
            eval_checkpoint_config = copy.deepcopy(eval_checkpoint_config)
            base_save_path = args.checkpoint_root if args.checkpoint_root is not None else eval_checkpoint_config.get("save_path", "./checkpoints/")
            eval_checkpoint_config["save_path"] = os.path.join(base_save_path, args.exp)
            eval_checkpoint_config["method"] = rl_config["algorithm"]
            eval_checkpoint_config["difficulty"] = infer_difficulty_from_config_path(args.config)
            eval_checkpoint_config["seed"] = args.seed
            eval_checkpoint_config["split"] = "val"
            if "Dreamer" == rl_config["algorithm"]:
                eval_checkpoint_callback = DreamerEvalCheckpointCallback(exp_name=args.exp, **eval_checkpoint_config)
            else:
                eval_checkpoint_callback = EvalCheckpointCallback(exp_name=args.exp, **eval_checkpoint_config)
        # Train the model
        learn_kwargs = {
            "total_timesteps": total_timesteps,
            "callback": eval_checkpoint_callback,
            "tb_log_name": args.exp,
        }
        if "reset_num_timesteps" in inspect.signature(model.learn).parameters:
            learn_kwargs["reset_num_timesteps"] = not resuming_from_checkpoint
        model.learn(**learn_kwargs)
        # Save the model
        save_name = args.exp if not eval_checkpoint_config else eval_checkpoint_config.get("name_prefix", args.exp)
        model.save(save_name)
        return
    # model evaluation
    else:
        assert os.path.isfile(rl_config["episode_data"])
        logger.info("Testing the trained model on episode data {}".format(rl_config["episode_data"]))
        assert "eval_actions" in rl_config
        logger.info("Evaluating the model with actions/id: {}".format(rl_config["eval_actions"]))
        # RL testing mode
        with open(rl_config["episode_data"], "rb") as f:
            episode_data = pkl.load(f)
        logger.info("Loaded episode data with {} episodes.".format(len(episode_data.keys())))
        # Checkpoint evaluation
        rew_list = []
        success = []
        failures = []
        timeouts = []
        episode_lengths = []
        decision_step = {}
        succ_decision = {}
        action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
        base_action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
        shielded_action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
        fail_rule_counts = {}
        fail_rl_agent_counts = {}
        mean_agent_tsr_list = []
        num_rl_agents_value = 1
        safe_prob_sums = np.zeros(4, dtype=np.float64)
        base_safe_prob_sum = 0.0
        shielded_safe_prob_sum = 0.0
        intervention_count = 0.0
        kl_sum = 0.0
        l1_sum = 0.0
        hazard_count = 0.0
        forced_stop_count = 0.0
        plpg_step_count = 0
        for action, id in rl_config["eval_actions"].items():
            decision_step[id] = 0
            succ_decision[id] = 0
        vis_id = [] if "vis_id" not in rl_config else rl_config["vis_id"]
        # over write the checkpoint path if not none
        if args.checkpoint_path is not None:
            rl_config["checkpoint_path"] = args.checkpoint_path
            logger.info("Overwrite the checkpoint path to {}".format(args.checkpoint_path))

        for ts in list(episode_data.keys()): 
            # if (ts not in vis_id) and len(vis_id) > 0:
            #     continue
            logger.info("Evaluating episode {}...".format(ts))
            episode_cache = episode_data[ts]
            max_steps = 10000
            if "label_info" in episode_cache:
                logger.info("Episode label: {}".format(episode_cache["label_info"]))
            if not args.save_steps:
                assert "oracle_step" in episode_cache["label_info"], "Need oracle step for evaluation."
                max_steps = episode_cache["label_info"]["oracle_step"] * 2
            eval_env, cached_observation = make_env(simulation_config, episode_cache, True)
            if rl_config["algorithm"] == "ExpertCollector" or rl_config["algorithm"] == "Random":
                # expert and random agent do not need a policy network
                model = algorithm_class(eval_env)
            elif rl_config["algorithm"] == "MaxSynth":
                model = algorithm_class(eval_env, **policy_kwargs)
            elif rl_config["algorithm"] in ["HRI", "NLM"]:
                # HRI and NLM are trained w/ ext code, just load the network
                model = algorithm_class(rl_config["policy_network"], \
                                        eval_env, \
                                        **hyperparameters, \
                                        policy_kwargs=policy_kwargs)
                model.load(rl_config["checkpoint_path"])
            else:
                # SB3-based agents
                policy_kwargs_use = copy.deepcopy(policy_kwargs)
                if rl_config["algorithm"] == 'A2C':
                    model = algorithm_class.load(rl_config["checkpoint_path"], \
                                    eval_env, **hyperparameters)
                else:
                    model = algorithm_class.load(rl_config["checkpoint_path"], \
                                    eval_env, **hyperparameters, policy_kwargs=policy_kwargs_use)
            logger.info("Loaded model from {}".format(rl_config["checkpoint_path"]))
            o = eval_env.init()
            is_multi_agent = getattr(eval_env, "num_envs", 1) > 1 and hasattr(eval_env, "rl_agents")
            rew = 0    
            step = 0   
            local_decision_step = {}
            local_succ_decision = {}
            for acc, id in rl_config["eval_actions"].items():
                local_decision_step[id] = 0
                local_succ_decision[id] = 1
            d = False
            if rl_config["algorithm"] == 'Dreamer':
                prev_rssmstate = model.policy.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, model.action_size).to(model.device)
                while (not d) and (step < max_steps):
                    step += 1
                    oracle_action = eval_env.expert_action
                    with torch.no_grad():
                        embed = model.policy.ObsEncoder(torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(model.device))    
                        _, posterior_rssm_state = model.policy.RSSM.rssm_observe(embed, prev_action, not d, prev_rssmstate)
                        model_state = model.policy.RSSM.get_model_state(posterior_rssm_state)
                        action, _ = model.policy.ActionModel(model_state)
                        prev_rssmstate = posterior_rssm_state
                        prev_action = action
                    env_action = torch.argmax(action, dim=-1).cpu().numpy()
                    action_array = np.atleast_1d(env_action)
                    for action_item in action_array:
                        action_hist[int(action_item)] = action_hist.get(int(action_item), 0) + 1
                    if oracle_action in local_decision_step.keys():
                        local_decision_step[oracle_action] = 1
                        if int(np.atleast_1d(env_action)[0]) != oracle_action:
                            local_succ_decision[oracle_action] = 0
                    o, r, d, i = eval_env.step(env_action)
                    if is_multi_agent:
                        reward_value = float(np.mean(r))
                        joint_info = i[0]
                        num_rl_agents_value = int(joint_info.get("num_rl_agents", num_rl_agents_value))
                        if joint_info.get("joint_failure", False):
                            rew += reward_value
                            break
                        rew += reward_value
                    else:
                        if i["Fail"][0]:
                            rew += r
                            break
                        rew += r
            else:
                while (not d) and (step < max_steps):
                    step += 1
                    oracle_action = eval_env.expert_action
                    plpg_info = None
                    if hasattr(model, "predict_with_plpg_info"):
                        action, _, plpg_info = model.predict_with_plpg_info(o, deterministic=True)
                    else:
                        action, _ = model.predict(o, deterministic=True)
                    action_array = np.atleast_1d(action)
                    for action_item in action_array:
                        action_hist[int(action_item)] = action_hist.get(int(action_item), 0) + 1
                    if plpg_info is not None:
                        base_actions = np.atleast_1d(plpg_info["base_action"])
                        shielded_actions = np.atleast_1d(plpg_info["shielded_action"])
                        for base_action in base_actions:
                            base_action_hist[int(base_action)] += 1
                        for shielded_action in shielded_actions:
                            shielded_action_hist[int(shielded_action)] += 1
                        safe_prob_sums += np.asarray(plpg_info["safety_probs"], dtype=np.float64).sum(axis=0)
                        base_safe_prob_sum += float(np.asarray(plpg_info["base_policy_safe_prob"], dtype=np.float64).sum())
                        shielded_safe_prob_sum += float(np.asarray(plpg_info["shielded_policy_safe_prob"], dtype=np.float64).sum())
                        intervention_count += float(np.asarray(plpg_info["intervened"], dtype=np.float64).sum())
                        kl_sum += float(np.asarray(plpg_info["kl_base_to_shielded"], dtype=np.float64).sum())
                        l1_sum += float(np.asarray(plpg_info["l1_shift_base_to_shielded"], dtype=np.float64).sum())
                        hazard_count += float(np.asarray(plpg_info["hazard"], dtype=np.float64).sum())
                        forced_stop_count += float(np.asarray(plpg_info["forced_stop"], dtype=np.float64).sum())
                        plpg_step_count += int(np.asarray(plpg_info["base_policy_safe_prob"]).shape[0]) if np.asarray(plpg_info["base_policy_safe_prob"]).ndim > 0 else 1
                    # save step_wise decision succ per trajectory
                    if oracle_action in local_decision_step.keys():
                        local_decision_step[oracle_action] = 1
                        if int(np.atleast_1d(action)[0]) != oracle_action:
                            local_succ_decision[oracle_action] = 0
                    o, r, d, i = eval_env.step(action)
                    if (ts in vis_id) or (-1 in vis_id):
                        cached_observation["Time_Obs"][step] = i
                    if is_multi_agent:
                        reward_value = float(np.mean(r))
                        joint_info = i[0]
                        num_rl_agents_value = int(joint_info.get("num_rl_agents", num_rl_agents_value))
                        if joint_info.get("joint_failure", False):
                            rew += reward_value
                            break
                        rew += reward_value
                    else:
                        if i["Fail"][0]:
                            rew += r
                            break
                        rew += r
            if is_multi_agent:
                joint_info = i[0]
                if joint_info.get("joint_is_success", False):
                    success.append(1)
                    failures.append(0)
                    timeouts.append(1 if joint_info.get("joint_timeout", False) else 0)
                else:
                    success.append(0)
                    failures.append(1 if joint_info.get("joint_failure", False) else 0)
                    timeouts.append(1 if joint_info.get("joint_timeout", False) else 0)
                mean_agent_tsr_list.append(float(joint_info.get("mean_agent_success", 0.0)))
                for layer_id in joint_info.get("joint_fail_agent_layer_ids", []):
                    key = "fail_rl_agent_{}".format(layer_id)
                    fail_rl_agent_counts[key] = fail_rl_agent_counts.get(key, 0) + 1
            elif i["is_success"]:
                success.append(1)
                failures.append(0)
                timeouts.append(1 if i.get("overtime", False) else 0)
            else:
                success.append(0)
                failures.append(1 if i["Fail"][0] else 0)
                timeouts.append(1 if i.get("overtime", False) else 0)
                for rule_name in i.get("FailRuleNames", [[]])[0]:
                    fail_rule_counts[rule_name] = fail_rule_counts.get(rule_name, 0) + 1
            for acc, id in rl_config["eval_actions"].items():
                if local_decision_step[id] == 0:
                    local_succ_decision[id] = 0
                decision_step[id] += local_decision_step[id]
                succ_decision[id] += local_succ_decision[id]
            if step >= max_steps:
                rew -= 3
                timeouts[-1] = 1
            episode_lengths.append(step)
            rew_list.append(rew)
            if args.save_steps:
                episode_cache["label_info"]['oracle_step'] = step
            logger.info("Episode {} took {} steps.".format(ts, step))
            logger.info("Episode {} achieved a score of {}".format(ts, rew))
            logger.info("Episode {} Success: {}".format(ts, success[-1]))
            logger.info("Episode {} Decision Step: {}".format(ts, local_decision_step))
            logger.info("Episode {} Success Decision: {}".format(ts, local_succ_decision))
            if (ts in vis_id) or (-1 in vis_id):
                # worlds[ts] = cached_observation
                with open(os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, ts)), "wb") as f:
                    pkl.dump(cached_observation, f)
        mean_reward = np.mean(rew_list)
        np.save(os.path.join(args.log_dir, "{}_rewards.npy".format(args.exp)), rew_list)
        sr = np.mean(success)
        failure_rate = np.mean(failures)
        timeout_rate = np.mean(timeouts)
        mean_length = np.mean(episode_lengths)
        mSuccD, aSuccD, SuccDAct = cal_step_metric(decision_step, succ_decision)
        mean_agent_tsr = np.mean(mean_agent_tsr_list) if len(mean_agent_tsr_list) > 0 else sr
        logger.info("Mean Score achieved: {}".format(mean_reward))
        logger.info("Success Rate: {}".format(sr))
        logger.info("Failure Rate: {} - Timeout Rate: {} - Mean Episode Length: {}".format(failure_rate, timeout_rate, mean_length))
        logger.info("Mean Decision Succ: {}".format(mSuccD))
        logger.info("Average Decision Succ: {}".format(aSuccD))
        logger.info("Decision Succ for each action: {}".format(SuccDAct))
        logger.info("Action Histogram: {}".format(action_hist))
        logger.info("Failure Rule Counts: {}".format(fail_rule_counts))
        if plpg_step_count > 0:
            logger.info(
                "PLPG Metrics: base_safe={} shielded_safe={} gain={} intervention_rate={} kl={} l1={} hazard_rate={} forced_stop_rate={}".format(
                    base_safe_prob_sum / plpg_step_count,
                    shielded_safe_prob_sum / plpg_step_count,
                    (shielded_safe_prob_sum - base_safe_prob_sum) / plpg_step_count,
                    intervention_count / plpg_step_count,
                    kl_sum / plpg_step_count,
                    l1_sum / plpg_step_count,
                    hazard_count / plpg_step_count,
                    forced_stop_count / max(hazard_count, 1.0),
                )
            )
        failure_rule_names = _load_task_rule_names(simulation_config["rule_yaml_file"])
        csv_row = {
            "exp_name": args.exp,
            "method": rl_config["algorithm"],
            "difficulty": infer_difficulty_from_config_path(args.config),
            "seed": args.seed,
            "split": "test",
            "eval_index": 1,
            "timestep": 0,
            "wall_clock_time_sec": 0.0,
            "best_tsr_so_far": sr,
            "num_eval_episodes": len(rew_list),
            "tsr": sr,
            "mean_reward": mean_reward,
            "failure_rate": failure_rate,
            "timeout_rate": timeout_rate,
            "mean_episode_length": mean_length,
            "num_rl_agents": num_rl_agents_value,
            "mean_agent_tsr": mean_agent_tsr,
            "mean_decision_succ": mSuccD,
            "rollout_failures_since_last_eval": 0,
            "rollout_timeouts_since_last_eval": 0,
            "rollout_successes_since_last_eval": 0,
            "mean_base_policy_safe_prob": (base_safe_prob_sum / plpg_step_count) if plpg_step_count > 0 else 0.0,
            "mean_shielded_policy_safe_prob": (shielded_safe_prob_sum / plpg_step_count) if plpg_step_count > 0 else 0.0,
            "mean_safety_gain": ((shielded_safe_prob_sum - base_safe_prob_sum) / plpg_step_count) if plpg_step_count > 0 else 0.0,
            "shield_intervention_rate": (intervention_count / plpg_step_count) if plpg_step_count > 0 else 0.0,
            "mean_policy_kl_base_to_shielded": (kl_sum / plpg_step_count) if plpg_step_count > 0 else 0.0,
            "mean_l1_shift_base_to_shielded": (l1_sum / plpg_step_count) if plpg_step_count > 0 else 0.0,
            "hazard_step_rate": (hazard_count / plpg_step_count) if plpg_step_count > 0 else 0.0,
            "shield_forced_stop_rate": (forced_stop_count / max(hazard_count, 1.0)) if plpg_step_count > 0 else 0.0,
        }
        for action_id, action_name in ACTION_ID_TO_NAME.items():
            csv_row[f"action_{action_name}_count"] = action_hist.get(action_id, 0)
            csv_row[f"base_action_{action_name}_count"] = base_action_hist.get(action_id, 0)
            csv_row[f"shielded_action_{action_name}_count"] = shielded_action_hist.get(action_id, 0)
            csv_row[f"mean_safe_prob_action_{action_name}"] = (
                safe_prob_sums[action_id] / plpg_step_count
            ) if plpg_step_count > 0 else 0.0
        for action_id, value in SuccDAct.items():
            action_name = ACTION_ID_TO_NAME.get(action_id, str(action_id))
            csv_row["decision_succ_action_{}".format(action_name)] = value
        for key, value in fail_rl_agent_counts.items():
            csv_row[key] = value
        for rule_name in failure_rule_names:
            csv_row["fail_rule_{}".format(_sanitize_rule_name(rule_name))] = fail_rule_counts.get(rule_name, 0)
        append_eval_csv_row(os.path.join(args.log_dir, "{}_test_metrics.csv".format(args.exp)), csv_row, failure_rule_names)
        if args.save_steps:
            with open(os.path.join(args.log_dir, "{}_steps.pkl".format(args.exp)), "wb") as f:
                pkl.dump(episode_data, f)
        # for ts in worlds.keys():
        #     if worlds[ts] is not None:
        #         with open(os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, ts)), "wb") as f:
        #             pkl.dump(worlds[ts], f)


def cal_step_metric(decision_step, succ_decision):
    mean_decision_succ = {}
    total_decision = sum(decision_step.values())
    total_decision = max(total_decision, 1)
    total_succ = sum(succ_decision.values())
    for action, num in decision_step.items():
        num = max(num, 1)
        mean_decision_succ[action] = succ_decision[action]/num
    average_decision_succ = sum(mean_decision_succ.values())/len(mean_decision_succ)
    # mean decision succ (over all steps), average decision succ (over all actions), decision succ for each action
    return total_succ/total_decision, average_decision_succ, mean_decision_succ

if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    if args.collect_only:
        logger.info("Running in data collection mode.")
        logger.info("Loading simulation config from {}.".format(args.config))
        main_collect(args, logger)
    elif args.use_gym:
        logger.info("Running in RL mode.")
        logger.info("Loading RL config from {}.".format(args.config))
        # RL mode, will use gym wrapper to learn and test an agent
        main_gym(args, logger)
    else:
        # Sim mode, will use the logic-based simulator to run a simulation (no learning)
        logger.info("Running in simulation mode.")
        logger.info("Loading simulation config from {}.".format(args.config))
        e = time.time()
        main(args, logger)
        logger.info("Total time spent: {}".format(time.time()-e))
