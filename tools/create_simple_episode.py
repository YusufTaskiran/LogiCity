import os
import time
import yaml
import torch
import argparse
import importlib
import numpy as np
import pickle as pkl

from logicity.utils.load import CityLoader
from logicity.utils.logger import setup_logger
from logicity.utils.gym_wrapper import GymCityWrapper


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create simple cached episodes for eval/test.")
    parser.add_argument("--log_dir", type=str, default="./log_rl")
    parser.add_argument("--exp", type=str, default="simple_val")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=10)
    parser.add_argument("--require_stop", action="store_true", help="Only keep episodes where expert uses Stop at least once.")
    parser.add_argument(
        "--require_route_intersection",
        action="store_true",
        help="Only rollout episodes where the ego car route intersects at least one other agent route.",
    )
    parser.add_argument(
        "--require_all_pairwise_interactions",
        action="store_true",
        help="Only rollout episodes where every pair of agents has a route intersection and close arrival timing.",
    )
    parser.add_argument(
        "--max_arrival_gap",
        type=int,
        default=5,
        help="Maximum allowed trajectory-index gap at a shared route cell when requiring pairwise interactions.",
    )
    parser.add_argument(
        "--require_center_intersection_crossing",
        action="store_true",
        help="Only keep episodes where every agent starts on one side of the center intersection and ends on another side.",
    )
    parser.add_argument("--save_worlds", action="store_true", help="Also save per-episode world-cache pickles for visualization.")
    parser.add_argument(
        "--worlds_dir",
        type=str,
        default=None,
        help="Directory for per-episode world-cache pickles. Defaults to log_dir/<exp>_worlds.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional explicit output file path. Defaults to log_dir/<exp>_episodes.pkl.",
    )
    parser.add_argument(
        "--config",
        default="config/tasks/Nav/simple/experts/expert_episode_val.yaml",
        help="Config file for simple expert episode generation.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def make_env(simulation_config, return_cache=False):
    city, cached_observation = CityLoader.from_yaml(**simulation_config)
    env = GymCityWrapper(city)
    if return_cache:
        return env, cached_observation
    return env


def path_to_set(path):
    return {tuple(int(v) for v in point.tolist()) for point in path}


def routes_intersect(env):
    car_agent = env.agent
    other_agents = [agent for agent in env.env.agents if agent.layer_id != car_agent.layer_id]
    if not other_agents:
        return True
    car_path = path_to_set(car_agent.global_traj)
    for other_agent in other_agents:
        other_path = path_to_set(other_agent.global_traj)
        if car_path.intersection(other_path):
            return True
    return False


def first_arrival_index(path, shared_cell):
    for idx, point in enumerate(path):
        point_key = tuple(int(v) for v in point.tolist())
        if point_key == shared_cell:
            return idx
    return None


def pairwise_interaction_ok(agent_a, agent_b, max_arrival_gap):
    path_a = agent_a.global_traj
    path_b = agent_b.global_traj
    set_a = path_to_set(path_a)
    set_b = path_to_set(path_b)
    shared_cells = set_a.intersection(set_b)
    if len(shared_cells) == 0:
        return False
    best_gap = None
    for shared_cell in shared_cells:
        idx_a = first_arrival_index(path_a, shared_cell)
        idx_b = first_arrival_index(path_b, shared_cell)
        if idx_a is None or idx_b is None:
            continue
        gap = abs(idx_a - idx_b)
        if best_gap is None or gap < best_gap:
            best_gap = gap
    if best_gap is None:
        return False
    return best_gap <= max_arrival_gap


def all_pairwise_interactions_ok(env, max_arrival_gap):
    agents = list(env.env.agents)
    if len(agents) < 2:
        return True
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if not pairwise_interaction_ok(agents[i], agents[j], max_arrival_gap):
                return False
    return True


def _connected_components(mask):
    coords = torch.nonzero(mask)
    if coords.numel() == 0:
        return []
    coord_set = {tuple(int(v) for v in xy.tolist()) for xy in coords}
    components = []
    while coord_set:
        seed = coord_set.pop()
        stack = [seed]
        component = [seed]
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                neighbor = (nx, ny)
                if neighbor in coord_set:
                    coord_set.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    return components


def get_center_intersection_bounds(env):
    street_layer = env.env.city_grid[2]
    intersection_mask = street_layer == -1
    components = _connected_components(intersection_mask)
    if not components:
        return None
    grid_center = np.array([street_layer.shape[0] / 2.0, street_layer.shape[1] / 2.0], dtype=np.float32)
    best_bounds = None
    best_distance = None
    for component in components:
        rows = [c[0] for c in component]
        cols = [c[1] for c in component]
        bounds = (min(rows), max(rows), min(cols), max(cols))
        comp_center = np.array([(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0], dtype=np.float32)
        distance = float(np.linalg.norm(comp_center - grid_center))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_bounds = bounds
    return best_bounds


def classify_side(point, bounds):
    row, col = int(point[0]), int(point[1])
    min_r, max_r, min_c, max_c = bounds
    center_r = (min_r + max_r) / 2.0
    center_c = (min_c + max_c) / 2.0
    delta_r = row - center_r
    delta_c = col - center_c
    if abs(delta_r) >= abs(delta_c):
        return "up" if delta_r < 0 else "down"
    return "left" if delta_c < 0 else "right"


def path_crosses_bounds(path, bounds):
    min_r, max_r, min_c, max_c = bounds
    for point in path:
        row, col = int(point[0]), int(point[1])
        if min_r <= row <= max_r and min_c <= col <= max_c:
            return True
    return False


def all_agents_cross_center_intersection(env):
    bounds = get_center_intersection_bounds(env)
    if bounds is None:
        return False
    for agent in env.env.agents:
        if not path_crosses_bounds(agent.global_traj, bounds):
            return False
        start_side = classify_side(agent.start, bounds)
        goal_side = classify_side(agent.goal, bounds)
        if start_side == goal_side:
            return False
    return True


def side_candidates_for_agent(agent, bounds):
    start_points = []
    if getattr(agent, "start_point_list", None) is not None:
        start_points = [tuple(int(v) for v in pt) for pt in agent.start_point_list]
    goal_points = []
    desired_locations = getattr(agent, "desired_locations", None)
    if desired_locations is not None:
        goal_points = [tuple(int(v) for v in xy.tolist()) for xy in torch.nonzero(desired_locations)]

    start_side_points = {"left": [], "right": [], "up": [], "down": []}
    goal_side_points = {"left": [], "right": [], "up": [], "down": []}
    for pt in start_points:
        start_side_points[classify_side(pt, bounds)].append(pt)
    for pt in goal_points:
        goal_side_points[classify_side(pt, bounds)].append(pt)
    return start_side_points, goal_side_points


def recompute_agent_route(agent):
    agent.pos = agent.start.clone()
    if agent.type == "Car":
        agent.global_traj = agent.global_planner.plan(agent.start, agent.goal, 1)
    else:
        agent.global_traj = agent.global_planner(agent.movable_region, agent.start, agent.goal)
    agent.reach_goal = False
    agent.reach_goal_buffer = 0
    agent.last_move_dir = None
    return len(agent.global_traj) > 0


def assign_center_crossing_layout(eval_env, max_attempts=50):
    bounds = get_center_intersection_bounds(eval_env)
    if bounds is None:
        return False
    agents = list(eval_env.env.agents)
    for _ in range(max_attempts):
        used_points = set()
        success = True
        for agent in agents:
            start_side_points, goal_side_points = side_candidates_for_agent(agent, bounds)
            valid_sides = [
                side for side in start_side_points.keys()
                if len(start_side_points[side]) > 0 and len(goal_side_points[side]) > 0
            ]
            if len(valid_sides) < 2:
                success = False
                break
            assigned = False
            side_order = np.random.permutation(valid_sides).tolist()
            for start_side in side_order:
                goal_side_order = np.random.permutation([s for s in valid_sides if s != start_side]).tolist()
                start_candidates = [pt for pt in start_side_points[start_side] if pt not in used_points]
                if len(start_candidates) == 0:
                    continue
                np.random.shuffle(start_candidates)
                for goal_side in goal_side_order:
                    goal_candidates = [pt for pt in goal_side_points[goal_side] if pt not in used_points]
                    if len(goal_candidates) == 0:
                        continue
                    np.random.shuffle(goal_candidates)
                    for start_pt in start_candidates[:20]:
                        for goal_pt in goal_candidates[:20]:
                            agent.start = torch.tensor(start_pt)
                            agent.goal = torch.tensor(goal_pt)
                            if not recompute_agent_route(agent):
                                continue
                            if not path_crosses_bounds(agent.global_traj, bounds):
                                continue
                            if classify_side(agent.start, bounds) == classify_side(agent.goal, bounds):
                                continue
                            used_points.add(start_pt)
                            used_points.add(goal_pt)
                            assigned = True
                            break
                        if assigned:
                            break
                    if assigned:
                        break
                if assigned:
                    break
            if not assigned:
                success = False
                break
        if not success:
            continue
        for agent in agents:
            eval_env._redraw_agent_layer(agent)
        return True
    return False


def main(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    simulation_config = config["simulation"]
    rl_config = config["stable_baselines"]
    rl_agent_cfg = simulation_config.get("rl_agent", {})
    logger.info("Simulation config: %s", simulation_config)

    require_route_intersection = bool(args.require_route_intersection or rl_agent_cfg.get("require_route_intersection", False))
    require_all_pairwise_interactions = bool(
        args.require_all_pairwise_interactions or rl_agent_cfg.get("require_all_pairwise_interactions", False)
    )
    max_arrival_gap = int(args.max_arrival_gap if args.max_arrival_gap is not None else rl_agent_cfg.get("max_arrival_gap", 5))
    require_center_intersection_crossing = bool(
        args.require_center_intersection_crossing or rl_agent_cfg.get("require_center_intersection_crossing", False)
    )

    algorithm_class = dynamic_import("logicity.rl_agent.alg", rl_config["algorithm"])
    assert rl_config["algorithm"] == "ExpertCollector", "Simple episode generation expects ExpertCollector."

    all_episodes = {}
    rew_list = []
    success = []
    key = 0
    attempts = 0
    worlds_dir = args.worlds_dir or os.path.join(args.log_dir, f"{args.exp}_worlds")
    if args.save_worlds:
        os.makedirs(worlds_dir, exist_ok=True)

    while key < args.max_episodes:
        attempts += 1
        logger.info("Attempt %s | accepted %s/%s", attempts, key, args.max_episodes)
        eval_env, cached_observation = make_env(simulation_config, True)
        model = algorithm_class(eval_env)
        obs, episode_cache = eval_env.reset(True)
        if require_center_intersection_crossing:
            if not assign_center_crossing_layout(eval_env):
                logger.info("Discarding attempt %s because center-crossing assignment failed.", attempts)
                continue
            obs = eval_env.init()
            episode_cache = eval_env.save_episode()
        controlled_agent_name = f"{eval_env.agent.type}_{eval_env.agent_layer_id}"
        if controlled_agent_name in cached_observation["Static Info"]["Agents"]:
            cached_observation["Static Info"]["Agents"][controlled_agent_name]["concepts"] = eval_env.agent.concepts
        if require_route_intersection and not routes_intersect(eval_env):
            logger.info("Discarding attempt %s because ego route does not intersect another route.", attempts)
            continue
        if require_all_pairwise_interactions and not all_pairwise_interactions_ok(eval_env, max_arrival_gap):
            logger.info(
                "Discarding attempt %s because not all agent pairs have close route interactions (max_gap=%s).",
                attempts,
                max_arrival_gap,
            )
            continue
        if require_center_intersection_crossing and not all_agents_cross_center_intersection(eval_env):
            logger.info(
                "Discarding attempt %s because not all agents cross the center intersection from different sides.",
                attempts,
            )
            continue

        rew = 0
        step = 0
        done = False
        stop_used = False
        start_t = time.time()

        while not done:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            if int(action) == 1:
                stop_used = True
            obs, reward, done, info = eval_env.step(action)
            rew += reward
            cached_observation["Time_Obs"][step] = info

        if not info["is_success"]:
            logger.info("Discarding attempt %s because expert did not succeed.", attempts)
            continue
        if args.require_stop and not stop_used:
            logger.info("Discarding attempt %s because no Stop action was used.", attempts)
            continue

        label_info = {
            "action": 1 if stop_used else 0,
            "oracle_step": step,
            "stop_used": stop_used,
        }
        episode_cache["label_info"] = label_info
        all_episodes[key] = episode_cache
        rew_list.append(rew)
        success.append(1)
        if args.save_worlds:
            world_path = os.path.join(worlds_dir, f"{args.exp}_{key}.pkl")
            logger.info("Saving visualizable world pickle to %s", world_path)
            with open(world_path, "wb") as f:
                pkl.dump(cached_observation, f)
        logger.info("Episode %s took %s steps and %.3fs.", key, step, time.time() - start_t)
        logger.info("Episode %s score: %s", key, rew)
        logger.info("Episode %s label info: %s", key, label_info)
        logger.info("Accepted episodes: %s/%s after %s attempts", key + 1, args.max_episodes, attempts)
        key += 1

    logger.info("Generated %s episodes after %s attempts.", len(all_episodes), attempts)
    logger.info("Success rate: %s", np.mean(success) if success else 0.0)
    logger.info("Mean score achieved: %s", np.mean(rew_list) if rew_list else 0.0)

    output_path = args.output_path or os.path.join(args.log_dir, f"{args.exp}_episodes.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Saving episodes to %s", output_path)
    with open(output_path, "wb") as f:
        pkl.dump(all_episodes, f)


if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    logger.info("Collecting %s simple episodes from expert...", args.max_episodes)
    logger.info("Loading simulation config from %s.", args.config)
    main(args, logger)
