import itertools
from typing import Dict, Iterable, List, Optional
from types import SimpleNamespace

import numpy as np
import torch as th

from .templates import build_template
from ..utils.pred_converter.z3 import CollidingClose, HigherPri, IsAtInter, IsInInter, IsParked, SameInter


class CentralizedJointTrafficShield:
    def __init__(
        self,
        template_name,
        pred_grounding_index,
        rule_yaml_file=None,
        sensor_noise=None,
        num_agents: Optional[int] = None,
    ):
        self.template = build_template(
            template_name,
            pred_grounding_index,
            rule_yaml_file=rule_yaml_file,
            sensor_noise=sensor_noise,
        )
        self.action_names = list(self.template.action_names)
        self.num_agents = num_agents
        self._joint_actions_cache: Dict[int, List[tuple[int, ...]]] = {}
        self._imminent_tti_threshold = 8
        self._critical_tti_threshold = 4

    def _joint_actions(self, num_agents: int) -> List[tuple[int, ...]]:
        if num_agents not in self._joint_actions_cache:
            self._joint_actions_cache[num_agents] = list(
                itertools.product(range(len(self.action_names)), repeat=num_agents)
            )
        return self._joint_actions_cache[num_agents]

    def _format_joint_action(self, joint_action: Iterable[int]) -> List[str]:
        return [self.action_names[int(action_id)] for action_id in joint_action]

    def _normalize_scene_graph(self, scene_graph_by_layer: Optional[Dict]) -> Dict[int, Dict]:
        normalized = {}
        if not scene_graph_by_layer:
            return normalized
        for layer_id, scene_graph in scene_graph_by_layer.items():
            try:
                normalized[int(layer_id)] = scene_graph or {}
            except (TypeError, ValueError):
                continue
        return normalized

    def _object_data(self, scene_graph: Dict, object_layer_id: int) -> Dict:
        return (
            scene_graph.get("objects", {}).get(str(int(object_layer_id)))
            or scene_graph.get("objects", {}).get(int(object_layer_id))
            or {}
        )

    def _relation_true(
        self,
        scene_graph_by_layer: Dict[int, Dict],
        source_layer_id: int,
        target_layer_id: int,
        relation_name: str,
    ) -> float:
        target_str = str(int(target_layer_id))
        for scene_graph in scene_graph_by_layer.values():
            object_data = self._object_data(scene_graph, source_layer_id)
            for relation in object_data.get("relations", []):
                if relation.get("name") == relation_name and str(relation.get("object")) == target_str:
                    return 1.0
        return 0.0

    def _speed_factor(self, action_id: int) -> float:
        if action_id == 3:
            return 0.0
        if action_id == 0:
            return 0.4
        if action_id == 1:
            return 0.7
        return 1.0

    def _local_action_safety(self, obs_row: th.Tensor):
        fact_weights = self.template.obs_to_fact_weights(obs_row)
        local_scores = self.template.postprocess_action_scores(
            [1.0] * len(self.action_names),
            fact_weights,
            obs_row=obs_row,
        )
        local_metrics = self.template.fact_weights_to_metrics(fact_weights)
        return fact_weights, np.asarray(local_scores, dtype=np.float32), local_metrics

    def _should_defer_local_rl_intersection_logic(
        self,
        agent_idx: int,
        pair_features: Dict[tuple[int, int], Dict[str, float]],
        local_metrics: Dict[str, float],
        num_agents: int,
    ) -> bool:
        # Keep local control for pedestrian conflicts and collision-close cues.
        if float(local_metrics.get("colliding_close", 0.0)) > 0.05:
            return False
        if float(local_metrics.get("ped_conflict", 0.0)) > 0.05:
            return False

        has_same_inter_rl = False
        for other_idx in range(num_agents):
            if other_idx == agent_idx:
                continue
            pair_key = (min(agent_idx, other_idx), max(agent_idx, other_idx))
            features = pair_features.get(pair_key)
            if not features:
                continue
            if float(features.get("same_inter", 0.0)) <= 0.0:
                continue
            has_same_inter_rl = True

        return has_same_inter_rl

    def _agent_intersection_context_id(self, agent_obj, intersection_matrix) -> int:
        if agent_obj is None or intersection_matrix is None:
            return 0
        pos = getattr(agent_obj, "pos", None)
        traj = getattr(agent_obj, "global_traj", None)
        if pos is None:
            return 0
        pos = pos.to(dtype=th.long)
        current_inter_id = int(intersection_matrix[2, pos[0], pos[1]].item())
        if current_inter_id > 0:
            return current_inter_id

        fallback_line_inter_id = int(intersection_matrix[0, pos[0], pos[1]].item())
        if traj is None or len(traj) == 0:
            return max(fallback_line_inter_id, 0)

        matches = th.all(traj == pos, dim=1).nonzero(as_tuple=False)
        if matches.numel() == 0:
            return max(fallback_line_inter_id, 0)

        current_idx = int(matches[0].item())
        h, w = intersection_matrix.shape[1], intersection_matrix.shape[2]

        for idx in range(current_idx + 1, len(traj)):
            point = traj[idx]
            if point[0] < 0 or point[0] >= h or point[1] < 0 or point[1] >= w:
                continue
            inter_id = int(intersection_matrix[2, point[0], point[1]].item())
            if inter_id > 0:
                return inter_id

        lookbehind_start = max(0, current_idx - 3)
        for idx in range(current_idx - 1, lookbehind_start - 1, -1):
            point = traj[idx]
            if point[0] < 0 or point[0] >= h or point[1] < 0 or point[1] >= w:
                continue
            inter_id = int(intersection_matrix[2, point[0], point[1]].item())
            if inter_id > 0:
                return inter_id

        return max(fallback_line_inter_id, 0)

    def _time_to_next_intersection(self, agent_obj, intersection_matrix) -> Optional[int]:
        if agent_obj is None or intersection_matrix is None:
            return None
        pos = getattr(agent_obj, "pos", None)
        traj = getattr(agent_obj, "global_traj", None)
        if pos is None or traj is None or len(traj) == 0:
            return None
        pos = pos.to(dtype=th.long)
        if int(intersection_matrix[2, pos[0], pos[1]].item()) > 0:
            return 0
        matches = th.all(traj == pos, dim=1).nonzero(as_tuple=False)
        if matches.numel() == 0:
            return None
        current_idx = int(matches[0].item())
        h, w = intersection_matrix.shape[1], intersection_matrix.shape[2]
        for idx in range(current_idx + 1, len(traj)):
            point = traj[idx]
            if point[0] < 0 or point[0] >= h or point[1] < 0 or point[1] >= w:
                continue
            if int(intersection_matrix[2, point[0], point[1]].item()) > 0:
                return idx - current_idx
        return None

    def _active_intersection_context_id(
        self,
        agent_obj,
        intersection_matrix,
        is_at_intersection: float,
        is_in_intersection: float,
        max_gate_lookahead_steps: int = 3,
    ) -> int:
        if agent_obj is None or intersection_matrix is None:
            return 0

        pos = getattr(agent_obj, "pos", None)
        traj = getattr(agent_obj, "global_traj", None)
        if pos is None:
            return 0

        pos = pos.to(dtype=th.long)
        current_inter_id = int(intersection_matrix[2, pos[0], pos[1]].item())
        if is_in_intersection > 0.0 and current_inter_id > 0:
            return current_inter_id

        if is_at_intersection <= 0.0:
            return 0

        if traj is None or len(traj) == 0:
            return 0

        matches = th.all(traj == pos, dim=1).nonzero(as_tuple=False)
        if matches.numel() == 0:
            return 0

        current_idx = int(matches[0].item())
        h, w = intersection_matrix.shape[1], intersection_matrix.shape[2]
        max_idx = min(len(traj), current_idx + 1 + max_gate_lookahead_steps)
        for idx in range(current_idx, max_idx):
            point = traj[idx]
            if point[0] < 0 or point[0] >= h or point[1] < 0 or point[1] >= w:
                continue
            inter_id = int(intersection_matrix[2, point[0], point[1]].item())
            if inter_id > 0:
                return inter_id

        return 0

    def _future_path_conflict(self, src_agent_obj, dst_agent_obj, horizon_steps: int = 8) -> float:
        if src_agent_obj is None or dst_agent_obj is None:
            return 0.0
        src_pos = getattr(src_agent_obj, "pos", None)
        dst_pos = getattr(dst_agent_obj, "pos", None)
        src_traj = getattr(src_agent_obj, "global_traj", None)
        dst_traj = getattr(dst_agent_obj, "global_traj", None)
        if src_pos is None or dst_pos is None or src_traj is None or dst_traj is None:
            return 0.0

        src_matches = th.all(src_traj == src_pos, dim=1).nonzero(as_tuple=False)
        dst_matches = th.all(dst_traj == dst_pos, dim=1).nonzero(as_tuple=False)
        if src_matches.numel() == 0 or dst_matches.numel() == 0:
            return 0.0

        src_idx = int(src_matches[0].item())
        dst_idx = int(dst_matches[0].item())
        src_future = [tuple(src_traj[min(src_idx + step, len(src_traj) - 1)].tolist()) for step in range(horizon_steps + 1)]
        dst_future = [tuple(dst_traj[min(dst_idx + step, len(dst_traj) - 1)].tolist()) for step in range(horizon_steps + 1)]

        for step in range(min(len(src_future), len(dst_future))):
            if src_future[step] == dst_future[step]:
                return 1.0
        for step in range(min(len(src_future), len(dst_future)) - 1):
            if src_future[step + 1] == dst_future[step] and dst_future[step + 1] == src_future[step]:
                return 1.0
        return 0.0

    def _same_lane_follow_conflict(self, src_agent_obj, dst_agent_obj) -> tuple[float, int]:
        if src_agent_obj is None or dst_agent_obj is None:
            return 0.0, -1

        src_pos = getattr(src_agent_obj, "pos", None)
        dst_pos = getattr(dst_agent_obj, "pos", None)
        src_traj = getattr(src_agent_obj, "global_traj", None)
        dst_traj = getattr(dst_agent_obj, "global_traj", None)
        if src_pos is None or dst_pos is None or src_traj is None or dst_traj is None:
            return 0.0, -1

        src_matches = th.all(src_traj == src_pos, dim=1).nonzero(as_tuple=False)
        dst_matches = th.all(dst_traj == dst_pos, dim=1).nonzero(as_tuple=False)
        if src_matches.numel() == 0 or dst_matches.numel() == 0:
            return 0.0, -1

        src_idx = int(src_matches[0].item())
        dst_idx = int(dst_matches[0].item())
        if src_idx >= len(src_traj) - 1 or dst_idx >= len(dst_traj) - 1:
            return 0.0, -1

        src_dir = (src_traj[src_idx + 1] - src_traj[src_idx]).to(dtype=th.int64)
        dst_dir = (dst_traj[dst_idx + 1] - dst_traj[dst_idx]).to(dtype=th.int64)
        if not th.equal(src_dir, dst_dir):
            return 0.0, -1

        rel = (dst_pos - src_pos).to(dtype=th.int64)
        lateral = int(abs(rel[0].item() * src_dir[1].item() - rel[1].item() * src_dir[0].item()))
        if lateral != 0:
            return 0.0, -1

        forward_gap = int(rel[0].item() * src_dir[0].item() + rel[1].item() * src_dir[1].item())
        if forward_gap == 0:
            return 1.0, -1

        # Positive means dst is ahead of src. Negative means src is ahead of dst.
        abs_gap = abs(forward_gap)
        if abs_gap > 12:
            return 0.0, -1

        if abs_gap <= 4:
            score = 1.0
        elif abs_gap <= 8:
            score = 0.8
        else:
            score = 0.6

        rear_agent = 0 if forward_gap > 0 else 1
        return score, rear_agent

    def _spawn_release_conflict(self, src_agent_obj, dst_agent_obj, horizon_steps: int = 3) -> float:
        if src_agent_obj is None or dst_agent_obj is None:
            return 0.0

        src_pos = getattr(src_agent_obj, "pos", None)
        dst_pos = getattr(dst_agent_obj, "pos", None)
        if src_pos is None or dst_pos is None:
            return 0.0

        src_xy = tuple(int(v) for v in src_pos.tolist())
        dst_xy = tuple(int(v) for v in dst_pos.tolist())
        manhattan = abs(src_xy[0] - dst_xy[0]) + abs(src_xy[1] - dst_xy[1])

        # Two parked RL cars that spawn essentially bumper-to-bumper should not
        # be released together, even before ordinary collision predicates fire.
        if manhattan == 0:
            return 1.0
        if manhattan == 1:
            return 0.95
        if manhattan == 2:
            return 0.75
        if manhattan == 3:
            return 0.45

        return self._future_path_conflict(src_agent_obj, dst_agent_obj, horizon_steps=horizon_steps)

    def _spawn_release_buffer_conflict(self, parked_agent_obj, released_agent_obj) -> float:
        if parked_agent_obj is None or released_agent_obj is None:
            return 0.0

        parked_pos = getattr(parked_agent_obj, "pos", None)
        released_pos = getattr(released_agent_obj, "pos", None)
        if parked_pos is None or released_pos is None:
            return 0.0

        parked_xy = tuple(int(v) for v in parked_pos.tolist())
        released_xy = tuple(int(v) for v in released_pos.tolist())
        manhattan = abs(parked_xy[0] - released_xy[0]) + abs(parked_xy[1] - released_xy[1])

        # Follow-through phase after one of two dangerous parked neighbors has
        # been released. Keep the waiting car stopped until the released one
        # has actually created enough space.
        if manhattan <= 3:
            return 1.0
        if manhattan <= 5:
            return 0.9
        if manhattan == 6:
            return 0.8
        if manhattan == 7:
            return 0.65
        if manhattan == 8:
            return 0.5
        if manhattan == 9:
            return 0.35
        if manhattan == 10:
            return 0.2
        if manhattan == 11:
            return 0.12
        if manhattan == 12:
            return 0.06
        return 0.0

    def _upcoming_intersection_cells(self, agent_obj, intersection_matrix, max_cells: int = 24) -> List[tuple[int, int]]:
        if agent_obj is None or intersection_matrix is None:
            return []
        pos = getattr(agent_obj, "pos", None)
        traj = getattr(agent_obj, "global_traj", None)
        if pos is None or traj is None or len(traj) == 0:
            return []
        pos = pos.to(dtype=th.long)
        matches = th.all(traj == pos, dim=1).nonzero(as_tuple=False)
        if matches.numel() == 0:
            return []
        current_idx = int(matches[0].item())
        h, w = intersection_matrix.shape[1], intersection_matrix.shape[2]
        cells: List[tuple[int, int]] = []
        started = False
        for idx in range(current_idx + 1, len(traj)):
            point = traj[idx]
            if point[0] < 0 or point[0] >= h or point[1] < 0 or point[1] >= w:
                continue
            inter_id = int(intersection_matrix[2, point[0], point[1]].item())
            if inter_id > 0:
                started = True
                cells.append((int(point[0].item()), int(point[1].item())))
                if len(cells) >= max_cells:
                    break
            elif started:
                break
        return cells

    def _timed_upcoming_intersection_cells(
        self,
        agent_obj,
        intersection_matrix,
        max_cells: int = 24,
    ) -> List[tuple[int, int, int]]:
        if agent_obj is None or intersection_matrix is None:
            return []
        pos = getattr(agent_obj, "pos", None)
        traj = getattr(agent_obj, "global_traj", None)
        if pos is None or traj is None or len(traj) == 0:
            return []
        pos = pos.to(dtype=th.long)
        matches = th.all(traj == pos, dim=1).nonzero(as_tuple=False)
        if matches.numel() == 0:
            return []
        current_idx = int(matches[0].item())
        h, w = intersection_matrix.shape[1], intersection_matrix.shape[2]
        cells: List[tuple[int, int, int]] = []
        started = False
        for idx in range(current_idx + 1, len(traj)):
            point = traj[idx]
            if point[0] < 0 or point[0] >= h or point[1] < 0 or point[1] >= w:
                continue
            inter_id = int(intersection_matrix[2, point[0], point[1]].item())
            if inter_id > 0:
                started = True
                cells.append((idx - current_idx, int(point[0].item()), int(point[1].item())))
                if len(cells) >= max_cells:
                    break
            elif started:
                break
        return cells

    def _intersection_path_conflict(self, src_agent_obj, dst_agent_obj, intersection_matrix) -> float:
        src_cells = self._timed_upcoming_intersection_cells(src_agent_obj, intersection_matrix)
        dst_cells = self._timed_upcoming_intersection_cells(dst_agent_obj, intersection_matrix)
        if not src_cells or not dst_cells:
            return 0.0

        best_conflict = 0.0
        for src_t, sx, sy in src_cells:
            for dst_t, dx, dy in dst_cells:
                dist = abs(sx - dx) + abs(sy - dy)
                time_gap = abs(src_t - dst_t)

                # Strong conflict only when the agents are expected to occupy the
                # same cell at almost the same time.
                if dist == 0:
                    if time_gap == 0:
                        best_conflict = max(best_conflict, 1.0)
                    elif time_gap == 1:
                        best_conflict = max(best_conflict, 0.8)
                    elif time_gap == 2:
                        best_conflict = max(best_conflict, 0.45)
                # Mild conflict for adjacent cells only when arrival times are
                # near-synchronous. This keeps some crossing sensitivity without
                # turning every shared intersection into a hard stop.
                elif dist == 1:
                    if time_gap == 0:
                        best_conflict = max(best_conflict, 0.35)
                    elif time_gap == 1:
                        best_conflict = max(best_conflict, 0.2)

        # Mere use of the same intersection at clearly separated times should
        # not be treated as a path conflict by the centralized shield.
        if best_conflict > 0.0:
            return best_conflict
        return 0.0

    def _pair_features(
        self,
        src_agent_idx: int,
        dst_agent_idx: int,
        rl_layer_ids: Iterable[int],
        per_agent_facts: List[Dict[str, float]],
        scene_graph_by_layer: Dict[int, Dict],
        global_world=None,
        global_intersection_matrix=None,
        global_agents=None,
    ) -> Dict[str, float]:
        rl_layer_ids = list(rl_layer_ids)
        src_layer_id = int(rl_layer_ids[src_agent_idx])
        dst_layer_id = int(rl_layer_ids[dst_agent_idx])
        src_entity = f"Entity_Car_{src_layer_id}"
        dst_entity = f"Entity_Car_{dst_layer_id}"
        src_tti = None
        dst_tti = None
        src_parked = 0.0
        dst_parked = 0.0
        spawn_release_conflict = 0.0
        same_lane_follow_conflict = 0.0
        rear_agent = -1
        src_priority = 0.0
        dst_priority = 0.0

        if global_world is not None and global_intersection_matrix is not None and global_agents is not None:
            src_agent_obj = global_agents.get(str(src_layer_id))
            dst_agent_obj = global_agents.get(str(dst_layer_id))
            src_priority = float(getattr(src_agent_obj, "priority", 0.0) or 0.0)
            dst_priority = float(getattr(dst_agent_obj, "priority", 0.0) or 0.0)
            src_parked = float(IsParked(global_world, global_intersection_matrix, global_agents, src_entity))
            dst_parked = float(IsParked(global_world, global_intersection_matrix, global_agents, dst_entity))
            src_at = float(IsAtInter(global_world, global_intersection_matrix, global_agents, src_entity))
            src_in = float(IsInInter(global_world, global_intersection_matrix, global_agents, src_entity))
            dst_at = float(IsAtInter(global_world, global_intersection_matrix, global_agents, dst_entity))
            dst_in = float(IsInInter(global_world, global_intersection_matrix, global_agents, dst_entity))
            # The centralized shield should not depend solely on the global
            # predicate path here. During debugging we repeatedly observe cases
            # where the per-agent symbolic observation already marks an RL car
            # as `at_inter`, while the global helper still reports zero. Blend
            # both views and trust the stronger gate-engagement signal.
            src_at = max(src_at, float(per_agent_facts[src_agent_idx].get("ego_is_at_inter", 0.0)))
            src_in = max(src_in, float(per_agent_facts[src_agent_idx].get("ego_is_in_inter", 0.0)))
            dst_at = max(dst_at, float(per_agent_facts[dst_agent_idx].get("ego_is_at_inter", 0.0)))
            dst_in = max(dst_in, float(per_agent_facts[dst_agent_idx].get("ego_is_in_inter", 0.0)))
            src_inter_id = self._agent_intersection_context_id(src_agent_obj, global_intersection_matrix)
            dst_inter_id = self._agent_intersection_context_id(dst_agent_obj, global_intersection_matrix)
            same_inter = float(src_inter_id > 0 and src_inter_id == dst_inter_id)
            local_same_inter = max(
                self._relation_true(scene_graph_by_layer, src_layer_id, dst_layer_id, "SameInter"),
                self._relation_true(scene_graph_by_layer, dst_layer_id, src_layer_id, "SameInter"),
            )
            src_active_inter_id = self._active_intersection_context_id(
                src_agent_obj,
                global_intersection_matrix,
                src_at,
                src_in,
            )
            dst_active_inter_id = self._active_intersection_context_id(
                dst_agent_obj,
                global_intersection_matrix,
                dst_at,
                dst_in,
            )
            active_same_inter = float(
                src_active_inter_id > 0
                and src_active_inter_id == dst_active_inter_id
            )
            src_tti = self._time_to_next_intersection(src_agent_obj, global_intersection_matrix)
            dst_tti = self._time_to_next_intersection(dst_agent_obj, global_intersection_matrix)
            # `at + at` is a coordination case, not yet a hard shared-occupancy
            # conflict. Reserve "critical" for cases where at least one RL car
            # is already inside the intersection.
            src_higher = float(HigherPri(global_world, global_intersection_matrix, global_agents, src_entity, dst_entity))
            dst_higher = float(HigherPri(global_world, global_intersection_matrix, global_agents, dst_entity, src_entity))
            local_src_higher = self._relation_true(scene_graph_by_layer, src_layer_id, dst_layer_id, "HigherPri")
            local_dst_higher = self._relation_true(scene_graph_by_layer, dst_layer_id, src_layer_id, "HigherPri")
            src_higher = max(src_higher, local_src_higher)
            dst_higher = max(dst_higher, local_dst_higher)
            same_inter = local_same_inter if local_same_inter > 0.0 else same_inter
            # Strong centralized coordination must use the active local
            # intersection around the current gate/occupancy state, not only a
            # broad future-route intersection context. However, for RL-RL
            # approach conflicts we also need an earlier trigger before the
            # local `at_inter` predicate fires, otherwise collision-close is
            # reached too late.
            approach_imminent = 0.0
            if same_inter > 0.0 and src_tti is not None and dst_tti is not None:
                same_priority_gap = abs(int(src_tti) - int(dst_tti))
                both_close = max(int(src_tti), int(dst_tti)) <= self._imminent_tti_threshold
                one_very_close = min(int(src_tti), int(dst_tti)) <= max(2, self._critical_tti_threshold)
                priority_ordered = (src_higher > 0.0) or (dst_higher > 0.0)
                tied_approach = same_priority_gap <= 2
                if both_close and (priority_ordered or tied_approach):
                    approach_imminent = 1.0
                elif one_very_close and same_priority_gap <= 3:
                    approach_imminent = 1.0
            # If two RL agents already share the same local intersection context
            # and at least one of them is at/in that gate, centralized
            # coordination should activate immediately. Otherwise the joint
            # shield reacts too late and only sees the conflict once
            # collision-close has already appeared.
            src_gate_engaged = (src_at > 0.0) or (src_in > 0.0)
            dst_gate_engaged = (dst_at > 0.0) or (dst_in > 0.0)
            gate_context_engaged = 0.0
            if same_inter > 0.0:
                # Immediate coordination if both RL cars are already at/in the
                # same gate context.
                if src_gate_engaged and dst_gate_engaged:
                    gate_context_engaged = 1.0
                # If the local relational view already says this exact pair is
                # tied to the same intersection, one engaged car is enough to
                # hand control to the centralized coordinator.
                elif local_same_inter > 0.0 and (src_gate_engaged or dst_gate_engaged):
                    gate_context_engaged = 1.0
                # Otherwise only activate early when the non-engaged car is
                # actually close to entering that same intersection. This avoids
                # coupling agents that merely share some later route
                # intersection.
                elif src_gate_engaged and dst_tti is not None and int(dst_tti) <= 3:
                    gate_context_engaged = 1.0
                elif dst_gate_engaged and src_tti is not None and int(src_tti) <= 3:
                    gate_context_engaged = 1.0
            same_inter_imminent = max(active_same_inter, approach_imminent, gate_context_engaged)
            same_inter_critical = float(
                same_inter_imminent > 0.0
                and (
                    src_in > 0.0
                    or dst_in > 0.0
                )
            )
            colliding_close = float(
                max(
                    CollidingClose(global_world, global_intersection_matrix, global_agents, src_entity, dst_entity),
                    CollidingClose(global_world, global_intersection_matrix, global_agents, dst_entity, src_entity),
                )
            )
            local_colliding_close = max(
                self._relation_true(scene_graph_by_layer, src_layer_id, dst_layer_id, "CollidingClose"),
                self._relation_true(scene_graph_by_layer, dst_layer_id, src_layer_id, "CollidingClose"),
            )
            colliding_close = max(colliding_close, local_colliding_close)
            future_path_conflict = max(
                self._future_path_conflict(src_agent_obj, dst_agent_obj, horizon_steps=self._imminent_tti_threshold),
                self._intersection_path_conflict(src_agent_obj, dst_agent_obj, global_intersection_matrix),
            )
            same_lane_follow_conflict, rear_agent = self._same_lane_follow_conflict(src_agent_obj, dst_agent_obj)
            if src_parked > 0.0 and dst_parked > 0.0:
                spawn_release_conflict = self._spawn_release_conflict(src_agent_obj, dst_agent_obj)
            elif src_parked > 0.0 or dst_parked > 0.0:
                # Narrow continuation of the parked-pair release exception:
                # once one of two dangerously close parked RL cars is released,
                # the other remains blocked until the released one has cleared.
                if src_parked > 0.0:
                    spawn_release_conflict = self._spawn_release_buffer_conflict(src_agent_obj, dst_agent_obj)
                    preferred_mover_hint = 1
                else:
                    spawn_release_conflict = self._spawn_release_buffer_conflict(dst_agent_obj, src_agent_obj)
                    preferred_mover_hint = 0
                same_inter = 0.0
                same_inter_imminent = 0.0
                same_inter_critical = 0.0
                src_higher = 0.0
                dst_higher = 0.0
                colliding_close = 0.0
                future_path_conflict = 0.0
                # Outside this short release buffer, revert to the intended
                # semantics: moving RL agents ignore parked RL agents.
                if spawn_release_conflict <= 0.0:
                    preferred_mover_hint = -1
                preferred_mover = preferred_mover_hint
        else:
            src_at = float(per_agent_facts[src_agent_idx].get("ego_is_at_inter", 0.0))
            src_in = float(per_agent_facts[src_agent_idx].get("ego_is_in_inter", 0.0))
            dst_at = float(per_agent_facts[dst_agent_idx].get("ego_is_at_inter", 0.0))
            dst_in = float(per_agent_facts[dst_agent_idx].get("ego_is_in_inter", 0.0))

            same_inter = max(
                self._relation_true(scene_graph_by_layer, src_layer_id, dst_layer_id, "SameInter"),
                self._relation_true(scene_graph_by_layer, dst_layer_id, src_layer_id, "SameInter"),
            )
            src_higher = self._relation_true(scene_graph_by_layer, src_layer_id, dst_layer_id, "HigherPri")
            dst_higher = self._relation_true(scene_graph_by_layer, dst_layer_id, src_layer_id, "HigherPri")
            colliding_close = max(
                self._relation_true(scene_graph_by_layer, src_layer_id, dst_layer_id, "CollidingClose"),
                self._relation_true(scene_graph_by_layer, dst_layer_id, src_layer_id, "CollidingClose"),
            )
            same_inter_imminent = same_inter
            same_inter_critical = same_inter
            future_path_conflict = 0.0

        priority_tie = float(
            same_inter > 0.0
            and src_higher <= 0.0
            and dst_higher <= 0.0
        )
        preferred_mover = locals().get("preferred_mover", -1)
        if spawn_release_conflict > 0.0 and (
            (src_parked > 0.0 and dst_parked > 0.0)
            or (src_parked > 0.0 and dst_parked <= 0.0)
            or (dst_parked > 0.0 and src_parked <= 0.0)
        ):
            if src_priority > dst_priority:
                preferred_mover = 0
            elif dst_priority > src_priority:
                preferred_mover = 1
            elif src_tti is not None and dst_tti is not None:
                if src_tti + 1 < dst_tti:
                    preferred_mover = 0
                elif dst_tti + 1 < src_tti:
                    preferred_mover = 1
                else:
                    preferred_mover = 0 if src_layer_id <= dst_layer_id else 1
            else:
                preferred_mover = 0 if src_layer_id <= dst_layer_id else 1
        elif priority_tie > 0.0:
            if src_tti is not None and dst_tti is not None:
                if src_tti + 1 < dst_tti:
                    preferred_mover = 0
                elif dst_tti + 1 < src_tti:
                    preferred_mover = 1
                else:
                    preferred_mover = 0 if src_layer_id <= dst_layer_id else 1
            else:
                preferred_mover = 0 if src_layer_id <= dst_layer_id else 1

        # Treat only configurations with at least one vehicle already inside
        # the intersection as hard occupancy overlap. Two vehicles both waiting
        # at the boundary should be coordinated, not immediately collapsed into
        # an all-stop hard conflict.
        gate_overlap = max(
            src_at * dst_in,
            src_in * dst_at,
        )
        interior_overlap = src_in * dst_in
        occupancy_overlap = max(gate_overlap, interior_overlap)
        boundary_overlap = src_at * dst_at
        active_coordination_signal = max(
            float(active_same_inter),
            gate_overlap,
            interior_overlap,
            future_path_conflict,
            0.35 * boundary_overlap * max(src_higher, dst_higher),
            0.25 * boundary_overlap * priority_tie,
            same_lane_follow_conflict,
            colliding_close,
            spawn_release_conflict,
        )

        pair_hazard = max(
            colliding_close,
            spawn_release_conflict,
            future_path_conflict,
            same_lane_follow_conflict,
            interior_overlap,
            0.35 * same_inter_critical * gate_overlap,
            0.25 * same_inter_imminent * float(same_inter > 0.0) * float(
                (src_tti is not None and dst_tti is not None and abs(int(src_tti) - int(dst_tti)) <= 2)
            ),
            # Two cars both at the boundary is still relevant, but it should
            # stay as a coordination cue unless another hard signal exists.
            0.15 * same_inter_imminent * boundary_overlap * max(src_higher, dst_higher),
        )
        return {
            "same_inter": float(np.clip(same_inter, 0.0, 1.0)),
            "same_inter_imminent": float(np.clip(same_inter_imminent, 0.0, 1.0)),
            "same_inter_critical": float(np.clip(same_inter_critical, 0.0, 1.0)),
            "src_higher": float(np.clip(src_higher, 0.0, 1.0)),
            "dst_higher": float(np.clip(dst_higher, 0.0, 1.0)),
            "priority_tie": float(np.clip(priority_tie, 0.0, 1.0)),
            "preferred_mover": float(preferred_mover),
            "src_parked": float(np.clip(src_parked, 0.0, 1.0)),
            "dst_parked": float(np.clip(dst_parked, 0.0, 1.0)),
            "colliding_close": float(np.clip(colliding_close, 0.0, 1.0)),
            "spawn_release_conflict": float(np.clip(spawn_release_conflict, 0.0, 1.0)),
            "same_lane_follow_conflict": float(np.clip(same_lane_follow_conflict, 0.0, 1.0)),
            "rear_agent": float(rear_agent),
            "active_coordination_signal": float(np.clip(active_coordination_signal, 0.0, 1.0)),
            "gate_overlap": float(np.clip(gate_overlap, 0.0, 1.0)),
            "interior_overlap": float(np.clip(interior_overlap, 0.0, 1.0)),
            "occupancy_overlap": float(np.clip(occupancy_overlap, 0.0, 1.0)),
            "boundary_overlap": float(np.clip(boundary_overlap, 0.0, 1.0)),
            "future_path_conflict": float(np.clip(future_path_conflict, 0.0, 1.0)),
            "pair_hazard": float(np.clip(pair_hazard, 0.0, 1.0)),
        }

    def _joint_action_safety(
        self,
        joint_action: tuple[int, ...],
        local_action_safety: np.ndarray,
        per_agent_hazard: np.ndarray,
        per_agent_local_metrics: List[Dict[str, float]],
        per_agent_facts: List[Dict[str, float]],
        pair_features: Dict[tuple[int, int], Dict[str, float]],
    ) -> float:
        score = 1.0
        num_agents = len(joint_action)
        local_score_product = 1.0
        for agent_idx, action_id in enumerate(joint_action):
            local_score_product *= float(local_action_safety[agent_idx, action_id])
        # Preserve decentralized local safety logic and avoid automatic decay
        # purely because more RL agents are present.
        score *= float(np.clip(local_score_product, 0.0, 1.0)) ** (1.0 / max(num_agents, 1))

        # Hard local emergency override: if an RL car already has a clear local
        # collision-close or extreme pedestrian stop condition, the centralized
        # joint selector must not re-allow motion for that agent.
        for agent_idx, action_id in enumerate(joint_action):
            local_metrics = per_agent_local_metrics[agent_idx]
            local_colliding_close = float(local_metrics.get("colliding_close", 0.0))
            local_ped_conflict = float(local_metrics.get("ped_conflict", 0.0))
            if action_id != 3 and (
                local_colliding_close >= 0.5
                or local_ped_conflict >= 0.85
            ):
                score *= 0.01

        for src_agent_idx in range(num_agents):
            for dst_agent_idx in range(src_agent_idx + 1, num_agents):
                features = pair_features[(src_agent_idx, dst_agent_idx)]
                if (
                    features["same_inter"] <= 0.0
                    and features["colliding_close"] <= 0.0
                    and features.get("spawn_release_conflict", 0.0) <= 0.0
                ):
                    continue

                src_action = int(joint_action[src_agent_idx])
                dst_action = int(joint_action[dst_agent_idx])
                src_moves = src_action != 3
                dst_moves = dst_action != 3
                src_speed = self._speed_factor(src_action)
                dst_speed = self._speed_factor(dst_action)
                src_local_hazard = float(per_agent_hazard[src_agent_idx])
                dst_local_hazard = float(per_agent_hazard[dst_agent_idx])
                coord_ready = features.get("active_coordination_signal", 0.0) > 0.0

                pair_penalty = 0.0
                hard_conflict = max(
                    features["colliding_close"],
                    features.get("spawn_release_conflict", 0.0),
                    features.get("interior_overlap", 0.0),
                )
                parked_release_conflict = (
                    features.get("spawn_release_conflict", 0.0) > 0.0
                    and (
                        (features.get("src_parked", 0.0) > 0.0 and features.get("dst_parked", 0.0) > 0.0)
                        or (features.get("src_parked", 0.0) > 0.0 and features.get("dst_parked", 0.0) <= 0.0)
                        or (features.get("dst_parked", 0.0) > 0.0 and features.get("src_parked", 0.0) <= 0.0)
                    )
                )
                preferred_idx = int(features.get("preferred_mover", -1))
                rear_agent = int(features.get("rear_agent", -1))

                # Hard safety stays local/traffic-semantic: collision-close,
                # simultaneous interior occupation, and direct collision-close.
                if parked_release_conflict:
                    if preferred_idx == 0:
                        designated_moves = src_moves
                        other_moves = dst_moves
                    elif preferred_idx == 1:
                        designated_moves = dst_moves
                        other_moves = src_moves
                    else:
                        designated_moves = src_moves
                        other_moves = dst_moves

                    if designated_moves and not other_moves:
                        pair_penalty = 0.0
                    elif (not designated_moves) and other_moves:
                        pair_penalty = 0.94
                    elif designated_moves and other_moves:
                        pair_penalty = 0.985
                    else:
                        pair_penalty = 0.92
                elif features.get("same_lane_follow_conflict", 0.0) > 0.0:
                    if rear_agent == 0:
                        rear_moves = src_moves
                        front_moves = dst_moves
                        rear_speed = src_speed
                        front_speed = dst_speed
                    elif rear_agent == 1:
                        rear_moves = dst_moves
                        front_moves = src_moves
                        rear_speed = dst_speed
                        front_speed = src_speed
                    else:
                        rear_moves = src_moves or dst_moves
                        front_moves = src_moves and dst_moves
                        rear_speed = max(src_speed, dst_speed)
                        front_speed = min(src_speed, dst_speed)

                    if rear_moves and (not front_moves or rear_speed > front_speed):
                        pair_penalty = max(
                            pair_penalty,
                            0.92 * features.get("same_lane_follow_conflict", 0.0),
                        )
                elif hard_conflict > 0.0 and src_moves and dst_moves:
                    pair_penalty = max(
                        pair_penalty,
                        max(src_speed, dst_speed) * max(
                            0.95 * features["colliding_close"],
                            0.98 * features.get("spawn_release_conflict", 0.0),
                            0.95 * features.get("interior_overlap", 0.0),
                        ),
                    )
                elif coord_ready and src_moves and dst_moves:
                    pair_penalty = max(
                        pair_penalty,
                        max(src_speed, dst_speed) * max(
                            0.65 * features.get("gate_overlap", 0.0),
                            0.45 * features["same_inter_critical"],
                            0.60 * features["future_path_conflict"],
                        ),
                    )
                elif features["colliding_close"] > 0.0 and (src_moves or dst_moves):
                    pair_penalty = max(
                        pair_penalty,
                        0.20 * max(src_speed, dst_speed) * features["colliding_close"],
                    )
                score *= float(np.clip(1.0 - np.clip(pair_penalty, 0.0, 1.0), 0.0, 1.0))

        # Centralized advantage: after local safety is accounted for, apply a
        # sharp RL-RL coordination mask over same-intersection groups.
        coordination_adjacency = {agent_idx: set() for agent_idx in range(num_agents)}
        for src_agent_idx in range(num_agents):
            for dst_agent_idx in range(src_agent_idx + 1, num_agents):
                features = pair_features[(src_agent_idx, dst_agent_idx)]
                hard_conflict = max(
                    features["colliding_close"],
                    features.get("interior_overlap", 0.0),
                )
                coord_ready = features.get("active_coordination_signal", 0.0) > 0.0
                if (
                    coord_ready
                    and hard_conflict < 0.95
                    and max(float(per_agent_hazard[src_agent_idx]), float(per_agent_hazard[dst_agent_idx])) < 0.80
                ):
                    coordination_adjacency[src_agent_idx].add(dst_agent_idx)
                    coordination_adjacency[dst_agent_idx].add(src_agent_idx)

        visited = set()
        coordination_components = []
        for agent_idx in range(num_agents):
            if agent_idx in visited or len(coordination_adjacency[agent_idx]) == 0:
                continue
            stack = [agent_idx]
            component = set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                stack.extend(coordination_adjacency[node] - visited)
            if len(component) >= 2:
                coordination_components.append(sorted(component))

        for component in coordination_components:
            dominance = {agent_idx: 0.0 for agent_idx in component}
            for idx_a, src_agent_idx in enumerate(component):
                for dst_agent_idx in component[idx_a + 1 :]:
                    pair_key = (min(src_agent_idx, dst_agent_idx), max(src_agent_idx, dst_agent_idx))
                    features = pair_features[pair_key]
                    if features["src_higher"] > features["dst_higher"]:
                        dominance[pair_key[0]] += 1.0
                    elif features["dst_higher"] > features["src_higher"]:
                        dominance[pair_key[1]] += 1.0

            best_score = max(dominance.values())
            top_agents = [agent_idx for agent_idx, dom_score in dominance.items() if dom_score == best_score]
            agents_in_intersection = [
                agent_idx
                for agent_idx in component
                if float(per_agent_facts[agent_idx].get("ego_is_in_inter", 0.0)) > 0.0
            ]
            if len(agents_in_intersection) == 1:
                designated_mover = agents_in_intersection[0]
            elif len(agents_in_intersection) > 1:
                designated_mover = min(
                    agents_in_intersection,
                    key=lambda agent_idx: (float(per_agent_hazard[agent_idx]), -dominance.get(agent_idx, 0.0), agent_idx),
                )
            elif len(top_agents) == 1:
                designated_mover = top_agents[0]
            else:
                designated_mover = min(
                    top_agents,
                    key=lambda agent_idx: (float(per_agent_hazard[agent_idx]), agent_idx),
                )

            designated_action = int(joint_action[designated_mover])
            other_actions = {agent_idx: int(joint_action[agent_idx]) for agent_idx in component if agent_idx != designated_mover}
            designated_moves = designated_action != 3
            other_movers = [agent_idx for agent_idx, action_id in other_actions.items() if action_id != 3]
            component_hazard = max(float(per_agent_hazard[agent_idx]) for agent_idx in component)
            designated_local_metrics = per_agent_local_metrics[designated_mover]
            designated_local_emergency = max(
                float(designated_local_metrics.get("colliding_close", 0.0)),
                float(designated_local_metrics.get("ped_conflict", 0.0)),
            )
            designated_gate_engaged = max(
                float(per_agent_facts[designated_mover].get("ego_is_at_inter", 0.0)),
                float(per_agent_facts[designated_mover].get("ego_is_in_inter", 0.0)),
            )
            other_gate_engaged = max(
                (
                    max(
                        float(per_agent_facts[agent_idx].get("ego_is_at_inter", 0.0)),
                        float(per_agent_facts[agent_idx].get("ego_is_in_inter", 0.0)),
                    )
                    for agent_idx in component
                    if agent_idx != designated_mover
                ),
                default=0.0,
            )
            release_ready = (
                designated_local_emergency < 0.1
                and designated_gate_engaged > 0.0
                and other_gate_engaged > 0.0
            )

            coord_multiplier = 1.0
            if component_hazard < 0.2:
                if designated_moves and len(other_movers) == 0:
                    coord_multiplier = 1.0
                elif not designated_moves and len(other_movers) == 0:
                    coord_multiplier = 0.005 if release_ready else 0.03
                elif designated_moves and len(other_movers) > 0:
                    coord_multiplier = 0.02
                elif (not designated_moves) and len(other_movers) == 1:
                    coord_multiplier = 0.05
                else:
                    coord_multiplier = 0.01
            else:
                # If local hazard is not negligible, keep the mask softer.
                if designated_moves and len(other_movers) == 0:
                    coord_multiplier = 1.0
                elif not designated_moves and len(other_movers) == 0:
                    coord_multiplier = 0.06 if release_ready else 0.35
                elif designated_moves and len(other_movers) > 0:
                    coord_multiplier = 0.10
                elif (not designated_moves) and len(other_movers) == 1:
                    coord_multiplier = 0.20
                else:
                    coord_multiplier = 0.05

            # In centralized coordination, once one RL car is selected to pass
            # while others yield, prefer a cautious crossing speed. Letting the
            # designated mover blast through at `fast` is exactly what keeps
            # producing the late collision in the debug episode.
            if designated_moves and len(other_movers) == 0:
                if designated_action == 0:  # slow
                    speed_coord_multiplier = 1.0
                elif designated_action == 1:  # normal
                    speed_coord_multiplier = 0.92 if component_hazard < 0.5 else 0.82
                else:  # fast
                    speed_coord_multiplier = 0.72 if component_hazard < 0.5 else 0.45
                coord_multiplier *= speed_coord_multiplier

            score *= coord_multiplier

        num_stops = sum(1 for action_id in joint_action if int(action_id) == 3)
        max_active_conflict = max(
            (
                max(
                    features["colliding_close"],
                    features.get("interior_overlap", 0.0),
                )
                for features in pair_features.values()
            ),
            default=0.0,
        )
        if num_stops >= 2 and max_active_conflict < 0.2 and float(np.max(per_agent_hazard) if len(per_agent_hazard) > 0 else 0.0) < 0.25:
            if num_stops == num_agents:
                score *= 0.82
            else:
                score *= 0.90

        # For parked-release conflicts, "everyone keeps stopping" should not
        # remain a stable alternative once a designated mover exists.
        for src_agent_idx in range(num_agents):
            for dst_agent_idx in range(src_agent_idx + 1, num_agents):
                features = pair_features[(src_agent_idx, dst_agent_idx)]
                if not (
                    features.get("spawn_release_conflict", 0.0) > 0.0
                    and (
                        (features.get("src_parked", 0.0) > 0.0 and features.get("dst_parked", 0.0) > 0.0)
                        or (features.get("src_parked", 0.0) > 0.0 and features.get("dst_parked", 0.0) <= 0.0)
                        or (features.get("dst_parked", 0.0) > 0.0 and features.get("src_parked", 0.0) <= 0.0)
                    )
                ):
                    continue
                if int(joint_action[src_agent_idx]) == 3 and int(joint_action[dst_agent_idx]) == 3:
                    score *= 0.08

        return float(np.clip(score, 0.0, 1.0))

    def shield_policy(self, base_probs: th.Tensor, obs_tensor: th.Tensor, joint_context: Optional[Dict] = None):
        num_agents = int(base_probs.shape[0])
        if self.num_agents is not None and num_agents != int(self.num_agents):
            raise ValueError(
                "Centralized shield expected {} agents but received {}.".format(self.num_agents, num_agents)
            )

        scene_graph_by_layer = self._normalize_scene_graph((joint_context or {}).get("scene_graph_by_layer"))
        rl_layer_ids = list((joint_context or {}).get("rl_layer_ids") or list(range(num_agents)))
        global_world = (joint_context or {}).get("global_world")
        global_intersection_matrix = (joint_context or {}).get("global_intersection_matrix")
        global_agent_snapshots = (joint_context or {}).get("global_agents") or {}
        global_agents = {}
        for layer_id, snapshot in global_agent_snapshots.items():
            agent_obj = SimpleNamespace(
                type=snapshot.get("type"),
                layer_id=int(snapshot.get("layer_id")),
                id=int(snapshot.get("id")),
                priority=int(snapshot.get("priority")),
                concepts=list(snapshot.get("concepts", [])),
                pos=snapshot.get("pos"),
                start=snapshot.get("start"),
                goal=snapshot.get("goal"),
                global_traj=snapshot.get("global_traj"),
                reach_goal=bool(snapshot.get("reach_goal", False)),
                moving_direction=snapshot.get("moving_direction"),
                last_move_dir=snapshot.get("last_move_dir"),
            )
            global_agents[str(int(layer_id))] = agent_obj

        per_agent_facts = []
        per_agent_local_scores = []
        per_agent_hazard = []
        per_agent_local_metrics = []
        for obs_row in obs_tensor.detach().cpu():
            fact_weights, local_scores, local_metrics = self._local_action_safety(obs_row)
            per_agent_facts.append(fact_weights)
            per_agent_local_scores.append(local_scores)
            per_agent_hazard.append(float(local_metrics.get("hazard", 0.0)))
            per_agent_local_metrics.append(local_metrics)

        local_action_safety = np.stack(per_agent_local_scores, axis=0)

        pair_features = {}
        pair_hazard_by_agent = np.zeros(num_agents, dtype=np.float32)
        for src_agent_idx in range(num_agents):
            for dst_agent_idx in range(src_agent_idx + 1, num_agents):
                features = self._pair_features(
                    src_agent_idx,
                    dst_agent_idx,
                    rl_layer_ids,
                    per_agent_facts,
                    scene_graph_by_layer,
                    global_world=global_world,
                    global_intersection_matrix=global_intersection_matrix,
                    global_agents=global_agents if len(global_agents) > 0 else None,
                )
                pair_features[(src_agent_idx, dst_agent_idx)] = features
                pair_hazard_by_agent[src_agent_idx] = max(
                    pair_hazard_by_agent[src_agent_idx],
                    features["pair_hazard"],
                )
                pair_hazard_by_agent[dst_agent_idx] = max(
                    pair_hazard_by_agent[dst_agent_idx],
                    features["pair_hazard"],
                )

        # RL-vs-RL intersection right-of-way should be decided by the
        # centralized coordinator, not by independent local stop pressure,
        # until the agents are actually at/in the interaction zone.
        for agent_idx in range(num_agents):
            if self._should_defer_local_rl_intersection_logic(
                agent_idx,
                pair_features,
                per_agent_local_metrics[agent_idx],
                num_agents,
            ):
                local_action_safety[agent_idx, :] = 1.0
                per_agent_hazard[agent_idx] = max(
                    float(per_agent_local_metrics[agent_idx].get("colliding_close", 0.0)),
                    float(per_agent_local_metrics[agent_idx].get("ped_conflict", 0.0)),
                )

        joint_actions = self._joint_actions(num_agents)
        base_probs_np = base_probs.detach().cpu().numpy()
        joint_base_probs = np.zeros(len(joint_actions), dtype=np.float32)
        joint_safety = np.zeros(len(joint_actions), dtype=np.float32)
        for joint_idx, joint_action in enumerate(joint_actions):
            joint_prob = 1.0
            for agent_idx, action_id in enumerate(joint_action):
                joint_prob *= float(base_probs_np[agent_idx, action_id])
            joint_base_probs[joint_idx] = float(joint_prob)
            joint_safety[joint_idx] = self._joint_action_safety(
                joint_action,
                local_action_safety,
                np.asarray(per_agent_hazard, dtype=np.float32),
                per_agent_local_metrics,
                per_agent_facts,
                pair_features,
            )

        weighted = joint_base_probs * joint_safety
        denom = float(max(weighted.sum(), 1e-8))
        joint_shielded_probs = weighted / denom

        shielded_probs = np.zeros_like(base_probs_np, dtype=np.float32)
        safety_probs = np.zeros_like(base_probs_np, dtype=np.float32)
        for joint_idx, joint_action in enumerate(joint_actions):
            score = float(joint_safety[joint_idx])
            for agent_idx, action_id in enumerate(joint_action):
                shielded_probs[agent_idx, action_id] += float(joint_shielded_probs[joint_idx])

        for agent_idx in range(num_agents):
            for action_id in range(len(self.action_names)):
                conditioned_safe = 0.0
                for joint_idx, joint_action in enumerate(joint_actions):
                    if joint_action[agent_idx] != action_id:
                        continue
                    other_prob = 1.0
                    for other_agent_idx, other_action_id in enumerate(joint_action):
                        if other_agent_idx == agent_idx:
                            continue
                        other_prob *= float(base_probs_np[other_agent_idx, other_action_id])
                    conditioned_safe += other_prob * float(joint_safety[joint_idx])
                safety_probs[agent_idx, action_id] = float(conditioned_safe)

        base_policy_safe_prob = float(weighted.sum())
        shielded_policy_safe_prob = float((joint_shielded_probs * joint_safety).sum())
        hazard = np.maximum(np.asarray(per_agent_hazard, dtype=np.float32), pair_hazard_by_agent)
        top_joint_indices = np.argsort(joint_shielded_probs)[::-1][: min(5, len(joint_actions))]
        top_joint_actions = []
        for joint_idx in top_joint_indices.tolist():
            top_joint_actions.append(
                {
                    "joint_action_ids": [int(action_id) for action_id in joint_actions[joint_idx]],
                    "joint_action_names": self._format_joint_action(joint_actions[joint_idx]),
                    "joint_base_prob": float(joint_base_probs[joint_idx]),
                    "joint_safety": float(joint_safety[joint_idx]),
                    "joint_weighted": float(weighted[joint_idx]),
                    "joint_shielded_prob": float(joint_shielded_probs[joint_idx]),
                }
            )
        pair_debug = {
            f"{src_agent_idx}-{dst_agent_idx}": {k: float(v) for k, v in features.items()}
            for (src_agent_idx, dst_agent_idx), features in pair_features.items()
        }

        return {
            "safety_probs": th.tensor(safety_probs, dtype=base_probs.dtype, device=base_probs.device),
            "shielded_probs": th.tensor(shielded_probs, dtype=base_probs.dtype, device=base_probs.device),
            "base_policy_safe_prob": th.full(
                (num_agents,),
                base_policy_safe_prob,
                dtype=base_probs.dtype,
                device=base_probs.device,
            ),
            "shielded_policy_safe_prob": th.full(
                (num_agents,),
                shielded_policy_safe_prob,
                dtype=base_probs.dtype,
                device=base_probs.device,
            ),
            "template_metrics": {
                "hazard": th.tensor(hazard, dtype=base_probs.dtype, device=base_probs.device),
                "joint_hazard": th.full(
                    (num_agents,),
                    float(np.max(hazard) if len(hazard) > 0 else 0.0),
                    dtype=base_probs.dtype,
                    device=base_probs.device,
                ),
            },
            "joint_debug": {
                "rl_layer_ids": [int(layer_id) for layer_id in rl_layer_ids],
                "pair_features": pair_debug,
                "top_joint_actions": top_joint_actions,
                "base_policy_safe_prob_scalar": float(base_policy_safe_prob),
                "shielded_policy_safe_prob_scalar": float(shielded_policy_safe_prob),
                "joint_hazard_scalar": float(np.max(hazard) if len(hazard) > 0 else 0.0),
            },
        }
