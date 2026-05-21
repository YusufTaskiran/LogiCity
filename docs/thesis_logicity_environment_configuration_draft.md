# LogiCity Environment Configuration Draft

## 4.2 LogiCity Environment Configuration

This subsection describes the concrete LogiCity environment instance used in the thesis experiments. The aim is to define the task world, agent composition, symbolic observations, action space, rules, reward structure, termination conditions, and dataset construction procedure. This section therefore focuses on the configured experimental environment rather than on the probabilistic logic shielding method itself.

### 4.2.1 Task Variant

The thesis uses an adapted **Safe Path Following** setting in LogiCity. The controlled agent is required to navigate along a planned route in a multi-agent urban scene while avoiding safety violations and progressing toward its goal. The task is sequential and partially observable.

The concrete setting is:

- a **single ego agent** controlled by reinforcement learning
- **background agents** controlled by the symbolic expert local planner
- a shared world containing multiple cars and a pedestrian
- evaluation over complete episodes rather than one-step prediction

More specifically:

- `Car_1` is the RL-controlled ego car
- `Car_2` and `Car_3` are rule-based background cars
- `Pedestrian_1` is a rule-based background pedestrian

Thus, the environment is multi-agent at the world level, but only one agent is optimized during training in the current decentralized setup.

### 4.2.2 Urban Map and Static Semantics

All experiments use a fixed city map:

- `config/maps/square_2x2.yaml`

The map size is:

- `103 x 103`

The world contains:

- traffic streets
- walking streets
- buildings with semantic labels such as `Office`, `Store`, `Gas Station`, `Garage`, and `House`

The static map is not procedurally changed between train, validation, and test. Instead, the same map topology is reused while agent start-goal assignments and scenario instances change across episodes.

The map has an explicit urban structure with:

- orthogonal traffic streets
- orthogonal walking streets
- derived intersection regions

The simulator computes a structured intersection representation in [city.py](/d:/Dev/LogicityFresh/LogiCity/logicity/core/city.py:126). This intersection representation distinguishes:

- car entry lines
- pedestrian entry lines
- interior intersection blocks

These labeled structures are central to the thesis because the safety rules and failure definitions are intersection-centric. In particular, predicates such as `IsAtInter` and `IsInInter`, and failures such as simultaneous same-block entry, depend on this explicit intersection geometry.

### 4.2.3 Agent Configuration

The agent configuration is loaded from:

- `config/agents/thesis/main_3cars_1ped.yaml`

The configured agents are:

| Component | Configuration |
| --- | --- |
| Controlled agent | `Car_1` |
| Background cars | `Car_2`, `Car_3` |
| Background pedestrian | `Pedestrian_1` |
| Car planner | `A*vg` |
| Pedestrian planner | `A*` |
| RL scene region | `agent_region = 100` |

The initial semantic concepts defined in the agent YAML are:

| Agent | Type | Initial priority |
| --- | --- | --- |
| `Car_1` | Car | `1` |
| `Car_2` | Car | `5` |
| `Car_3` | Car | `3` |
| `Pedestrian_1` | Pedestrian | `0` |

However, for the RL environment, car priorities are not fixed to these initial YAML values. The RL configuration uses:

- `max_priority: 3`

and dataset generation or episode reset can assign new priorities within that bounded range. As a result, the thesis environment uses a priority-based interaction system, but the effective priorities are episode-dependent rather than permanently frozen to the YAML defaults.

The train, validation, and test settings all use the same agent composition:

- 3 cars
- 1 pedestrian

The main differences across splits come from the sampled or cached episode layouts, not from a different number or type of agents.

### 4.2.4 Symbolic Predicates and Observations

The simulator internally supports a richer grounded symbolic representation than the final RL observation. The current expert-rule file and local planner use the following grounded predicates:

- `IsPedestrian`
- `IsCar`
- `IsAtInter`
- `IsInInter`
- `HigherPri`
- `IsAhead`
- `IsCloseAhead`
- `CollidingClose`
- plus the action predicates `Stop`, `Fast`, `Normal`, `Slow`

For RL, the current thesis configuration uses:

- `grounding_mode: thesis_minimal`

This compresses the grounded symbolic state into a fixed 6-dimensional vector, implemented in [gym_wrapper.py](/d:/Dev/LogicityFresh/LogiCity/logicity/utils/gym_wrapper.py:233). The observation components are:

1. `ego_in_inter`
2. `other_in_inter`
3. `ego_at_inter`
4. `close_ahead`
5. `ahead`
6. `higher_pri`

Their semantics are:

- `ego_in_inter`: the ego car is inside an intersection interior block
- `other_in_inter`: at least one other visible car or pedestrian is in the intersection
- `ego_at_inter`: the ego car is at an intersection entry region
- `close_ahead`: an entity is dangerously close ahead of ego
- `ahead`: an entity is ahead of ego in the same forward direction
- `higher_pri`: at least one visible car or pedestrian at or in the intersection has higher priority than ego

This observation is binary-valued in interpretation, although represented numerically as a `float32` vector.

An important distinction in the environment is the following:

- the simulator itself has richer grounded symbolic information
- the RL policy receives only the compact 6-bit symbolic summary
- the decentralized shield operates over that same compact symbolic observation
- the expert rule system evaluates the fuller grounded symbolic state

This distinction is important for the thesis because it explains why the decentralized method can remain limited even when the expert is structurally stronger.

### 4.2.5 Field of View and Partial Observability

The thesis environment is partially observable. The RL agent does not operate directly on the full global state. Instead, its observation is built from a local field of view.

The current thesis configuration uses:

- `obs_fov: 25`

This value is passed into the local planner through [CityLoader](/d:/Dev/LogicityFresh/LogiCity/logicity/utils/load.py:14), and then used by the local planner when cropping the world around the ego agent.

The maximum number of entities represented in the RL local symbolic grounding is:

- `fov_entities: Entity = 4`

This includes:

- the ego entity in slot `0`
- up to `3` additional visible entities in the remaining slots

If fewer than 3 non-ego entities are available, the environment inserts placeholder entities so that the observation dimensionality remains fixed. This is implemented in [z3_rl.py](/d:/Dev/LogicityFresh/LogiCity/logicity/planners/local/z3_rl.py:246).

Therefore, the RL interface is both:

- local
- fixed-width

This is a deliberate design choice. It preserves partial observability while keeping the observation vector stable for PPO-based learning.

### 4.2.6 Action Space

The current thesis environment uses the original 4-action macro-action structure:

| Action ID | Meaning |
| --- | --- |
| `0` | Fast |
| `1` | Normal |
| `2` | Slow |
| `3` | Stop |

The RL environment exposes a discrete action space:

- `action_space: 4`

The exact macro-action mapping is defined in the train configs and passed into the environment wrapper. The policy therefore outputs one of four discrete high-level speed decisions at each decision step.

All thesis baselines use the same action space. The difference between PPO and PPO+PLS is not the action set, but whether symbolic action reweighting is applied before action execution.

### 4.2.7 Rule Configuration

The thesis environment uses:

- ontology file: `config/rules/ontology_simple_weak.yaml`
- rule file: `config/rules/Nav/easy/expert_minimal_4action.yaml`
- rule type: `Z3_Expert`

This rule file contains three functionally distinct rule groups:

- `Sim` rules for the rule-based simulation behavior of background agents
- `Task` rules used for task-level reward and failure evaluation
- `Expert` rules used as the normative action reference

The main expert stop logic is:

1. stop if another entity is already in the intersection while ego is at the intersection entry
2. stop if another entity at the intersection has higher priority
3. stop if another car in the same intersection has higher priority while ego is inside
4. stop if another entity is close ahead

The expert `Slow` rule is:

- slow if stopping is not required but another entity is ahead

The expert `Fast` rule is:

- fast if stopping and slowing are not required and ego is neither at nor in an intersection

The expert `Normal` rule is:

- fallback action when none of the other action predicates apply

Thus, the environment contains a deterministic symbolic traffic policy that governs the background agents and also provides the expert comparison target used during evaluation.

### 4.2.8 Reward Function and Termination Conditions

The RL reward and episode logic are implemented in [gym_wrapper.py](/d:/Dev/LogicityFresh/LogiCity/logicity/utils/gym_wrapper.py:274) and [city_env.py](/d:/Dev/LogicityFresh/LogiCity/logicity/core/city_env.py:10).

The current real-train configurations use:

- `step_cost = 0`
- `progress_reward = 0.1`
- `goal_reward = 10`
- `overtime_cost = -6`
- action costs for the RL-controlled agent set to `0` for all 4 actions

So the learning reward is primarily shaped by:

- positive route progress
- goal completion
- overtime penalty
- failure penalties

#### Failure penalties

The current setup includes three relevant hard-failure categories:

1. `rule_based_fail`
2. `deadzone_fail`
3. `simultaneous_entry_fail`

The environment checks whether a stop-required condition was active and the ego failed to choose `Stop`. If so:

- the episode is marked as failed
- a penalty of `-10` is applied for that rule-based failure event

The environment also checks world-based hard failures after motion:

- deadzone failure
- simultaneous same-intersection entry failure

If either occurs:

- the episode is marked as failed
- a penalty of `-10` per such event is added

#### Success and termination

An episode is considered successful when:

- the ego vehicle reaches its goal

An episode terminates when one of the following occurs:

- goal reached
- hard failure
- maximum horizon reached

The current horizon is:

- `max_horizon = 150`

When the horizon is reached without success or failure:

- the episode terminates as overtime
- the overtime penalty is applied

### 4.2.9 Train/Validation/Test Scenario Generation

The thesis uses cached episode datasets rather than generating all evaluation episodes online. This improves reproducibility and ensures fair comparison between PPO and PLS.

The episode generators are:

- `tools/create_single_agent_two_agent_car_ped_region_episode.py`
- `tools/create_single_agent_two_agent_car_ped_close_ahead_episode.py`

The first generator creates general `rich` traffic-interaction episodes in the `3 cars + 1 pedestrian` scene family. The second generator creates structured `close_ahead` following scenarios within the same overall scene family.

The real splits are built by merging these scenario families into fixed cached datasets:

| Split | Cached file | Intended size | Scenario composition |
| --- | --- | ---: | --- |
| Train | `dataset/thesis/main_3cars_1ped/train_100_episodes.pkl` | `100` | rich + close-ahead |
| Validation | `dataset/thesis/main_3cars_1ped/val_20_episodes.pkl` | `20` | rich + close-ahead |
| Test | `dataset/thesis/main_3cars_1ped/test_50_episodes.pkl` | `50` | rich + close-ahead |

The validation and test generators can additionally save rollout world files for visualization, making qualitative inspection possible.

The RL training configs then load these datasets directly through:

- `training_episode_data` for training
- `episode_data` for validation
- `episode_data` in the eval config for final testing

This means PPO and PPO+PLS are trained and evaluated on the same cached worlds, which is important for fair method comparison.

### 4.2.10 RL Interface

The simulator is exposed to reinforcement learning through:

- [CityEnv](/d:/Dev/LogicityFresh/LogiCity/logicity/core/city_env.py)
- [GymCityWrapper](/d:/Dev/LogicityFresh/LogiCity/logicity/utils/gym_wrapper.py)

This wrapper stack provides:

- the flattened observation vector
- the discrete action interface
- reward computation
- success/failure signals
- expert action access for diagnostics
- grounded symbolic information used for evaluation and debugging

Therefore, the final thesis environment is a Gym-compatible RL environment built on top of LogiCity’s symbolic multi-agent world model.

### 4.2.11 Implementation Deviations from the Original LogiCity Setup

The thesis does not use the original LogiCity setup unchanged. Several important modifications were introduced:

| Component | Original LogiCity style | This thesis |
| --- | --- | --- |
| Task framing | Safe path following with symbolic grounding | adapted single-ego multi-agent safe path following |
| Agent composition | broader generic simulator support | fixed `3 cars + 1 pedestrian` thesis scene family |
| Observation | richer predicate grounding vectors | compact decentralized 6-bit symbolic observation |
| Field of view | local predicate-based observation | `obs_fov = 25`, fixed-width `4`-entity symbolic interface |
| Scenario generation | generic environment initialization | cached expert-generated `rich` and `close_ahead` scenarios |
| Failure semantics | original task penalties and rule semantics | combined rule-based, deadzone, and simultaneous-entry hard failures |
| Evaluation | standard success-oriented simulator evaluation | TSR, stop-decision alignment, failure breakdown, expert-policy diagnostics |

These deviations are not incidental implementation details. They are part of the thesis contribution, because they adapt LogiCity into an experimental environment specifically suited for studying decentralized symbolic safety shielding in multi-agent urban navigation.

## Summary

The configured LogiCity environment used in this thesis is a fixed-map, multi-agent, partially observable safe-path-following task with:

- one RL-controlled car
- two rule-based cars
- one rule-based pedestrian
- a 4-action speed-control interface
- a 6-bit decentralized symbolic observation
- explicit intersection and priority semantics
- expert-defined safety behavior
- cached train/validation/test episode datasets

This configuration provides the concrete world in which the PPO and PPO+PLS methods are compared in the experimental chapters.

