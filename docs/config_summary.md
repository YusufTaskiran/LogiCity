# LogiCity Config Summary

## Folder structure

- `config/maps/`: static map layout
- `config/agents/`: agent rosters for each split/difficulty
- `config/rules/ontology_*.yaml`: predicate sets used to build logical observations
- `config/rules/Nav/...`: navigation rules
  - `Sim`: deterministic policy for non-RL agents
  - `Task`: rules used to score and fail the RL agent
  - `Expert`: expert policy rules when `rule_type: Z3_Expert`
- `config/tasks/Nav/...`: top-level experiment configs tying map, agents, ontology, rules, and RL hyperparameters together

## High-level answer

The RL agent does **not** observe pixels. It observes a fixed-length binary/float vector of grounded logical predicates over a local field of view.

The policy action space is `Discrete(4)`:

- `0`: `Slow`
- `1`: `Normal`
- `2`: `Fast`
- `3`: `Stop`

Those 4 policy actions are mapped onto the car's lower-level 13-action motion model, where direction is still filtered by the car's route and local planner.

Reward is mainly:

- task-rule penalty from the logic engine
- movement cost by chosen speed mode
- overtime penalty if the horizon is exceeded

Episodes terminate on:

- goal reached
- horizon exceeded
- task-rule failure

## Map

All PPO navigation configs in this repo point to the same map:

- `config/maps/square_5x5.yaml`

This map defines:

- grid size `250 x 250`
- a repeated square-block city layout
- traffic streets and walking streets
- buildings placed inside each block

So the main difficulty changes are **not** different maps for PPO navigation. They come from:

- different agent sets
- different ontologies
- different task rules
- different concept reset distributions

## Observation space

The Gym wrapper exposes:

- `Box(low=0.0, high=1.0, shape=(logic_grounding_shape,))`
- or `logic_grounding_shape + 1` if `cat_length` is enabled

For the PPO configs here, `fov_entities.Entity = 5`, so the observation size depends on the ontology:

`observation_dim = (# unary state predicates * 5) + (# binary state predicates * 25)`

Action predicates such as `Stop`, `Slow`, `Normal`, and `Fast` are **not** included in the observation because they have `function: None`.

## Difficulty comparison

| Difficulty | PPO task file | Map | Ontology | State predicates used for observation | Observation dim | Policy actions | Horizon | Action cost `(slow, normal, fast, stop)` | Overtime | Task failure rules |
|---|---|---|---|---:|---:|---|---:|---|---:|---|
| Easy | `config/tasks/Nav/easy/algo/ppo.yaml` | `square_5x5` | `ontology_easy.yaml` | 7 unary, 2 binary | 85 | 4: slow/normal/fast/stop | 400 | `(-2, 0, -2, -5)` | `-6` | One stop-rule safety constraint |
| Medium | `config/tasks/Nav/medium/algo/ppo.yaml` | `square_5x5` | `ontology_medium.yaml` | 8 unary, 4 binary | 140 | 4: slow/normal/fast/stop | 400 | `(-2, 0, -2, -5)` | `-12` | One richer stop-rule safety/right-of-way constraint |
| Hard | `config/tasks/Nav/hard/algo/ppo.yaml` | `square_5x5` | `ontology_full.yaml` | 11 unary, 6 binary | 205 | 4: slow/normal/fast/stop | 400 | `(-2, 0, -2, -5)` | `-20` | One richer stop-rule safety/right-of-way constraint |
| Expert | `config/tasks/Nav/expert/algo/ppo.yaml` | `square_5x5` | `ontology_full.yaml` | 11 unary, 6 binary | 205 | 4: slow/normal/fast/stop | 200 | `(-2, -1, -2, -3)` | `-10` | Three task rules: `Stop`, `Slow`, and `Fast` |

## What each difficulty actually enforces

### Easy

Observation predicates:

- unary: `IsPedestrian`, `IsCar`, `IsAmbulance`, `IsOld`, `IsTiro`, `IsAtInter`, `IsInInter`
- binary: `HigherPri`, `CollidingClose`

Task failure is a single implication:

- if the agent is about to enter/interact unsafely at an intersection, or is `CollidingClose` to another entity, it must choose `Stop`

So in practice, `easy` is mostly:

- collision avoidance
- not entering intersections when blocked
- yielding to higher-priority traffic at intersections

### Medium

Adds more relational structure:

- new concepts like `IsBus`
- new spatial relations like `NextTo`, `RightOf`

The `Task` section still contains one failure rule, but it is broader than `easy`. It now also encodes rules such as:

- buses stopping for pedestrians in certain relative positions
- ambulances interacting with old pedestrians
- richer right-of-way logic

So `medium` is still "one dead safety rule", but the condition for when stopping is required is more complex.

### Hard

Uses the full ontology:

- adds `IsPolice`, `IsReckless`, `IsYoung`, `IsClose`, `LeftOf`, `RightOf`, `NextTo`

The active PPO `hard` task still has a single `Task` failure rule in `config/rules/Nav/hard/expert.yaml`, centered on `Stop(entity)`, but the stop condition is now richer and includes police/role-based traffic logic.

So `hard` is broader than plain collision checking, but it is still mostly a "must stop in these dangerous or rule-relevant situations" setup.

### Expert

This is the first setup where the RL task itself explicitly checks more than just stopping:

- `Stop` task rule
- `Slow` task rule
- `Fast` task rule

That means failure can come from:

- failing to stop when the rule requires stop
- failing to slow when the rule requires slow
- failing to fast when the rule requires fast

This is much closer to full traffic-rule compliance than the easier tasks.

## Important detail: active PPO configs use `expert.yaml`

The PPO training configs for `easy`, `medium`, and `hard` do **not** point to the `rl.yaml` rule files. They point to the `expert.yaml` files:

- `easy/algo/ppo.yaml` -> `config/rules/Nav/easy/expert.yaml`
- `medium/algo/ppo.yaml` -> `config/rules/Nav/medium/expert.yaml`
- `hard/algo/ppo.yaml` -> `config/rules/Nav/hard/expert.yaml`
- `expert/algo/ppo.yaml` -> `config/rules/Nav/expert/expert.yaml`

So if you want to understand the active PPO training behavior, those `expert.yaml` task-rule files are the right ones to inspect first.

## Agent setup

The RL-controlled agent is configured by `rl_agent.agent_name`, typically:

- `Car_1`

The surrounding traffic differs by difficulty through the agent files:

- `config/agents/easy/...`
- `config/agents/medium/...`
- `config/agents/hard/...`
- `config/agents/expert/...`

The reset distributions also change the RL car's concepts across difficulties:

- easy: `ambulance`, `tiro`, `normal`
- medium: `ambulance`, `tiro`, `bus`, `normal`
- hard: `ambulance`, `bus`, `police`, `normal`
- expert: `ambulance`, `bus`, `police`, `reckless`, `tiro`, `normal`

## Reward shaping summary

Per step, the wrapper computes:

- if the step fails a task rule: use the raw task penalty directly
- otherwise: `(action_cost + task_rule_reward) / path_length`

There is no explicit positive terminal reward for success in the wrapper. The shaping is mostly:

- avoid violating logical rules
- avoid expensive speed choices
- avoid taking too long

## Failure summary

If you want the shortest possible interpretation:

- `easy`: failure is basically collision/intersection stop-rule violation
- `medium`: failure is still one stop-rule violation, but with richer right-of-way and role-based logic
- `hard`: same pattern, with the fullest stop-condition logic short of full multi-action compliance
- `expert`: failure includes broader traffic-rule compliance over `Stop`, `Slow`, and `Fast`

## Most relevant files

- `config/tasks/Nav/easy/algo/ppo.yaml`
- `config/tasks/Nav/medium/algo/ppo.yaml`
- `config/tasks/Nav/hard/algo/ppo.yaml`
- `config/tasks/Nav/expert/algo/ppo.yaml`
- `config/rules/ontology_easy.yaml`
- `config/rules/ontology_medium.yaml`
- `config/rules/ontology_full.yaml`
- `config/rules/Nav/easy/expert.yaml`
- `config/rules/Nav/medium/expert.yaml`
- `config/rules/Nav/hard/expert.yaml`
- `config/rules/Nav/expert/expert.yaml`
- `logicity/utils/gym_wrapper.py`
- `logicity/planners/local/z3_rl.py`
- `logicity/planners/local/z3_expert.py`
- `logicity/agents/car.py`
