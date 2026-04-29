# Thesis Experiment Setup

This document describes the current experiment setup used for the single-agent navigation thesis experiments on the `main_3cars_1ped` splits, using the minimal 4-action expert and the current `PPO` / `PPO + PLS` implementations.

## Active Training Configs

- PPO: `config/tasks/Nav/thesis/algo/ppo_single_train.yaml`
- PPO + PLS: `config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml`
- Ontology: `config/rules/ontology_simple.yaml`
- Expert rules: `config/rules/Nav/easy/expert_minimal_4action.yaml`

## Observation Space

### Export Format

The RL observation is a flattened logical grounding vector exported by the local Z3 planner. The wrapper exposes it as a `Box(low=0, high=1, shape=(logic_grounding_shape,))`.

The grounding size is computed in `logicity/planners/local/z3_rl.py` by grounding:

- every unary predicate once per entity slot
- every binary predicate once per ordered entity pair

The current configs use:

- `fov_entities.Entity = 4`
- `grounding_mode = "full"` implicitly, because neither `ego_only_grounding` nor `compact_relational` is enabled

### Vector Size

Current ontology predicates used as observation features:

- Unary predicates:
  - `IsPedestrian`
  - `IsCar`
  - `IsAtInter`
  - `IsInInter`
- Binary predicates:
  - `HigherPri`
  - `CollidingClose`
  - `IsCloseAhead`

With `4` entity slots:

- Unary part: `4 predicates * 4 entities = 16`
- Binary part: `3 predicates * 4 * 4 = 48`
- Total observation size: `64`

Action symbols (`Stop`, `Fast`, `Normal`, `Slow`) are listed in the ontology file, but they are not exported as grounded observation features because they have no backing function and are skipped by the grounding-shape code.

### Predicate Semantics

#### `IsPedestrian(x)`

- Returns `1` if the entity name contains `"Pedestrian"`.
- Returns `0` for placeholder entities (`"PH"`) and for non-pedestrians.

This is a pure type predicate based on the grounded entity label, not on motion or geometry.

#### `IsCar(x)`

- Returns `1` if the entity name contains `"Car"`.
- Returns `0` for placeholders and non-cars.

Like `IsPedestrian`, this is a pure type predicate.

#### `IsAtInter(x)`

This checks whether the current position of entity `x` lies on the "at intersection" mask.

Implementation details:

- The agent position is read from its layer in `world_matrix`.
- Cars use `intersect_matrix[0]`.
- Pedestrians use `intersect_matrix[1]`.

Interpretation:

- For cars, this is the boundary / entry region where the agent is considered to be at the intersection.
- For pedestrians, the corresponding pedestrian intersection mask is used.

This predicate is used heavily in both the expert rules and the PLS stop logic.

#### `IsInInter(x)`

This checks whether the current position of entity `x` lies inside the actual intersection interior.

Implementation details:

- The agent position is read from its layer in `world_matrix`.
- The check uses `intersect_matrix[2]` for all agent types.

Interpretation:

- `IsAtInter` means the agent is at the intersection boundary / decision area.
- `IsInInter` means the agent is already occupying the intersection itself.

#### `HigherPri(x, y)`

This checks whether entity `x` has higher priority than entity `y`.

Implementation details:

- The predicate reads each agent’s `priority` field from the runtime `agents` dictionary.
- It returns `1` when `priority(x) < priority(y)`.

So numerically smaller priority values mean higher priority.

This is used in both:

- expert stop conditions
- PLS waiting-conflict rules

#### `CollidingClose(x, y)`

This is the immediate hazard predicate.

It is triggered as follows:

1. Ignore self-pairs and placeholder entities.
2. Read current positions of `x` and `y`.
3. Read the current moving direction of `x`.
4. Compute Euclidean distance between `x` and `y`.
5. Reject if distance is larger than `OCC_CHECK_RANGE[type(x)]`.
6. If distance is zero:
   - return a random collision flag
   - `50%` chance for `0/1`
7. Otherwise:
   - compute the angle between the heading of `x` and the relative vector from `x` to `y`
   - trigger only if `angle < OCC_CHECK_ANGEL`

Current constants:

- `OCC_CHECK_RANGE["Car"] = 8`
- `OCC_CHECK_RANGE["Pedestrian"] = 3`
- `OCC_CHECK_ANGEL = 0.1`
- `PED_AGGR = 0.5`

Type-specific behavior:

- If `x` is a car and the geometric test passes, `CollidingClose = 1`.
- If `x` is a pedestrian and the geometric test passes, it triggers stochastically with probability `0.5`.

Interpretation:

- This is a short-range, forward-cone, immediate conflict indicator.
- It is not generic proximity. It is direction-aware and deliberately stricter than `IsCloseAhead`.

#### `IsCloseAhead(x, y)`

This is the early warning predicate added as a softer precursor to `CollidingClose`.

It is triggered as follows:

1. Ignore self-pairs and placeholder entities.
2. Read positions of `x` and `y`.
3. Read the current moving direction of `x`.
4. Compute distance `dist(x, y)`.
5. Define:
   - `close_min = OCC_CHECK_RANGE[type(x)]`
   - `close_max = OCC_CHECK_RANGE[type(x)] * AHEAD_CLOSE_RANGE_SCALE`
6. Reject unless:
   - `dist > close_min`
   - and `dist <= close_max`
7. Compute the angle between the heading of `x` and the relative vector from `x` to `y`.
8. Trigger only if:
   - `angle < OCC_CHECK_ANGEL * AHEAD_CLOSE_ANGLE_SCALE`

Current constants:

- `AHEAD_CLOSE_RANGE_SCALE = 2.0`
- `AHEAD_CLOSE_ANGLE_SCALE = 2.0`
- `OCC_CHECK_ANGEL = 0.1`

So the effective warning band is:

- Cars:
  - `CollidingClose`: `dist <= 8`, angle `< 0.1`
  - `IsCloseAhead`: `8 < dist <= 16`, angle `< 0.2`
- Pedestrians:
  - `CollidingClose`: `dist <= 3`, angle `< 0.1`
  - `IsCloseAhead`: `3 < dist <= 6`, angle `< 0.2`

Interpretation:

- `IsCloseAhead` is a medium-range, direction-aware forward warning zone.
- It was introduced specifically so the policy and shield can represent "slow down first" before the sharper `CollidingClose` stop condition.

## Action Space

### RL Macro Actions

The RL agent has `4` macro actions:

- `0 = slow`
- `1 = normal`
- `2 = fast`
- `3 = stop`

This comes from the training configs:

- `action_space: 4`
- `action_mapping:`
  - `0 -> [1,1,1,1,0,0,0,0,0,0,0,0,0]`
  - `1 -> [0,0,0,0,1,1,1,1,0,0,0,0,0]`
  - `2 -> [0,0,0,0,0,0,0,0,1,1,1,1,0]`
  - `3 -> [0,0,0,0,0,0,0,0,0,0,0,0,1]`

### Underlying Car Primitive Actions

The car agent internally uses `13` primitive actions:

- `0: left_1`
- `1: right_1`
- `2: up_1`
- `3: down_1`
- `4: left_2`
- `5: right_2`
- `6: up_2`
- `7: down_2`
- `8: left_3`
- `9: right_3`
- `10: up_3`
- `11: down_3`
- `12: stop`

And the corresponding displacement vectors are:

- Slow primitives:
  - left/right/up/down by `1` cell
- Normal primitives:
  - left/right/up/down by `2` cells
- Fast primitives:
  - left/right/up/down by `3` cells
- Stop:
  - no movement

### How Macro Actions Move the Agent

Each macro action does not directly encode a direction. Instead, it selects a speed tier:

- `slow`
  - enables only the 1-cell primitives
- `normal`
  - enables only the 2-cell primitives
- `fast`
  - enables only the 3-cell primitives
- `stop`
  - enables only the stop primitive

The environment then resolves the allowed primitive move relative to the agent’s current route and local planner output.

Practical interpretation:

- `slow` means advance one grid cell along a valid route direction
- `normal` means advance two cells if route-consistent and legal
- `fast` means advance three cells if route-consistent and legal
- `stop` means remain in place

Cars also reject multi-cell moves that would improperly bypass intersection boundary points through the `move_bypass()` check.

## Rewards

### Training / Evaluation Reward Settings

Current thesis train and eval configs use:

- `step_cost = 0`
- `progress_reward = 0.1`
- `goal_reward = 10`
- `action_cost[slow] = 0`
- `action_cost[normal] = 0`
- `action_cost[fast] = 0`
- `action_cost[stop] = 0`
- `overtime_cost = -6`

### Reward Computation in the Gym Wrapper

At each environment step, the wrapper computes:

1. Progress term
   - `progress_reward * max(route_index_delta, 0)`
   - route progress is measured by how far the ego advanced along `global_traj`
2. Base SAT/task reward from `obs_dict["Reward"][0]`
3. Action cost
   - currently always `0` in the active configs
4. Step cost
   - currently always `0`

If the step is a failure step:

- reward is:
  - `obs_dict["Reward"][0] + step_cost + progress_term`
- this branch deliberately skips path-length normalization

If the step is not a failure step:

- reward is:
  - `step_cost + progress_term + (action_cost + obs_dict["Reward"][0]) / path_length`

Episode-end modifiers:

- Success:
  - add `goal_reward = 10`
- Overtime in wrapper (`t >= horizon`):
  - add `overtime_cost = -6`
- In `main.py` evaluation loop only:
  - if `step >= max_steps`, an extra `-3` is applied and the episode is marked truncated

### Task-Level Failure Penalty

The expert rule file also includes a task rule with:

- `reward: -10`
- `dead: true`

for collision / intersection-rule violations.

So in practice, unsafe behavior can produce:

- a task-level negative reward from rule evaluation
- episode termination due to failure

## Expert Rules

The active expert policy is defined in `config/rules/Nav/easy/expert_minimal_4action.yaml`.

It uses only these predicates:

- `IsAtInter`
- `IsInInter`
- `HigherPri`
- `CollidingClose`
- `IsCloseAhead`

### Rule 1: `Stop`

```text
Stop(entity) == Exists(dummyEntity,
  Or(
    IsAtInter(entity) & IsInInter(dummyEntity),
    IsAtInter(entity) & IsAtInter(dummyEntity) & HigherPri(dummyEntity, entity),
    CollidingClose(entity, dummyEntity)
  )
)
```

This means the expert stops if there exists another entity such that at least one of the following is true:

1. Ego is at the intersection and another entity is already in the intersection.
2. Ego is at the intersection and another entity is also at the intersection and has higher priority.
3. Another entity is in immediate close forward conflict with ego.

Interpretation:

- The first clause is the occupancy rule.
- The second clause is the higher-priority waiting rule.
- The third clause is the emergency immediate-hazard rule.

### Rule 2: `Slow`

```text
Slow(entity) == Not(Stop(entity)) &
                Exists(dummyEntity, IsCloseAhead(entity, dummyEntity))
```

If the ego does not need to stop, but there is at least one early forward warning (`IsCloseAhead`), the expert chooses `Slow`.

Interpretation:

- This is the "warning zone" behavior.
- It is explicitly a softer predecessor to the stop rule.

### Rule 3: `Fast`

```text
Fast(entity) == Not(Stop(entity)) &
                Not(Slow(entity)) &
                Not(IsAtInter(entity)) &
                Not(IsInInter(entity))
```

If the ego:

- does not need to stop
- does not need to slow
- is not at the intersection
- is not in the intersection

then the expert chooses `Fast`.

Interpretation:

- `Fast` is only allowed on open road away from the intersection decision region.

### Rule 4: `Normal`

```text
Normal(entity) == Not(Stop(entity)) &
                  Not(Slow(entity)) &
                  Not(Fast(entity))
```

This is the fallback rule.

Interpretation:

- If none of the stronger conditions apply, the expert chooses `Normal`.
- In practice, this covers states that are not dangerous, but are still close enough to intersection structure that the expert does not want `Fast`.

## Methods

## PPO

The PPO baseline uses:

- algorithm: `PPO`
- policy: `MlpPolicy`
- feature extractor: `MLPFeatureExtractor`
- feature dimension: `32`

Current hyperparameters:

- learning rate: `3e-4`
- clip range: `0.2`
- value loss coefficient: `0.5`
- batch size: `64`
- rollout steps: `256`
- epochs per update: `10`
- entropy coefficient: `0.001`
- total timesteps: `30000`

In the active setup:

- PPO receives the `64D` grounded observation vector
- samples one of the four macro actions
- the action is mapped to the speed-tier primitive set
- training reward is the reward defined above

## PPO + PLS

### High-Level Idea

`PLSPPO` augments PPO with a probabilistic logic shield.

The current implementation uses:

- a compiled joint ProbLog program
- the current base policy probabilities as an annotated disjunction
- the observed symbolic facts as state facts
- logic rules that define action safety

The shield is enabled:

- at train time
- at test time

### Shield Configuration

Current settings:

- `backend = "problog"`
- `enabled = true`
- `train_time = true`
- `test_time = true`
- `graded_safety = true`
- `shield_temperature = 1.0`
- `use_privileged_internal_safety = false`
- `stop_priority_scale = 1.0`
- `safety_coefficient = 0.1`
- `use_safety_loss = true`
- `action_space = ["slow", "normal", "fast", "stop"]`
- `safe_fallback_action = "stop"`

The current PLS shield uses a graded safety model with four regimes:

- `must_stop`
- `warning`
- `fast_zone`
- `normal_zone`

Each regime assigns soft safety values to the four actions, and the shield reweights the base policy according to those values.

Because `stop_priority_scale = 1.0`, the extra anti-stop heuristic is disabled, so no additional hand-crafted movement preference is injected on top of the graded ProbLog safety model.

### Safety Model

For the current old-observation setup, the shield extracts these state facts from the observation:

- `ego_at_inter`
- `ego_in_inter`
- for each other entity `i`:
  - `higher_pri_i`
  - `other_in_inter_i`
  - `other_at_inter_i`
  - `colliding_close_i`
  - `close_ahead_i`

These facts are read directly from the observation entries as probabilities clipped to `[0, 1]`.

In the current thesis setup, the grounded observation is still effectively binary, so these state facts behave like crisp logical inputs in practice. However, the shield no longer thresholds them internally, which keeps the implementation compatible with future probabilistic fact inputs.

### Safety Rules in the PLS Model

The current PLS logic is derived from the same minimal predicate structure as the four-action expert, but it uses a graded safety model instead of a deterministic one-hot safe-action assignment.

Derived conflict facts:

- `occupancy_conflict_i :- ego_at_inter, other_in_inter_i.`
- `waiting_conflict_i :- ego_at_inter, higher_pri_i, other_at_inter_i.`
- `close_conflict_i :- colliding_close_i.`
- `warning_close_i :- close_ahead_i.`

These are combined into higher-level regimes:

- `must_stop`
  - true if any occupancy conflict, waiting conflict, or close conflict holds
- `warning_state`
  - true if `must_stop` is false and at least one `warning_close_i` holds
- `fast_zone_state`
  - true if `must_stop` is false, `warning_state` is false, and ego is neither at nor in the intersection
- `normal_zone_state`
  - true if none of the above regimes applies

Instead of declaring only one action safe, the PLS model assigns graded safety values to all four actions in each regime.

Current graded safety values are:

- `must_stop`
  - `Slow = 0.05`
  - `Normal = 0.01`
  - `Fast = 0.0`
  - `Stop = 1.0`
- `warning_state`
  - `Slow = 0.95`
  - `Normal = 0.55`
  - `Fast = 0.10`
  - `Stop = 1.0`
- `fast_zone_state`
  - `Slow = 0.65`
  - `Normal = 0.85`
  - `Fast = 0.98`
  - `Stop = 1.0`
- `normal_zone_state`
  - `Slow = 0.80`
  - `Normal = 0.97`
  - `Fast = 0.25`
  - `Stop = 1.0`

So the current PLS no longer mirrors the expert as a hard one-hot rule system. Instead, it preserves the same expert-inspired regime structure while expressing safety probabilistically.

### Shielded Policy

The ProbLog program includes:

- a policy annotated disjunction over the four actions
- the grounded state facts
- the safety rules

From that same compiled program it queries:

- action safety
- policy safety
- joint action-and-safe probabilities

The implementation then forms the shielded policy as:

- `pi_plus(a|s) propto pi(a|s) * P(safe | s, a)`

In the current implementation, the raw safety values are additionally modulated by a shield temperature before reweighting:

- `adjusted_safe(a|s) = P(safe | s, a)^(1 / T)`

and the final shielded policy becomes:

- `pi_plus(a|s) propto pi(a|s) * adjusted_safe(a|s)`

where `T` is the shield temperature.

Interpretation:

- `T = 1.0` leaves the safety values unchanged
- `T > 1.0` softens the shield by pushing safety values closer to `1`
- `T < 1.0` sharpens the shield by pushing low safety values closer to `0`

So the current PLS does not simply hard-mask actions. It first computes graded action safety in ProbLog, optionally softens or sharpens those values with the temperature parameter, and then uses them to reweight the base policy.

### Safety Term in the Loss

Training uses two PLS-specific ingredients:

1. Shielded policy gradient
   - PPO training computes `log pi_plus(a|s)` from the shielded action distribution
2. Safety loss
   - `safety_loss = -log P_{pi_plus}(safe | s)`

Current implementation detail:

- the safety model output is treated as fixed with respect to the policy
- `safe_probs` are detached
- but `pi_plus` is built in Torch from:
  - `base_probs * safe_probs`
- so gradients flow through the reweighting and normalization of the shielded policy

Total training loss:

- `policy_loss`
- `+ ent_coef * entropy_loss`
- `+ vf_coef * value_loss`
- `+ safety_coefficient * safety_loss`

with `safety_coefficient = 0.1`.

### Example Shielding Workflow

Consider a state where:

- `ego_at_inter = 1`
- `higher_pri_3 = 1`
- `other_in_inter_3 = 1`

This triggers:

- `must_stop`

Under the current graded PLS model, this regime assigns:

- `safe_probs = [0.05, 0.01, 0.0, 1.0]`

meaning:

- `Slow` is strongly discouraged
- `Normal` is even more strongly discouraged
- `Fast` is completely unsafe
- `Stop` is maximally safe

If the base policy is approximately uniform:

- `base_probs = [0.25, 0.25, 0.25, 0.25]`

then before renormalization the shield computes:

- weighted probabilities = `[0.0125, 0.0025, 0.0, 0.25]`

and after renormalization the shielded policy becomes approximately:

- `shielded_probs ~= [0.047, 0.009, 0.0, 0.944]`

If shield temperature `T` is used, these safety values are first adjusted by:

- `adjusted_safe(a|s) = P(safe | s, a)^(1/T)`

and then the same reweighting and renormalization is applied.

So in this required-stop state:

- PPO alone may still put mass on unsafe move actions
- PPO + PLS pushes most probability mass onto `Stop`
- `Fast` can be completely eliminated when its safety is `0`
- and the safety loss penalizes low `P_{pi_plus}(safe | s)`

## Summary

The current thesis setup uses:

- a `64D` grounded symbolic observation
- `4` macro actions mapped to `1/2/3`-cell motion tiers plus stop
- reward shaping with progress reward, goal reward, and overtime penalty
- a minimal 4-action expert using only:
  - `IsAtInter`
  - `IsInInter`
  - `HigherPri`
  - `CollidingClose`
  - `IsCloseAhead`
- a PLS model whose current safety logic is intentionally aligned with that expert

This means the present `PPO + PLS` implementation is best understood as:

- PPO trained on the same observation and reward signal as PPO
- plus a compiled joint ProbLog shield
- plus a safety loss term
- with crisp grounded input facts in the current experiments
- and a graded action-safety model that reweights the policy probabilistically
