# Methodology Draft

## 1. Problem Setting

This thesis studies whether symbolic safety structure can improve reinforcement learning for urban navigation in a multi-agent environment. The central task is a controlled driving problem in which one learning-based ego car must reach its route goal while interacting with other traffic participants at intersections and in short-range following situations. The focus is not only task completion, but also rule compliance and coordination safety.

The environment is multi-agent at the world level, but the main training setup is single-controlled-agent learning:

- `Car_1` is the RL-controlled ego vehicle.
- the remaining cars and pedestrians are rule-based agents driven by the expert local planner.
- the world evolves jointly, so the learning problem remains interactive and non-stationary from the ego agent’s perspective.

This setup isolates whether a shielded policy can improve the ego agent’s decision making without requiring all agents to be learned jointly.

## 2. Action Model

The ego vehicle acts in a discrete 4-action macro-action space:

- `0 = Fast`
- `1 = Normal`
- `2 = Slow`
- `3 = Stop`

These macro-actions map to low-level motion primitives already used by the environment. The shield operates directly over this 4-action policy output.

## 3. Expert Rule System

The environment includes an expert symbolic controller expressed through first-order rules over grounded predicates. The expert is used in two roles:

- as the controller for non-learning background agents
- as the normative decision source for dataset generation and evaluation diagnostics

The current expert stop rule is:

1. stop when ego is at an intersection and another entity is already in the intersection
2. stop when ego is at an intersection and another entity at the intersection has higher priority
3. stop when ego is in the intersection and another car in the intersection has higher priority
4. stop when another entity is close ahead

Formally, the expert stop action is triggered by:

- `IsAtInter(entity) && IsInInter(other)`
- `IsAtInter(entity) && IsAtInter(other) && HigherPri(other, entity)`
- `IsCar(entity) && IsInInter(entity) && IsInInter(other) && IsCar(other) && HigherPri(other, entity)`
- `IsCloseAhead(entity, other)`

The expert also defines the fallback motion classes:

- `Slow` when stopping is not required but an `ahead` condition exists
- `Fast` when no stopping or caution condition applies and ego is not at or in an intersection
- `Normal` otherwise

This expert policy is deterministic and symbolic. It is not learned.

## 4. Decentralized PLS Design

### 4.1 Motivation

A direct symbolic shield over the full relational state is not desirable for the thesis argument, because it gives the agent access to too much structured global information. Instead, the decentralized PLS design deliberately compresses the local world into a small observation that preserves the key interaction facts while still losing some structure. This enables a meaningful comparison:

- plain PPO must infer useful behavior from reward and interaction alone
- decentralized PLS receives symbolic local structure
- decentralized PLS can still fail in cases where local aggregation is insufficient

### 4.2 Minimal Observation

The decentralized shield uses a fixed 6-bit symbolic observation:

- `ego_in_inter`
- `other_in_inter`
- `ego_at_inter`
- `close_ahead`
- `ahead`
- `higher_pri`

These bits are computed from the RL agent’s grounded local observation. Their semantics are:

- `ego_in_inter`: ego is currently inside an intersection interior block
- `ego_at_inter`: ego is currently at an intersection entry region
- `other_in_inter`: at least one other relevant traffic participant is in the intersection
- `close_ahead`: some other entity is dangerously close ahead
- `ahead`: some other entity is ahead in the same forward direction
- `higher_pri`: at least one other relevant entity at or in the intersection has higher priority than ego

The current implementation aggregates both cars and pedestrians into:

- `other_in_inter`
- `higher_pri`

This alignment was necessary because the expert can require stopping due to pedestrian intersection occupancy and pedestrian priority interactions.

### 4.3 Local Visibility

The RL agent uses:

- `obs_fov = 25`

This field of view is large enough to expose the full relevant local intersection geometry in the thesis map, while still preserving the decentralized design principle: the shield reasons only over compact local facts, not over full joint world state.

## 5. Probabilistic Logic Shield

### 5.1 Shield Type

The shield is implemented as a ProbLog-based probabilistic logic shield. It does not output a single hard action. Instead, it computes graded safety preferences over the 4 discrete actions and uses them to reshape the base policy distribution.

The implemented shield therefore combines:

- symbolic regime recognition
- fixed probabilistic action safety priors
- learned PPO action probabilities

### 5.2 Regime Structure

The decentralized PLS partitions the local symbolic state into four regimes:

- `must_stop`
- `warning`
- `fast_zone`
- `normal_zone`

The current stop and warning rules are:

#### Must-stop rules

- `must_stop :- close_ahead.`
- `must_stop :- ego_at_inter, other_in_inter.`
- `must_stop :- ego_at_inter, higher_pri.`
- `must_stop :- ego_in_inter, other_in_inter, higher_pri.`

#### Warning rules

- `warning_state :- \+ must_stop, ahead.`
- `warning_state :- \+ must_stop, ego_at_inter.`
- `warning_state :- \+ must_stop, ego_in_inter, other_in_inter.`

#### Fast and normal rules

- `fast_zone_state :- \+ must_stop, \+ warning_state, \+ ego_at_inter, \+ ego_in_inter.`
- `normal_zone_state :- \+ must_stop, \+ warning_state, \+ fast_zone_state.`

These rules were designed to align the decentralized shield with the expert stop logic as closely as possible under local aggregation.

### 5.3 Graded Safety Preferences

Each regime assigns a graded safety score to every action. In the current implemented setup:

#### Must-stop

- `fast: 0.0`
- `normal: 0.01`
- `slow: 0.05`
- `stop: 1.0`

#### Warning

- `fast: 0.10`
- `normal: 0.65`
- `slow: 1.00`
- `stop: 0.45`

#### Fast-zone

- `fast: 1.00`
- `normal: 0.80`
- `slow: 0.55`
- `stop: 0.25`

#### Normal-zone

- `fast: 0.35`
- `normal: 1.00`
- `slow: 0.75`
- `stop: 0.40`

These are not learned from data. They are manually specified safety preferences. This is an intentional modeling choice: the shield represents symbolic prior structure, while PPO learns under that prior.

### 5.4 Policy-Shield Interaction

Let the PPO policy output a base action distribution over the 4 actions. The shield computes action safety scores from the current symbolic regime. These scores are combined with the base distribution through temperature-controlled reweighting.

The current configuration uses:

- ProbLog backend
- graded safety enabled
- shield temperature `1.0`
- safety loss enabled during training

The resulting shielded policy is therefore not a pure hard constraint, nor a pure learned policy. It is a hybrid:

- the base policy proposes actions
- the symbolic shield reshapes those action probabilities
- PPO is trained on the resulting interaction dynamics

This design lets the thesis analyze whether symbolic structure improves sample efficiency and safety without removing learning entirely.

## 6. Failure Semantics

The thesis setup uses hard failure semantics that combine symbolic rule violation with physical conflict events.

An episode is treated as failed if either of the following occurs:

### 6.1 Rule-based hard fail

If the current stop-required condition is true and the RL agent does not choose `Stop`, the episode incurs a hard failure.

This is computed from the pre-move world state using the current global stop-required logic.

### 6.2 Physical/coordination hard fail

Two world-based failure events are also active:

- `deadzone` failure: the ego enters another agent’s dangerous forward proximity region
- `simultaneous_entry` failure: multiple agents enter the same labeled intersection block on the same step

These failures are computed from world transitions after movement.

This combined failure definition is important for the methodology:

- rule-based failure captures symbolic safety violations
- deadzone and simultaneous-entry capture physical and coordination hazards

## 7. Dataset Construction Method

The thesis dataset is not generated from random uncontrolled rollouts. Instead, episodes are constructed offline through expert-driven scenario generation and then cached as reusable episode initial states.

### 7.1 Scene Family

The final thesis scene family contains:

- `1` RL ego car
- `2` rule-based cars
- `1` pedestrian

This keeps the environment interaction-rich while still being small enough to diagnose failures.

### 7.2 Scenario Classes

The real dataset uses two scenario families:

- `rich`
- `close_ahead`

#### Rich

General multi-agent intersection episodes with the standard `3 cars + 1 pedestrian` scene composition.

#### Close-ahead

A specialized variant of the same scene family where:

- the RL car is placed behind another car in the same lane
- the geometry is chosen so a close-following situation is likely
- the rollout must realize the intended close-ahead behavior

The close-ahead generator was refined to:

- avoid invalid spawn overlaps
- keep same-lane following geometry
- preserve consistent route structure

### 7.3 Expert-Guided Filtering

Episodes are accepted only if the expert rollout satisfies the generator’s intended constraints. This ensures that the cached dataset contains meaningful, controlled interaction cases rather than arbitrary random worlds.

## 8. Multi-Agent Extension

The environment is fundamentally multi-agent, and the methodology naturally extends beyond the single-controlled-agent training case.

### 8.1 Decentralized Multi-Agent Interpretation

In the decentralized interpretation, each controlled agent would receive:

- its own local 6-bit symbolic observation
- its own local PLS reasoning process
- no explicit joint tie-breaking signal

This means the same shield can be applied independently to multiple agents, but only through their local views.

### 8.2 Shared-Policy Evaluation

The codebase also supports shared-policy multi-agent evaluation, where the same trained policy can be executed by multiple controlled cars. This provides a bridge from single-agent training to decentralized multi-agent testing.

### 8.3 Joint Extension

A natural extension is a joint PLS formulation in which additional coordination information is supplied beyond the 6-bit decentralized observation. The thesis motivation for this extension is clear:

- decentralized PLS improves asymmetric local cases
- decentralized PLS can still fail when local aggregation is insufficient for coordination
- a joint shield can explicitly encode the missing coordination structure

The methodology developed here therefore serves as the decentralized base case against which richer joint reasoning can later be compared.

## 9. Summary of the Methodological Contribution

The final methodological contribution of the thesis is a complete end-to-end framework that combines:

- a relational expert rule system
- a compact decentralized symbolic observation
- a probabilistic logic shield over a 4-action driving policy
- hybrid symbolic and physical failure semantics
- controlled expert-generated multi-agent datasets
- PPO-based policy learning under symbolic action shaping

The design is intentionally balanced between structure and limitation:

- it provides enough logic to improve safety-relevant decision making
- but it remains lossy and decentralized enough that important coordination limitations can still be studied empirically

