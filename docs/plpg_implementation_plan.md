# PLPG Implementation Plan for LogiCity Safe Path Following

## Goal

Implement **PLPG** for LogiCity Safe Path Following, starting with `easy-lite`.

The intended behavior is:

- the neural policy produces a base action distribution `π(a|s)`
- a probabilistic logic shield computes action safety scores
- the base distribution is transformed into a shielded distribution `π⁺(a|s)`
- the agent acts using `π⁺`
- training is updated to learn through `π⁺`
- a safety term can later be added to the loss

This document is an engineering plan for how to do that in the current codebase.

## Current Integration Points

The current PPO training path is:

1. `main.py`
   - loads the YAML config
   - creates `GymCityWrapper`
   - instantiates the RL algorithm from `logicity.rl_agent.alg`
2. `GymCityWrapper.step()`
   - receives a discrete action index
   - maps it to the 13-dimensional one-hot action vector used by the car
   - calls `self.env.move_rl_agent(...)`
3. `CityEnv.move_rl_agent()`
   - checks task failure via the local planner
   - applies the selected action
4. PPO learns through Stable-Baselines3

Relevant files:

- [main.py](D:/Dev/LogicityFresh/LogiCity/main.py)
- [logicity/utils/gym_wrapper.py](D:/Dev/LogicityFresh/LogiCity/logicity/utils/gym_wrapper.py)
- [logicity/core/city_env.py](D:/Dev/LogicityFresh/LogiCity/logicity/core/city_env.py)
- [logicity/rl_agent/alg/__init__.py](D:/Dev/LogicityFresh/LogiCity/logicity/rl_agent/alg/__init__.py)
- [config/tasks/Nav/easy/algo/ppo_lite_eval.yaml](D:/Dev/LogicityFresh/LogiCity/config/tasks/Nav/easy/algo/ppo_lite_eval.yaml)

## Design Choice

The recommended strategy is:

- keep the environment mostly unchanged
- keep PPO as the base actor-critic backbone
- add a **PLPG shield layer** between PPO’s policy distribution and the sampled environment action

This is lower risk than trying to rewrite the environment logic or replace the planner stack entirely.

## What We Will Implement First

Start with `easy-lite` only.

Scope for the first prototype:

- binary state predicates
- 4 discrete actions: `Slow`, `Normal`, `Fast`, `Stop`
- ProbLog safety program for the `easy` task rule
- shielded action selection
- PPO learning through the shielded policy

Not in the first prototype:

- medium/hard support
- exact continuous-action extensions
- full multi-difficulty ProbLog templates
- optimized compiled-circuit caching beyond what is necessary to validate the method

## Architecture

### 1. New PLPG Shield Module

Add a new module, for example:

- `logicity/plpg/shield.py`
- `logicity/plpg/program_builder.py`
- `logicity/plpg/state_adapter.py`

Responsibilities:

#### `state_adapter.py`

Convert the current Logicity state into the facts needed by the PLPG safety program.

For `easy-lite`, these facts can come from the same symbolic conditions already used by the task rule:

- `is_at_inter`
- `is_in_inter`
- `higher_pri`
- `colliding_close`

The adapter should expose a compact state object, for example:

```python
{
    "IsAtInter": 0 or 1,
    "IsInInter_dummy0": 0 or 1,
    "HigherPri_dummy0": 0 or 1,
    "CollidingClose_dummy0": 0 or 1,
    ...
}
```

For the first version, these can be binary facts.

#### `program_builder.py`

Build the fixed ProbLog template for `easy-lite`.

Inputs:

- policy probabilities over 4 actions
- binary safety facts for the current state

Outputs:

- a ProbLog program string or compiled object capable of querying:
  - `P(safe | s, action=Slow)`
  - `P(safe | s, action=Normal)`
  - `P(safe | s, action=Fast)`
  - `P(safe | s, action=Stop)`
  - `Pπ(safe | s)`

#### `shield.py`

Given:

- base action probabilities `π(a|s)`
- current state facts

compute:

- `P(safe | s, a)` for each action
- `Pπ(safe | s)`
- the shielded policy `π⁺(a|s)`

This module should be the main public API for the rest of the training code.

### 2. New PLPG-Aware PPO Algorithm

Do not patch SB3 PPO in place if it can be avoided.

Add a new algorithm class, for example:

- `logicity/rl_agent/alg/plpg_ppo.py`

and expose it in:

- [logicity/rl_agent/alg/__init__.py](D:/Dev/LogicityFresh/LogiCity/logicity/rl_agent/alg/__init__.py)

Recommended approach:

- subclass SB3 PPO
- override the rollout action-selection path
- later override the loss path to use shielded log-probabilities and safety regularization

Why a separate class:

- keeps baseline PPO untouched
- makes ablations easier
- makes YAML config switching simple

### 3. Config-Level Activation

Add a new config family, for example:

- `config/tasks/Nav/easy/algo/plpg_ppo_smoke.yaml`
- `config/tasks/Nav/easy/algo/plpg_ppo_eval.yaml`

These should look like the current `ppo_lite_*` configs, but switch:

- `algorithm: "PLPGPPO"`

and add a new block such as:

```yaml
plpg:
  enabled: true
  difficulty: "easy"
  alpha: 0.1
  use_binary_facts: true
  compile_once: true
```

## Data Flow

The intended runtime flow is:

1. `GymCityWrapper` produces observation `s`
2. PPO policy produces logits/probabilities over 4 actions
3. PLPG state adapter derives safety facts from the current scene
4. PLPG shield computes `P(safe | s, a)` for all 4 actions
5. PLPG shield computes `π⁺(a|s)`
6. action is sampled from `π⁺`
7. environment executes that action
8. PPO stores rollout data consistent with `π⁺`

The important property is:

- action selection and learning must be based on the same shielded distribution

That is the central PLPG requirement.

## How to Get the Safety Facts

There are two possible sources.

### Option A: Recompute from the current scene

Use the current environment state and derive the needed binary facts directly.

Pros:

- clean separation from the current planner implementation
- easier to control the exact PLPG safety abstraction

Cons:

- duplicates some logic already present elsewhere

### Option B: Reuse planner-derived predicate state

Tap into the current grounded symbolic information already produced by the local planner.

Pros:

- avoids duplicated predicate definitions
- closer to current task semantics

Cons:

- tighter coupling to existing planner internals

Recommended first step:

- use a small explicit state adapter for `easy-lite`

That is easier to debug than depending on a large planner-internal grounding structure.

## Easy-Lite ProbLog Template

For `easy-lite`, the task rule is essentially:

- if intersection occupancy or higher-priority conflict or collision-close is present,
- then `Stop` is the safe/required action

So the first PLPG safety template can be very small.

A conceptual shape:

```prolog
Pslow::act(slow);
Pnormal::act(normal);
Pfast::act(fast);
Pstop::act(stop).

is_at_inter.
is_in_inter.
higher_pri.
colliding_close.

unsafe :- is_at_inter, is_in_inter, act(fast).
unsafe :- is_at_inter, is_in_inter, act(normal).
unsafe :- is_at_inter, is_in_inter, act(slow).

unsafe :- is_at_inter, higher_pri, act(fast).
unsafe :- is_at_inter, higher_pri, act(normal).
unsafe :- is_at_inter, higher_pri, act(slow).

unsafe :- colliding_close, act(fast).
unsafe :- colliding_close, act(normal).
unsafe :- colliding_close, act(slow).

safe :- not(unsafe).
```

The exact final template should match the current task semantics more carefully, but this is the right first scale.

## Training Phases

### Phase 1: Shielded Acting Only

Goal:

- verify that the shield can compute `π⁺`
- verify that action sampling uses `π⁺`
- verify that evaluation metrics improve in safety-sensitive scenarios

At this phase:

- the environment action comes from `π⁺`
- PPO may still be learning approximately like PPO

This is not yet full PLPG, but it validates the shield.

### Phase 2: Learning Through `π⁺`

Goal:

- replace learning through `π` with learning through `π⁺`

This is the first real PLPG phase.

Needed changes:

- stored log-probabilities must correspond to `π⁺`
- entropy, ratio, and surrogate objective must be based on the shielded distribution

### Phase 3: Add Safety Regularization

Goal:

- add the PLPG safety term weighted by `α`

Needed quantity:

- `log Pπ⁺(safe | s)`

This becomes an extra loss term during update.

## Minimal File Plan

Recommended initial file additions:

- `logicity/plpg/__init__.py`
- `logicity/plpg/shield.py`
- `logicity/plpg/program_builder.py`
- `logicity/plpg/state_adapter.py`
- `logicity/rl_agent/alg/plpg_ppo.py`
- `config/tasks/Nav/easy/algo/plpg_ppo_smoke.yaml`
- `config/tasks/Nav/easy/algo/plpg_ppo_eval.yaml`

Possible later additions:

- `logicity/plpg/cache.py`
- `logicity/plpg/templates/easy.pl`
- `docs/plpg_ablation_plan.md`

## Recommended First Implementation Order

1. implement `state_adapter.py`
2. implement `shield.py` with a simple `compute_shielded_policy(...)` API
3. test `π -> π⁺` transformation in isolation
4. create `PLPGPPO` with shielded action selection
5. run smoke training on `easy-lite`
6. add shield-aware evaluation logging
7. implement the safety loss

## Evaluation Criteria

For the first PLPG prototype, compare against current PPO on:

- TSR
- mean reward
- failure rate
- timeout rate
- stop-action count
- `decision_succ_action_3`
- failure-rule counts

Additional PLPG-specific metrics to add later:

- mean `Pπ(safe | s)`
- mean `P(safe | s, Stop)`
- mean `P(safe | s, Fast)`
- average KL divergence between `π` and `π⁺`
- shield intervention rate

## Important Engineering Risks

### 1. Rollout/learning mismatch

If the rollout action is sampled from `π⁺` but PPO still computes log-probabilities under raw `π`, then the method is not PLPG.

This must be avoided.

### 2. Per-step inference overhead

ProbLog-based action-safety queries inside PPO rollouts may slow training.

Mitigation:

- start with `easy-lite`
- use a very small rule template
- cache or compile the fixed program structure if possible

### 3. Tight coupling to planner internals

If the shield depends too directly on current Z3-grounding internals, the prototype becomes fragile.

Mitigation:

- keep the PLPG state adapter narrow and explicit

### 4. Safety template mismatch

If the ProbLog safety rules do not match the current task semantics, evaluation becomes hard to interpret.

Mitigation:

- align `easy-lite` PLPG rules with the current `easy` task rule first

## Recommended First Research Prototype

The most practical first prototype is:

- `easy-lite` only
- binary facts
- 4-action PLPG shield
- shielded action sampling
- no full safety loss yet

That gives the cleanest path to answering:

- does a probabilistic logic shield improve Safe Path Following behavior in LogiCity?

Only after that should the full PLPG objective be implemented.

## Final Recommendation

Implement PLPG as a new algorithm path, not as a direct mutation of the baseline PPO path.

That gives:

- cleaner experiments
- easier debugging
- better ablations
- less risk of breaking the current PPO benchmark

