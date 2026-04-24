# Thesis Experiment Matrix

## Current Scope

The thesis setup now uses a single sensing regime: perfect logical predicates from the simulator. The uncertainty and noise variants were removed from the runtime code and thesis configs, so all final experiments should be described against this one clean benchmark setting.

The current experiment plan has four tracks:

- `PPO`
- `PPO + DLS`
- `PPO + PLS` with one shield per RL-controlled agent
- `PPO + PLS` with a joint shield concept for the two RL-controlled cars

Only the first three are part of the active executable benchmark. The joint-shield PLS track is currently a planned extension and should be presented as a design/next-step item unless and until it is implemented.

The corresponding implementations are:

- `logicity/rl_agent/alg/dls_ppo.py`
- `logicity/rl_agent/alg/pls_ppo.py`
- `logicity/shields/dls.py`
- `logicity/shields/pls.py`

## Environment

All thesis experiments use:

- map: `config/maps/square_2x2.yaml`
- agents: `config/agents/thesis/main_3cars_1ped.yaml`
- ontology: `config/rules/ontology_simple.yaml`
- expert rules: `config/rules/Nav/easy/expert_simple.yaml`

The benchmark contains four agents:

- `Car_1` is the ego vehicle for single-agent training and testing
- `Car_2`, `Car_3`, and `Pedestrian_1` remain in the scene

Shared-policy evaluation uses the same learned checkpoint for:

- `Car_1`
- `Car_2`

The observation setting remains fixed with `fov_entities.Entity = 4`.

## Action Space

The thesis uses a discrete 4-action control space:

- `0 = slow`
- `1 = normal`
- `2 = fast`
- `3 = stop`

The current action costs are:

- `slow = -3`
- `normal = 0`
- `fast = -4`
- `stop = -5`

## Shield Semantics

Both shielded methods act at the policy level using:

`pi_plus(a|s) propto P(safe|s,a) * pi(a|s)`

### DLS

`logicity/shields/dls.py` computes a hard safety mask from perfect predicates and renormalizes over safe actions only.

- `P(safe|s,a) in {0, 1}`
- unsafe actions are masked out
- `stop` remains the deterministic fallback action

### PLS

`logicity/shields/pls.py` computes soft safety values from perfect predicates and renormalizes the base policy with those values.

- `P(safe|s,a) in [0, 1]`
- actions are downweighted or upweighted instead of being only masked
- the thesis configs use the ProbLog backend by default

`logicity/rl_agent/alg/pls_ppo.py` also keeps the PLPG-style safety loss term weighted by `alpha` / `safety_coefficient`.

### Joint PLS Concept

The planned joint-shield extension is not the same as running one independent shield per agent. The intended idea is to score safety for a joint action tuple, for example:

`P(safe | s, a_1, a_2)`

for the two RL-controlled cars in the shared-policy setting.

The cleanest initial design is:

- keep the learned base policy per agent
- form the joint action distribution over the 16 two-car action pairs
- evaluate a joint safety score for each pair
- renormalize over those joint pairs
- execute the selected pair jointly

This joint-shield concept should currently be treated as a planned extension rather than a completed experiment.

## Frozen Benchmark Matrix

The current frozen benchmark for thesis reporting is:

- `PPO`
- `PPO + DLS`
- `PPO + PLS` using the current ProbLog-based implementation

The joint PLS concept is outside the frozen benchmark until implemented and rerun properly.

## PLS Rerun Requirement

The PLS branch must be rerun for final reporting because the method changed materially when it was aligned more closely with the PLS paper through the ProbLog-based safety model.

That means earlier PLS results should be treated only as exploratory and not as final evidence. Final PLS reporting must come from fresh runs of:

- training
- single-agent testing
- shared-policy evaluation

## Training And Evaluation Protocol

The active thesis configs for the frozen benchmark are:

- training:
  - `config/tasks/Nav/thesis/algo/ppo_single_train.yaml`
  - `config/tasks/Nav/thesis/algo/ppo_dls_single_train.yaml`
  - `config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml`
- single-agent test:
  - `config/tasks/Nav/thesis/algo/ppo_single_test.yaml`
  - `config/tasks/Nav/thesis/algo/ppo_dls_single_test.yaml`
  - `config/tasks/Nav/thesis/algo/ppo_pls_single_test.yaml`
- shared-policy evaluation:
  - `config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml`
  - `config/tasks/Nav/thesis/algo/ppo_dls_shared_eval.yaml`
  - `config/tasks/Nav/thesis/algo/ppo_pls_shared_eval.yaml`

The common training setup is:

- `40000` total timesteps
- evaluation every `1000` timesteps
- checkpoint saving every `1000` timesteps
- `max_horizon = 200`

Final reported results should be based on `3` seeds per method.

For the frozen benchmark:

- existing `PPO` runs can be reused if their configs already match the frozen perfect-sensor setup
- existing `PPO + DLS` runs can be reused if their configs already match the frozen perfect-sensor setup
- `PPO + PLS` must be rerun because the ProbLog-backed implementation changed the evaluated method

## Dataset Splits

The thesis dataset lives under `dataset/thesis/main_3cars_1ped/`:

- training split: `train_100_episodes.pkl`
- validation split: `val_20_episodes.pkl`
- test split: `test_50_episodes.pkl`

For shared-policy evaluation, use the shared-oracle annotated variants produced by `tools/annotate_shared_oracle_steps.py`.

## Metrics

Primary thesis metrics:

- `TSR`
- steps to reach target `TSR` thresholds during training
- `train_fail_since_last_eval`
- cumulative `train_fail`
- final-test `fail`
- final-test `timeout`

Secondary supporting metrics:

- `DSR`
- `mean_reward`
- shared-policy `car_1_tsr`
- shared-policy `car_2_tsr`
- shield diagnostics such as `shield_intervention_rate`
- runtime metrics such as `seconds_per_1k_timesteps`

## Result Presentation

The results section should now be organized by evaluation setting rather than sensor scenario:

1. training behaviour
2. final single-agent test results
3. final shared-policy multi-agent results
4. optional future-work note on the joint PLS concept

Recommended figures:

- TSR learning curve
- training failure curve using `train_fail_since_last_eval`
- cumulative training failure curve using `train_fail`
- mean reward learning curve
- final single-agent summary chart
- final shared-policy summary chart including per-agent TSR
- TSR convergence summary based on steps to reach target thresholds
- optional runtime overhead chart

Recommended summary tables:

- final single-agent results
- final shared-policy results with `joint_tsr`, `car_1_tsr`, and `car_2_tsr`
- training safety and convergence summary

If the joint PLS concept is not implemented in time, it should appear only in methodology/future-work discussion, not in the final comparison tables.
