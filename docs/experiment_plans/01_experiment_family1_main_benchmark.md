# Experiment Family 1: Controlled \(K=1\) Benchmark

## Objective

Test whether `PLPG-PPO` improves safe task completion and training efficiency relative to `PPO` in the **core thesis setting**:

- one RL-controlled `normal car`,
- background `normal cars` and `pedestrians`,
- shared-policy SPF instantiated with \(K=1\).

## Scope

- methods:
  - `PPO`
  - `PLPG-PPO`
- setting:
  - `core_easy`
- seeds:
  - `101`
  - `202`
  - `303`

This family is the controlled benchmark for the main thesis comparison. It is not a separate single-agent thesis setting. It is the simplest instantiation of the same shared-policy multi-agent formulation used throughout the thesis.

## Configs Prepared for This Family

### Agent rosters

- train: `config/agents/core_easy/train.yaml`
- val: `config/agents/core_easy/val.yaml`
- test: `config/agents/core_easy/test.yaml`

These rosters keep the RL-controlled agent homogeneous with the later multi-agent experiments:

- `Car_1`: RL-controlled normal car
- `Pedestrian_1`: expert-controlled pedestrian
- `Car_2`: expert-controlled normal car
- `Car_3`: expert-controlled normal car
- `Car_4`: expert-controlled normal car
- `Car_5`: expert-controlled normal car

### Episode-cache generation configs

- val: `config/tasks/Nav/easy/experts/core_episode_val.yaml`
- test: `config/tasks/Nav/easy/experts/core_episode_test.yaml`

The fixed episode caches use an `intersection_biased_success_only` filter with:

- ego route crossing the center intersection,
- all agents in the world also crossing that center intersection,
- at least `3` other agents crossing it,
- successful expert completion.

### Training configs

- PPO: `config/tasks/Nav/easy/algo/core_ppo_eval.yaml`
- PLPG-PPO: `config/tasks/Nav/easy/algo/core_plpg_ppo_eval.yaml`

### Held-out test configs

- PPO: `config/tasks/Nav/easy/algo/core_ppo_test.yaml`
- PLPG-PPO: `config/tasks/Nav/easy/algo/core_plpg_ppo_test.yaml`

These configs are matched on:

- map
- agent roster
- ontology
- rule file
- reward scheme
- `num_envs`
- total timestep budget
- evaluation cadence

For `core_easy`, the observation ontology is intentionally reduced to:

- unary predicates:
  - `IsAtInter`
  - `IsInInter`
- binary predicates:
  - `HigherPri`
  - `CollidingClose`

With `fov_entities.Entity = 5`, this yields an observation dimension of:

\[
2 \cdot 5 + 2 \cdot (5 - 1) = 10 + 8 = 18.
\]

The reason is that `HigherPri` and `CollidingClose` are grounded **ego-centrically** in this family:

- `HigherPri(other_i, ego)`
- `CollidingClose(ego, other_i)`

rather than over the full ordered pair matrix of all visible entities.

## Step-by-step Plan

1. Generate fixed validation episodes:

```powershell
python tools/create_episode.py --config config/tasks/Nav/easy/experts/core_episode_val.yaml --exp core_easy_val --max_episodes 40
```

2. Generate fixed test episodes:

```powershell
python tools/create_episode.py --config config/tasks/Nav/easy/experts/core_episode_test.yaml --exp core_easy_test --max_episodes 100
```

3. Verify that the episode caches exist:

- `log_rl/core_easy_val_episodes.pkl`
- `log_rl/core_easy_test_episodes.pkl`

4. Run one pilot seed for PPO:

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/core_ppo_eval.yaml --exp core_easy_ppo_seed101
```

5. Run one pilot seed for PLPG-PPO:

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/core_plpg_ppo_eval.yaml --exp core_easy_plpg_seed101
```

6. Inspect the pilot eval CSVs for:

- `TSR`
- `failure_rate`
- `timeout_rate`
- `mean_reward`
- wall-clock time

7. If the pilot behavior is sensible, run the remaining seeds:

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/core_ppo_eval.yaml --exp core_easy_ppo_seed202
python main.py --use_gym --config config/tasks/Nav/easy/algo/core_ppo_eval.yaml --exp core_easy_ppo_seed303
python main.py --use_gym --config config/tasks/Nav/easy/algo/core_plpg_ppo_eval.yaml --exp core_easy_plpg_seed202
python main.py --use_gym --config config/tasks/Nav/easy/algo/core_plpg_ppo_eval.yaml --exp core_easy_plpg_seed303
```

8. Select checkpoints using the fixed validation criterion:

- best validation `TSR`

9. Run held-out final test evaluation for each finished training run using the fixed `core_easy_test` cache, for example:

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/core_ppo_test.yaml --exp core_easy_ppo_seed101_test --checkpoint_path checkpoints/core_easy_ppo_seed101/<best-checkpoint>.zip
python main.py --use_gym --config config/tasks/Nav/easy/algo/core_plpg_ppo_test.yaml --exp core_easy_plpg_seed101_test --checkpoint_path checkpoints/core_easy_plpg_seed101/<best-checkpoint>.zip
```

10. Aggregate validation and test CSVs across seeds.

11. Plot:

- `TSR` vs timestep
- `failure_rate` vs timestep
- `timeout_rate` vs timestep
- `mean_reward` vs timestep
- the same core plots vs wall-clock time

12. Build the final summary table across:

- PPO
- PLPG-PPO
- seeds
- validation versus test

## Required Outputs

- `log_rl/core_easy_val_episodes.pkl`
- `log_rl/core_easy_test_episodes.pkl`
- training eval CSVs
- final test CSVs
- seed-aggregated plots
- final summary table

## Success Criteria

- `PLPG-PPO` is compared fairly against `PPO` under matched budgets
- conclusions are based primarily on:
  - `TSR`
  - `failure_rate`
  - `timeout_rate`
  - training efficiency
- this family provides a clean controlled reference point before moving to \(K=2\) and \(K=3\)

## Notes

- Do not change shield semantics or training budgets mid-family.
- The preferred PLPG design is already frozen here:
  - `alpha = 0.1`
  - perfect symbolic inputs
  - coarse shield semantics currently implemented in code
- Richer traffic roles such as `ambulance`, `bus`, and `police` are not part of this family. They belong to the optional extension setting.
