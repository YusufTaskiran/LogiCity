# Experiment Family 1: Single-Agent Main Benchmark

## Objective

Test whether `PLPG-PPO` improves safe task completion and training efficiency relative to `PPO` on single-agent SPF.

## Scope

- methods:
  - `PPO`
  - `PLPG-PPO`
- difficulties:
  - `easy`
  - `medium`
  - `hard`
- seeds:
  - `101`
  - `202`
  - `303`

## Step-by-step Plan

1. Verify that the real validation and test splits exist for all three difficulties.
2. Verify that the eval configs point to the correct real validation splits.
3. Run one pilot seed on `easy` for:
   - `PPO`
   - `PLPG-PPO`
4. Inspect the pilot CSVs for:
   - `TSR`
   - `failure_rate`
   - `timeout_rate`
   - `mean_reward`
   - wall-clock time
5. If the pilot behavior is sensible, run the remaining seeds on `easy`.
6. Run the same method pair on `medium` for all seeds.
7. Run the same method pair on `hard` for all seeds.
8. After each training run, select the checkpoint using the fixed validation criterion:
   - best validation `TSR`
9. Run held-out final test evaluation for each finished training run.
10. Aggregate validation and test CSVs across seeds.
11. Plot:
   - `TSR` vs timestep
   - `failure_rate` vs timestep
   - `timeout_rate` vs timestep
   - `mean_reward` vs timestep
   - the same core plots vs wall-clock time
12. Build the final summary table across:
   - `easy`
   - `medium`
   - `hard`

## Required Inputs

- fixed validation split
- fixed test split
- stable PPO config
- stable PLPG config

## Required Outputs

- training eval CSVs
- final test CSVs
- per-difficulty plots
- final summary table

## Success Criteria

- PLPG is compared fairly against PPO under matched budgets
- results are reported across seeds
- conclusions are based primarily on:
  - `TSR`
  - `failure_rate`
  - `timeout_rate`
  - training efficiency

## Notes

- This is the main thesis benchmark.
- Do not change shield semantics or training budgets mid-family.
- Freeze the preferred single-agent PLPG design before launching the full benchmark.
