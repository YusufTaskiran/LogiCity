# Experiment Family 5: Shield Design Ablations

## Objective

Study which aspects of the shield are actually responsible for PLPG behavior.

## Scope

- start on:
  - `easy`
- fixed seed:
  - `101`

## Step-by-step Plan

1. Freeze one stable baseline PLPG setup.
2. Run the shield-on vs shield-off comparison:
   - `PPO`
   - `PLPG shield-only`
   - `PLPG full`
3. Run the shield action granularity comparison:
   - coarse `Stop` vs `Move`
   - refined `Stop` / `Slow` / `Normal` / `Fast`
4. Run one fact-set ablation at a time:
   - remove `HigherPri`
   - remove `CollidingClose`
   - remove `IsInInter`
5. Run one rule-strength ablation at a time:
   - strict shield
   - reduced shield
   - partial shield
6. Run the safety-loss-source comparison:
   - base-policy safety
   - shielded-policy safety
7. For each ablation, compare:
   - `TSR`
   - `failure_rate`
   - `timeout_rate`
   - `mean_reward`
   - `shield_intervention_rate`
   - `mean_safety_gain`
8. Identify which design choices:
   - improve safe progress
   - create conservatism
   - matter most under uncertainty
9. Promote only the most useful shield findings into later benchmark conclusions.

## Required Inputs

- one fixed baseline PLPG setup
- one fixed validation split
- stable logging schema

## Required Outputs

- ablation CSVs
- ablation plots
- one shield-design summary table

## Success Criteria

- explain why PLPG works or fails in LogiCity
- distinguish useful shield structure from incidental implementation choices

## Notes

- This family is where the thesis can make a real methodological contribution beyond pure benchmarking.
- The shield action granularity study is one of the most important parts.
