# Experiment Family 2: Alpha Sweep

## Objective

Measure how the safety weight `alpha` affects the tradeoff between safety and conservatism.

## Scope

- fixed task:
  - `easy`
- fixed seed:
  - `101`
- fixed shield design:
  - current chosen PLPG shield

## Step-by-step Plan

1. Freeze one PLPG shield design before starting the sweep.
2. Choose one fixed training budget and eval schedule for all alpha runs.
3. Create one config per `alpha` value.
4. Run PLPG for:
   - `alpha = 0.0`
   - `0.01`
   - `0.05`
   - `0.1`
   - `0.2`
   - `0.5`
5. Inspect each eval CSV for:
   - `TSR`
   - `failure_rate`
   - `timeout_rate`
   - `mean_reward`
   - `shield_intervention_rate`
   - `mean_safety_gain`
6. Compare whether larger `alpha`:
   - lowers failures
   - increases timeout collapse
   - changes learning speed
7. Identify one preferred `alpha` for later benchmark use.
8. Optionally run one confirmatory check on `medium` with the preferred `alpha`.

## Required Inputs

- fixed `easy` validation split
- fixed PLPG shield design
- fixed training budget

## Required Outputs

- one eval CSV per alpha value
- alpha tradeoff plots
- one final chosen `alpha`

## Success Criteria

- isolate how much of PLPG behavior depends on the explicit safety regularizer
- distinguish:
  - shield-only effect
  - added safety-loss effect

## Notes

- This family is method analysis, not a full benchmark.
- Do not run this sweep across all difficulties unless extra time remains.
