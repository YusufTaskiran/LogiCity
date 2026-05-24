# Experiment Family 3: Sensor Noise Robustness

## Objective

Test whether PLPG remains useful when the shield receives imperfect symbolic safety facts.

## Scope

- fixed task:
  - `easy`
- fixed seed:
  - `101`
- fixed `alpha`
- fixed shield design per sub-study

## Step-by-step Plan

1. Freeze the PLPG shield design to test.
2. Decide the noise mode:
   - first: soft symmetric probabilities
   - later: bit-flip noise
3. Choose one fixed training budget and eval schedule.
4. Run PLPG at noise levels:
   - `0%`
   - `5%`
   - `10%`
   - `20%`
   - `30%`
5. After each run, inspect:
   - `TSR`
   - `failure_rate`
   - `timeout_rate`
   - `mean_reward`
   - `mean_base_policy_safe_prob`
   - `mean_shielded_policy_safe_prob`
   - `shield_intervention_rate`
   - rollout failures and timeouts
6. Identify the robustness boundary:
   - low-noise success region
   - moderate-noise region
   - collapse region
7. If one noise level is borderline, extend only that run with a longer training budget.
8. Plot the full robustness curves.
9. Summarize:
   - where PLPG remains effective
   - where timeout-dominant conservatism appears
10. Optionally repeat one or two selected noise settings on `medium`.

## Required Inputs

- fixed `easy` validation split
- stable PLPG config
- chosen noise injection mechanism

## Required Outputs

- one eval CSV per noise setting
- robustness plots
- one narrative conclusion about the uncertainty limit of the current shield

## Success Criteria

- show whether PLPG under uncertainty still improves task completion
- distinguish:
  - real unsafe failures
  - over-conservative timeout collapse

## Notes

- This family is central for the probabilistic-shield story.
- Start with one fixed config, not all difficulties.
