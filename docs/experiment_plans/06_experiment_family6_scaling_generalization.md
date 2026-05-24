# Experiment Family 6: Larger-Map And Higher-Agent-Count Generalization

## Objective

Test whether PLPG remains useful as the environment becomes larger and more crowded.

## Scope

Two branches:

- single RL agent + more background agents
- multiple RL agents + more total agents

## Step-by-step Plan

1. Choose one already working compact baseline policy family.
2. Define the first scaling settings:
   - larger map such as `5x5`
   - more background agents
3. Run single-agent PPO on the larger-map setting.
4. Run single-agent PLPG on the same setting.
5. Compare:
   - `TSR`
   - `failure_rate`
   - `timeout_rate`
   - wall-clock cost
6. If single-agent scaling is stable, repeat with:
   - more background-agent density
7. Move to multi-agent scaling:
   - `2 RL cars` on larger map
   - then `3 RL cars` on larger map
8. Compare PPO vs PLPG under increasing:
   - map size
   - total agent count
   - RL-agent count
9. Record systems cost:
   - shield inference time
   - slowdown as agent count increases
10. Build one final scaling/generalization table.

## Required Inputs

- working compact baseline policies
- larger-map configs
- denser-agent configs
- stable evaluation protocol

## Required Outputs

- scaling CSVs
- map-size and agent-density plots
- one generalization summary table

## Success Criteria

- show whether the thesis contribution extends beyond compact interaction cases
- identify whether PLPG degrades because of:
  - interaction complexity
  - over-conservative stopping
  - compute overhead

## Notes

- This family strengthens the external validity of the thesis.
- Run it after the compact single-agent and initial multi-agent designs are already stable.
