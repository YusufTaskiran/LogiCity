# Experiment Family 4: Shared-Policy Multi-Agent RL

## Objective

Test whether PLPG remains useful when multiple RL-controlled agents interact in the same LogiCity environment.

## Scope

- shared-policy decentralized MARL
- first task:
  - `easy`
- first seeds:
  - `101`

## Step-by-step Plan

1. Define the first multi-agent task:
   - `2 RL cars`
   - compact map
   - shared policy
   - local observations
2. Verify that PPO training works in the multi-agent wrapper.
3. Run the shared-policy PPO baseline.
4. Implement or verify shared-policy PLPG training on the same task.
5. Run shared-policy PLPG on the same task.
6. Compare:
   - per-agent `TSR`
   - joint success rate
   - failure rate
   - timeout rate
   - training efficiency
7. If the `2 RL cars` setup works, scale to:
   - `3 RL cars`
8. If the method remains stable, add one harder task or denser background-agent condition.
9. Build the final MARL comparison table.

## Required Inputs

- working shared-policy multi-agent environment wrapper
- fixed multi-agent validation and test protocol
- PPO baseline
- PLPG multi-agent implementation

## Required Outputs

- PPO multi-agent CSVs
- PLPG multi-agent CSVs
- joint and per-agent evaluation tables

## Success Criteria

- demonstrate whether PLPG helps in interacting multi-agent learning
- show whether the method reduces unsafe joint behavior without destroying `TSR`

## Notes

- This family is important for the thesis assignment and should not be left until the very end.
- Start small and stable before scaling agent count.
