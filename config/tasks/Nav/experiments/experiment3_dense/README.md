Experiment 3 dense redesign: shared-policy vs separate-policy multi-agent PLS.

Current scope:
- dense easy compact benchmark
- K = 2 and K = 3
- goal-only PPO reward component
- uncertain symbolic observations enabled
- expert background traffic uses `expert_multispeed.yaml`

Notes:
- these configs are separate from the older `experiment3/` files
- fixed-test evaluation is prepared for both shared and separate policy
- continuous rollout evaluation is already available for shared-policy configs
- separate-policy rollout evaluation is not yet wired as a dedicated config path
