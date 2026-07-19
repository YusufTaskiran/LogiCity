Experiment 1 configs.

Purpose:
- main multi-agent PPO vs PLS benchmark
- easy difficulty
- uncertain symbolic observations
- shared-policy multi-agent setup aligned with experiments 2 and 3

Contents:
- `easy_ppo_k1.yaml`
- `easy_ppo_k2.yaml`
- `easy_ppo_k3.yaml`
- `easy_plpg_k1.yaml`
- `easy_plpg_k2.yaml`
- `easy_plpg_k3.yaml`

Notes:
- all configs use the same entity-correlated asymmetric observation-noise regime as experiments 2 and 3
- all configs use `multi_agent_training_scheme: "shared_policy"`
- use `--checkpoint_root checkpoints/experiment_1` to keep outputs isolated
