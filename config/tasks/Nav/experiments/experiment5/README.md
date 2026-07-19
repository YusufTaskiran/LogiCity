Experiment 5 configs.

Purpose:
- larger and denser multi-agent deployment evaluation
- easy-rule shared-policy multi-agent PLS
- `5x5` map with `K = 5, 10, 15, 20`
- uncertain symbolic observations

Contents:
- `easy_5x5_plpg_k5.yaml`
- `easy_5x5_plpg_k10.yaml`
- `easy_5x5_plpg_k15.yaml`
- `easy_5x5_plpg_k20.yaml`

Notes:
- these are rollout-evaluation configs, not fresh training configs
- use them with `--checkpoint_path` pointing to a trained shared-policy PLS checkpoint
- they use quota-based continuous evaluation so each RL agent is measured over the same number of resolved goal attempts
- the same noisy symbolic observation regime is active here as in experiments 2 to 4
