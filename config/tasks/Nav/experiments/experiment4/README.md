Experiment 4 configs.

Purpose:
- multi-agent homogeneous versus heterogeneous shields
- easy difficulty, `K = 3`
- shared-policy multi-agent PLS
- uncertain symbolic observations

Contents:
- `easy_plpg_homogeneous_conservative_k3.yaml`
- `easy_plpg_homogeneous_aggressive_k3.yaml`
- `easy_plpg_heterogeneous_mix_1c2a_k3.yaml`
- `easy_plpg_heterogeneous_mix_2c1a_k3.yaml`

Notes:
- all runs use shared-policy training
- agent identity is appended to the observation only for this experiment so the shield can apply agent-specific profiles
- conservative and aggressive shield styles are implemented through different `stop_benign_safety` values
- heterogeneous runs use explicit per-agent profiles:
  - `1c2a`: one conservative agent and two aggressive agents
  - `2c1a`: two conservative agents and one aggressive agent
- the older `easy_plpg_heterogeneous_mix_k3.yaml` is kept for compatibility but superseded by the explicit `1c2a` and `2c1a` configs
