# Lite Safe Path Following Experimental Setup

## Overview

This repo now contains a reduced-cost Safe Path Following curriculum for PPO:

- `easy-lite`
- `medium-lite`
- `hard-lite`

The RL-controlled agent is always:

- `Car_1`
- a plain `normal` car

Background agents vary by difficulty:

- `easy-lite`: pedestrian, ambulance car, normal car
- `medium-lite`: pedestrian, ambulance car, bus car, normal car
- `hard-lite`: pedestrian, ambulance car, police car, normal car

The lite curriculum keeps the original logic families but reduces training cost with:

- smaller horizons
- smaller or unchanged maps depending on level
- smaller FOV
- fewer agents
- Safe Path Following reward shaping

## Lite task configs

Training configs:

- `config/tasks/Nav/easy/algo/ppo_lite.yaml`
- `config/tasks/Nav/medium/algo/ppo_lite.yaml`
- `config/tasks/Nav/hard/algo/ppo_lite.yaml`

Agent rosters:

- `config/agents/easy_lite/{train,val,test}.yaml`
- `config/agents/medium_lite/{train,val,test}.yaml`
- `config/agents/hard_lite/{train,val,test}.yaml`

Episode-generation configs:

- `config/tasks/Nav/easy/experts/expert_collect_train_lite.yaml`
- `config/tasks/Nav/easy/experts/expert_episode_val_lite.yaml`
- `config/tasks/Nav/easy/experts/expert_episode_test_lite.yaml`
- `config/tasks/Nav/medium/experts/expert_collect_train_lite.yaml`
- `config/tasks/Nav/medium/experts/expert_episode_val_lite.yaml`
- `config/tasks/Nav/medium/experts/expert_episode_test_lite.yaml`
- `config/tasks/Nav/hard/experts/expert_collect_train_lite.yaml`
- `config/tasks/Nav/hard/experts/expert_episode_val_lite.yaml`
- `config/tasks/Nav/hard/experts/expert_episode_test_lite.yaml`

## Reward setup

The lite curriculum uses `reward_scheme: safe_path_following`.

Reward behavior:

- rule failure: keep the existing logical penalty, terminate
- success: add `goal_reward`
- safe movement: add progress reward from reduction in remaining distance along `global_traj`
- every non-terminal safe step: add a small `time_penalty`

This avoids the degenerate "always stop" policy now that action costs are zero.

## Split strategy

### Training split

For PPO training, you do **not** need a cached episode file.

Training samples are generated online by the environment during PPO rollouts.

If you want an expert trajectory dataset for analysis, imitation, or debugging, generate it with:

- `main.py --collect_only`

This produces expert demonstrations, not cached evaluation episodes.

### Validation and test splits

Validation and test use cached episode `.pkl` files.

These are generated with:

- `tools/create_episode.py`

Each cached episode stores:

- city grid
- agent start/goal/position/concepts
- `label_info`
- expert `oracle_step`

The lite configs use `episode_generation.mode: success_only`, so the generator simply keeps successful expert episodes until the requested split size is reached.

## How to create the splits

### 1. Optional: collect expert training trajectories

Easy:

```powershell
python main.py --collect_only --config config/tasks/Nav/easy/experts/expert_collect_train_lite.yaml --exp easy_lite_train
```

Medium:

```powershell
python main.py --collect_only --config config/tasks/Nav/medium/experts/expert_collect_train_lite.yaml --exp medium_lite_train
```

Hard:

```powershell
python main.py --collect_only --config config/tasks/Nav/hard/experts/expert_collect_train_lite.yaml --exp hard_lite_train
```

Output:

- `log_rl/<exp>_expert_demonstrations.pkl`

These are trajectory datasets, not `episode_data` caches.

### 2. Create validation episode caches

Easy:

```powershell
python tools/create_episode.py --config config/tasks/Nav/easy/experts/expert_episode_val_lite.yaml --exp easy_lite_val --max_episodes 40
```

Medium:

```powershell
python tools/create_episode.py --config config/tasks/Nav/medium/experts/expert_episode_val_lite.yaml --exp medium_lite_val --max_episodes 40
```

Hard:

```powershell
python tools/create_episode.py --config config/tasks/Nav/hard/experts/expert_episode_val_lite.yaml --exp hard_lite_val --max_episodes 40
```

Output:

- `log_rl/<exp>_episodes.pkl`

Recommended destination:

- `dataset/easy_lite/val_40_episodes.pkl`
- `dataset/medium_lite/val_40_episodes.pkl`
- `dataset/hard_lite/val_40_episodes.pkl`

### 3. Create test episode caches

Easy:

```powershell
python tools/create_episode.py --config config/tasks/Nav/easy/experts/expert_episode_test_lite.yaml --exp easy_lite_test --max_episodes 100
```

Medium:

```powershell
python tools/create_episode.py --config config/tasks/Nav/medium/experts/expert_episode_test_lite.yaml --exp medium_lite_test --max_episodes 100
```

Hard:

```powershell
python tools/create_episode.py --config config/tasks/Nav/hard/experts/expert_episode_test_lite.yaml --exp hard_lite_test --max_episodes 100
```

Output:

- `log_rl/<exp>_episodes.pkl`

Recommended destination:

- `dataset/easy_lite/test_100_episodes.pkl`
- `dataset/medium_lite/test_100_episodes.pkl`
- `dataset/hard_lite/test_100_episodes.pkl`

## How to use the splits

### PPO training

Train directly from the lite PPO configs:

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/ppo_lite.yaml --exp easy_lite_spf
python main.py --use_gym --config config/tasks/Nav/medium/algo/ppo_lite.yaml --exp medium_lite_spf
python main.py --use_gym --config config/tasks/Nav/hard/algo/ppo_lite.yaml --exp hard_lite_spf
```

No cached training episode file is required for PPO.

### Fixed evaluation

If you later want fixed cached validation/test evaluation in task configs, point `episode_data` to the generated files, for example:

- `dataset/easy_lite/val_40_episodes.pkl`
- `dataset/easy_lite/test_100_episodes.pkl`

and mirror that pattern for `medium_lite` and `hard_lite`.

## Notes

- `easy-lite` uses `config/maps/square_2x2.yaml`
- `medium-lite` and `hard-lite` use `config/maps/square_5x5.yaml`
- the RL car is always `normal`
- the current lite episode generation path is intended for successful SPF episodes, not the old stop-balanced concept/action benchmarking mode
