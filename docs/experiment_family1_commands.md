# Experiment Family 1 Commands

This file lists the commands for **Experiment Family 1**:

- single-agent benchmark
- compact `2x2` in-distribution setup
- methods:
  - `PPO`
  - `PLPG-PPO`
- difficulties:
  - `easy`
  - `medium`
  - `hard`
- seeds:
  - `101`
  - `202`
  - `303`

## 1. Split Generation

If the real cached splits are already present, these do not need to be rerun.

Current convention:

- validation split: `20` episodes
- test split: former `40`-episode validation split
- old `100`-episode test splits are kept only as archived files and are not used in Experiment Family 1

### Easy

Validation:

```powershell
python tools/create_episode.py --config config/tasks/Nav/easy/experts/expert_episode_val_lite.yaml --exp easy_lite_val --max_episodes 20 --num_workers 4 --vis_count 2 --progress_every 5 --seed 101
```

Test:

The test split is the existing `40`-episode cached split now stored under `easy_lite_test_*`.

### Medium

Validation:

```powershell
python tools/create_episode.py --config config/tasks/Nav/medium/experts/expert_episode_val_lite.yaml --exp medium_lite_val --max_episodes 20 --num_workers 4 --vis_count 2 --progress_every 5 --seed 101
```

Test:

The test split is the existing `40`-episode cached split now stored under `medium_lite_test_*`.

### Hard

Validation:

```powershell
python tools/create_episode.py --config config/tasks/Nav/hard/experts/expert_episode_val_lite.yaml --exp hard_lite_val --max_episodes 20 --num_workers 4 --vis_count 2 --progress_every 5 --seed 101
```

Test:

The test split is the existing `40`-episode cached split now stored under `hard_lite_test_*`.

## 2. Visualization Commands

Use `--ego_id 3` for the RL car.

### Medium

Validation:

```powershell
python tools/pkl2gif.py --pkl log_rl/medium_lite_val_0.pkl --ego_id 3 --output_folder vis/medium_lite_val_0 --output_gif vis/medium_lite_val_0.gif --scale_factor 2
python tools/pkl2gif.py --pkl log_rl/medium_lite_val_1.pkl --ego_id 3 --output_folder vis/medium_lite_val_1 --output_gif vis/medium_lite_val_1.gif --scale_factor 2
```

Test:

```powershell
python tools/pkl2gif.py --pkl log_rl/medium_lite_test_0.pkl --ego_id 3 --output_folder vis/medium_lite_test_0 --output_gif vis/medium_lite_test_0.gif --scale_factor 2
python tools/pkl2gif.py --pkl log_rl/medium_lite_test_1.pkl --ego_id 3 --output_folder vis/medium_lite_test_1 --output_gif vis/medium_lite_test_1.gif --scale_factor 2
```

### Hard

Validation:

```powershell
python tools/pkl2gif.py --pkl log_rl/hard_lite_val_0.pkl --ego_id 3 --output_folder vis/hard_lite_val_0 --output_gif vis/hard_lite_val_0.gif --scale_factor 2
python tools/pkl2gif.py --pkl log_rl/hard_lite_val_1.pkl --ego_id 3 --output_folder vis/hard_lite_val_1 --output_gif vis/hard_lite_val_1.gif --scale_factor 2
```

Test:

```powershell
python tools/pkl2gif.py --pkl log_rl/hard_lite_test_0.pkl --ego_id 3 --output_folder vis/hard_lite_test_0 --output_gif vis/hard_lite_test_0.gif --scale_factor 2
python tools/pkl2gif.py --pkl log_rl/hard_lite_test_1.pkl --ego_id 3 --output_folder vis/hard_lite_test_1 --output_gif vis/hard_lite_test_1.gif --scale_factor 2
```

## 3. Training Runs

Total:

- `3` difficulties
- `2` methods
- `3` seeds
- `18` training runs

### Easy PPO

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/ppo_lite_eval.yaml --exp easy_lite_ppo_seed101 --seed 101
python main.py --use_gym --config config/tasks/Nav/easy/algo/ppo_lite_eval.yaml --exp easy_lite_ppo_seed202 --seed 202
python main.py --use_gym --config config/tasks/Nav/easy/algo/ppo_lite_eval.yaml --exp easy_lite_ppo_seed303 --seed 303
```

### Easy PLPG

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/plpg_ppo_eval.yaml --exp easy_lite_plpg_seed101 --seed 101
python main.py --use_gym --config config/tasks/Nav/easy/algo/plpg_ppo_eval.yaml --exp easy_lite_plpg_seed202 --seed 202
python main.py --use_gym --config config/tasks/Nav/easy/algo/plpg_ppo_eval.yaml --exp easy_lite_plpg_seed303 --seed 303
```

### Medium PPO

```powershell
python main.py --use_gym --config config/tasks/Nav/medium/algo/ppo_lite_eval.yaml --exp medium_lite_ppo_seed101 --seed 101
python main.py --use_gym --config config/tasks/Nav/medium/algo/ppo_lite_eval.yaml --exp medium_lite_ppo_seed202 --seed 202
python main.py --use_gym --config config/tasks/Nav/medium/algo/ppo_lite_eval.yaml --exp medium_lite_ppo_seed303 --seed 303
```

### Medium PLPG

```powershell
python main.py --use_gym --config config/tasks/Nav/medium/algo/plpg_ppo_eval.yaml --exp medium_lite_plpg_seed101 --seed 101
python main.py --use_gym --config config/tasks/Nav/medium/algo/plpg_ppo_eval.yaml --exp medium_lite_plpg_seed202 --seed 202
python main.py --use_gym --config config/tasks/Nav/medium/algo/plpg_ppo_eval.yaml --exp medium_lite_plpg_seed303 --seed 303
```

### Hard PPO

```powershell
python main.py --use_gym --config config/tasks/Nav/hard/algo/ppo_lite_eval.yaml --exp hard_lite_ppo_seed101 --seed 101
python main.py --use_gym --config config/tasks/Nav/hard/algo/ppo_lite_eval.yaml --exp hard_lite_ppo_seed202 --seed 202
python main.py --use_gym --config config/tasks/Nav/hard/algo/ppo_lite_eval.yaml --exp hard_lite_ppo_seed303 --seed 303
```

### Hard PLPG

```powershell
python main.py --use_gym --config config/tasks/Nav/hard/algo/plpg_ppo_eval.yaml --exp hard_lite_plpg_seed101 --seed 101
python main.py --use_gym --config config/tasks/Nav/hard/algo/plpg_ppo_eval.yaml --exp hard_lite_plpg_seed202 --seed 202
python main.py --use_gym --config config/tasks/Nav/hard/algo/plpg_ppo_eval.yaml --exp hard_lite_plpg_seed303 --seed 303
```

## 4. Final Test Runs

Total:

- `18` final test runs

### Easy PPO Test

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/ppo_lite_test.yaml --exp easy_lite_ppo_test_seed101 --seed 101 --checkpoint_path checkpoints/easy_lite_ppo_seed101/best_model.zip --log_dir checkpoints/easy_lite_ppo_seed101
python main.py --use_gym --config config/tasks/Nav/easy/algo/ppo_lite_test.yaml --exp easy_lite_ppo_test_seed202 --seed 202 --checkpoint_path checkpoints/easy_lite_ppo_seed202/best_model.zip --log_dir checkpoints/easy_lite_ppo_seed202
python main.py --use_gym --config config/tasks/Nav/easy/algo/ppo_lite_test.yaml --exp easy_lite_ppo_test_seed303 --seed 303 --checkpoint_path checkpoints/easy_lite_ppo_seed303/best_model.zip --log_dir checkpoints/easy_lite_ppo_seed303
```

### Easy PLPG Test

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/plpg_ppo_test.yaml --exp easy_lite_plpg_test_seed101 --seed 101 --checkpoint_path checkpoints/easy_lite_plpg_seed101/best_model.zip --log_dir checkpoints/easy_lite_plpg_seed101
python main.py --use_gym --config config/tasks/Nav/easy/algo/plpg_ppo_test.yaml --exp easy_lite_plpg_test_seed202 --seed 202 --checkpoint_path checkpoints/easy_lite_plpg_seed202/best_model.zip --log_dir checkpoints/easy_lite_plpg_seed202
python main.py --use_gym --config config/tasks/Nav/easy/algo/plpg_ppo_test.yaml --exp easy_lite_plpg_test_seed303 --seed 303 --checkpoint_path checkpoints/easy_lite_plpg_seed303/best_model.zip --log_dir checkpoints/easy_lite_plpg_seed303
```

### Medium PPO Test

```powershell
python main.py --use_gym --config config/tasks/Nav/medium/algo/ppo_lite_test.yaml --exp medium_lite_ppo_test_seed101 --seed 101 --checkpoint_path checkpoints/medium_lite_ppo_seed101/best_model.zip --log_dir checkpoints/medium_lite_ppo_seed101
python main.py --use_gym --config config/tasks/Nav/medium/algo/ppo_lite_test.yaml --exp medium_lite_ppo_test_seed202 --seed 202 --checkpoint_path checkpoints/medium_lite_ppo_seed202/best_model.zip --log_dir checkpoints/medium_lite_ppo_seed202
python main.py --use_gym --config config/tasks/Nav/medium/algo/ppo_lite_test.yaml --exp medium_lite_ppo_test_seed303 --seed 303 --checkpoint_path checkpoints/medium_lite_ppo_seed303/best_model.zip --log_dir checkpoints/medium_lite_ppo_seed303
```

### Medium PLPG Test

```powershell
python main.py --use_gym --config config/tasks/Nav/medium/algo/plpg_ppo_test.yaml --exp medium_lite_plpg_test_seed101 --seed 101 --checkpoint_path checkpoints/medium_lite_plpg_seed101/best_model.zip --log_dir checkpoints/medium_lite_plpg_seed101
python main.py --use_gym --config config/tasks/Nav/medium/algo/plpg_ppo_test.yaml --exp medium_lite_plpg_test_seed202 --seed 202 --checkpoint_path checkpoints/medium_lite_plpg_seed202/best_model.zip --log_dir checkpoints/medium_lite_plpg_seed202
python main.py --use_gym --config config/tasks/Nav/medium/algo/plpg_ppo_test.yaml --exp medium_lite_plpg_test_seed303 --seed 303 --checkpoint_path checkpoints/medium_lite_plpg_seed303/best_model.zip --log_dir checkpoints/medium_lite_plpg_seed303
```

### Hard PPO Test

```powershell
python main.py --use_gym --config config/tasks/Nav/hard/algo/ppo_lite_test.yaml --exp hard_lite_ppo_test_seed101 --seed 101 --checkpoint_path checkpoints/hard_lite_ppo_seed101/best_model.zip --log_dir checkpoints/hard_lite_ppo_seed101
python main.py --use_gym --config config/tasks/Nav/hard/algo/ppo_lite_test.yaml --exp hard_lite_ppo_test_seed202 --seed 202 --checkpoint_path checkpoints/hard_lite_ppo_seed202/best_model.zip --log_dir checkpoints/hard_lite_ppo_seed202
python main.py --use_gym --config config/tasks/Nav/hard/algo/ppo_lite_test.yaml --exp hard_lite_ppo_test_seed303 --seed 303 --checkpoint_path checkpoints/hard_lite_ppo_seed303/best_model.zip --log_dir checkpoints/hard_lite_ppo_seed303
```

### Hard PLPG Test

```powershell
python main.py --use_gym --config config/tasks/Nav/hard/algo/plpg_ppo_test.yaml --exp hard_lite_plpg_test_seed101 --seed 101 --checkpoint_path checkpoints/hard_lite_plpg_seed101/best_model.zip --log_dir checkpoints/hard_lite_plpg_seed101
python main.py --use_gym --config config/tasks/Nav/hard/algo/plpg_ppo_test.yaml --exp hard_lite_plpg_test_seed202 --seed 202 --checkpoint_path checkpoints/hard_lite_plpg_seed202/best_model.zip --log_dir checkpoints/hard_lite_plpg_seed202
python main.py --use_gym --config config/tasks/Nav/hard/algo/plpg_ppo_test.yaml --exp hard_lite_plpg_test_seed303 --seed 303 --checkpoint_path checkpoints/hard_lite_plpg_seed303/best_model.zip --log_dir checkpoints/hard_lite_plpg_seed303
```

## 5. Plotting

After the runs are complete, pass all eval CSVs and test CSVs into the plotting script.

Example with one seed per difficulty/method:

```powershell
python tools/plot_experiment_family1.py --eval_csvs checkpoints/easy_lite_ppo_seed101/easy_lite_ppo_seed101_eval_metrics.csv checkpoints/easy_lite_plpg_seed101/easy_lite_plpg_seed101_eval_metrics.csv checkpoints/medium_lite_ppo_seed101/medium_lite_ppo_seed101_eval_metrics.csv checkpoints/medium_lite_plpg_seed101/medium_lite_plpg_seed101_eval_metrics.csv checkpoints/hard_lite_ppo_seed101/hard_lite_ppo_seed101_eval_metrics.csv checkpoints/hard_lite_plpg_seed101/hard_lite_plpg_seed101_eval_metrics.csv --test_csvs checkpoints/easy_lite_ppo_seed101/easy_lite_ppo_test_seed101_test_metrics.csv checkpoints/easy_lite_plpg_seed101/easy_lite_plpg_test_seed101_test_metrics.csv checkpoints/medium_lite_ppo_seed101/medium_lite_ppo_test_seed101_test_metrics.csv checkpoints/medium_lite_plpg_seed101/medium_lite_plpg_test_seed101_test_metrics.csv checkpoints/hard_lite_ppo_seed101/hard_lite_ppo_test_seed101_test_metrics.csv checkpoints/hard_lite_plpg_seed101/hard_lite_plpg_test_seed101_test_metrics.csv --output_dir plots/experiment_family_1
```

For the final thesis plots, pass all seed-specific CSVs for all methods and difficulties.
