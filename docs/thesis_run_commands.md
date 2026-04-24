# Thesis Run Commands

This runbook reflects the current thesis setup after removing the uncertainty and noise variants. The canonical configs are the non-`_perfect` thesis YAMLs under `config/tasks/Nav/thesis/algo/`.

The current frozen executable benchmark is:

- `PPO`
- `PPO + DLS`
- `PPO + PLS` with the current ProbLog-backed implementation

The planned `PPO + PLS` joint-shield concept is not included below because it is not yet implemented as a stable experiment pipeline.

Assumptions:

- run from the repo root: `d:\Dev\LogicityFresh\LogiCity`
- use the project Python environment
- final common training budget is `40000` timesteps
- evaluation frequency is `1000`
- use `3` seeds per method

## 1. Dataset Generation

```powershell
python tools/create_single_agent_two_agent_car_ped_region_episode.py --config config/tasks/Nav/thesis/experts/expert_episode_train.yaml --exp thesis_train_100 --seed 111 --max_episodes 100 --output_path dataset/thesis/main_3cars_1ped/train_100_episodes.pkl --log_dir log_rl
python tools/create_single_agent_two_agent_car_ped_region_episode.py --config config/tasks/Nav/thesis/experts/expert_episode_val.yaml --exp thesis_val_20 --seed 222 --max_episodes 20 --save_worlds --worlds_dir log_rl/thesis_dataset_worlds/val_20_worlds --output_path dataset/thesis/main_3cars_1ped/val_20_episodes.pkl --log_dir log_rl
python tools/create_single_agent_two_agent_car_ped_region_episode.py --config config/tasks/Nav/thesis/experts/expert_episode_test.yaml --exp thesis_test_50 --seed 333 --max_episodes 50 --save_worlds --worlds_dir log_rl/thesis_dataset_worlds/test_50_worlds --output_path dataset/thesis/main_3cars_1ped/test_50_episodes.pkl --log_dir log_rl
```

## 2. Shared-Oracle Annotation

```powershell
python tools/annotate_shared_oracle_steps.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --input_path dataset/thesis/main_3cars_1ped/val_20_episodes.pkl --output_path dataset/thesis/main_3cars_1ped/val_20_episodes_shared_oracle.pkl --log_dir log_rl --exp annotate_shared_oracle_val
python tools/annotate_shared_oracle_steps.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --input_path dataset/thesis/main_3cars_1ped/test_50_episodes.pkl --output_path dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --log_dir log_rl --exp annotate_shared_oracle_test
```

## 3. Expert Reference Runs

```powershell
python main.py --config config/tasks/Nav/thesis/algo/expert_single_val.yaml --exp thesis_expert_val --use_gym --save_steps --log_dir log_rl/thesis_expert_val
python main.py --config config/tasks/Nav/thesis/algo/expert_single_test.yaml --exp thesis_expert_test --use_gym --save_steps --log_dir log_rl/thesis_expert_test
```

## 4. Final Training Runs

Important:

- old PLS results from before the ProbLog alignment are obsolete for final reporting
- PLS training, single-agent testing, and shared-policy testing must be rerun from fresh checkpoints
- existing `PPO` and `PPO + DLS` runs can be reused if they were already produced with the frozen perfect-sensor setup

### PPO

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train.yaml --exp thesis_ppo_single_s1 --use_gym --log_dir log_rl/thesis_train_ppo_s1 --seed 111
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train.yaml --exp thesis_ppo_single_s2 --use_gym --log_dir log_rl/thesis_train_ppo_s2 --seed 222
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train.yaml --exp thesis_ppo_single_s3 --use_gym --log_dir log_rl/thesis_train_ppo_s3 --seed 333
```

### PPO + DLS

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_train.yaml --exp thesis_ppo_dls_single_s1 --use_gym --log_dir log_rl/thesis_train_dls_s1 --seed 111
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_train.yaml --exp thesis_ppo_dls_single_s2 --use_gym --log_dir log_rl/thesis_train_dls_s2 --seed 222
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_train.yaml --exp thesis_ppo_dls_single_s3 --use_gym --log_dir log_rl/thesis_train_dls_s3 --seed 333
```

### PPO + PLS

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml --exp thesis_ppo_pls_single_s1 --use_gym --log_dir log_rl/thesis_train_pls_s1 --seed 111
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml --exp thesis_ppo_pls_single_s2 --use_gym --log_dir log_rl/thesis_train_pls_s2 --seed 222
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml --exp thesis_ppo_pls_single_s3 --use_gym --log_dir log_rl/thesis_train_pls_s3 --seed 333
```

These are the mandatory replacement runs for the updated ProbLog-based PLS branch.

## 5. Single-Agent Test Runs

### PPO

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_test.yaml --exp thesis_ppo_single_s1_test --use_gym --log_dir log_rl/thesis_test_ppo_s1 --checkpoint_path checkpoints/thesis_ppo_single_s1/best_model
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_test.yaml --exp thesis_ppo_single_s2_test --use_gym --log_dir log_rl/thesis_test_ppo_s2 --checkpoint_path checkpoints/thesis_ppo_single_s2/best_model
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_test.yaml --exp thesis_ppo_single_s3_test --use_gym --log_dir log_rl/thesis_test_ppo_s3 --checkpoint_path checkpoints/thesis_ppo_single_s3/best_model
```

### PPO + DLS

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_test.yaml --exp thesis_ppo_dls_single_s1_test --use_gym --log_dir log_rl/thesis_test_dls_s1 --checkpoint_path checkpoints/thesis_ppo_dls_single_s1/best_model
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_test.yaml --exp thesis_ppo_dls_single_s2_test --use_gym --log_dir log_rl/thesis_test_dls_s2 --checkpoint_path checkpoints/thesis_ppo_dls_single_s2/best_model
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_test.yaml --exp thesis_ppo_dls_single_s3_test --use_gym --log_dir log_rl/thesis_test_dls_s3 --checkpoint_path checkpoints/thesis_ppo_dls_single_s3/best_model
```

### PPO + PLS

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_test.yaml --exp thesis_ppo_pls_single_s1_test --use_gym --log_dir log_rl/thesis_test_pls_s1 --checkpoint_path checkpoints/thesis_ppo_pls_single_s1/best_model
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_test.yaml --exp thesis_ppo_pls_single_s2_test --use_gym --log_dir log_rl/thesis_test_pls_s2 --checkpoint_path checkpoints/thesis_ppo_pls_single_s2/best_model
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_test.yaml --exp thesis_ppo_pls_single_s3_test --use_gym --log_dir log_rl/thesis_test_pls_s3 --checkpoint_path checkpoints/thesis_ppo_pls_single_s3/best_model
```

## 6. Shared-Policy Multi-Agent Test Runs

### PPO

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --exp thesis_ppo_single_s1_shared_test --log_dir log_rl/thesis_shared_test_ppo_s1 --checkpoint_path checkpoints/thesis_ppo_single_s1/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --exp thesis_ppo_single_s2_shared_test --log_dir log_rl/thesis_shared_test_ppo_s2 --checkpoint_path checkpoints/thesis_ppo_single_s2/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --exp thesis_ppo_single_s3_shared_test --log_dir log_rl/thesis_shared_test_ppo_s3 --checkpoint_path checkpoints/thesis_ppo_single_s3/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
```

### PPO + DLS

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_dls_shared_eval.yaml --exp thesis_ppo_dls_single_s1_shared_test --log_dir log_rl/thesis_shared_test_dls_s1 --checkpoint_path checkpoints/thesis_ppo_dls_single_s1/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_dls_shared_eval.yaml --exp thesis_ppo_dls_single_s2_shared_test --log_dir log_rl/thesis_shared_test_dls_s2 --checkpoint_path checkpoints/thesis_ppo_dls_single_s2/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_dls_shared_eval.yaml --exp thesis_ppo_dls_single_s3_shared_test --log_dir log_rl/thesis_shared_test_dls_s3 --checkpoint_path checkpoints/thesis_ppo_dls_single_s3/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
```

### PPO + PLS

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval.yaml --exp thesis_ppo_pls_single_s1_shared_test --log_dir log_rl/thesis_shared_test_pls_s1 --checkpoint_path checkpoints/thesis_ppo_pls_single_s1/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval.yaml --exp thesis_ppo_pls_single_s2_shared_test --log_dir log_rl/thesis_shared_test_pls_s2 --checkpoint_path checkpoints/thesis_ppo_pls_single_s2/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval.yaml --exp thesis_ppo_pls_single_s3_shared_test --log_dir log_rl/thesis_shared_test_pls_s3 --checkpoint_path checkpoints/thesis_ppo_pls_single_s3/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
```

## 7. Plotting

The main reporting set should include:

- `TSR` learning curve
- `train_fail_since_last_eval` learning curve
- cumulative `train_fail` learning curve
- `mean_reward` learning curve
- final single-agent summary
- final shared-policy summary with per-agent TSR
- TSR convergence summary from existing training CSVs

### Training-only comparison across methods using 3 seeds each

Use this after all three training seeds exist for each method, even if the new PLS test runs are not finished yet.

```powershell
python tools/plot_thesis_results.py `
  --avg-training "PPO=checkpoints/thesis_ppo_single_s1/thesis_ppo_single_s1_metrics.csv,checkpoints/thesis_ppo_single_s2/thesis_ppo_single_s2_metrics.csv,checkpoints/thesis_ppo_single_s3/thesis_ppo_single_s3_metrics.csv" `
  --avg-training "PPO + DLS=checkpoints/thesis_ppo_dls_single_s1/thesis_ppo_dls_single_s1_metrics.csv,checkpoints/thesis_ppo_dls_single_s2/thesis_ppo_dls_single_s2_metrics.csv,checkpoints/thesis_ppo_dls_single_s3/thesis_ppo_dls_single_s3_metrics.csv" `
  --avg-training "PPO + PLS=checkpoints/thesis_ppo_pls_single_s1/thesis_ppo_pls_single_s1_metrics.csv,checkpoints/thesis_ppo_pls_single_s2/thesis_ppo_pls_single_s2_metrics.csv,checkpoints/thesis_ppo_pls_single_s3/thesis_ppo_pls_single_s3_metrics.csv" `
  --convergence-threshold 0.70 `
  --convergence-threshold 0.80 `
  --convergence-threshold 0.90 `
  --output-dir vis/thesis_results_training_avg_3seeds
```

### Full comparison across methods using 3 seeds each

Use this after the single-agent and shared-policy test CSVs are also available for all three methods.

```powershell
python tools/plot_thesis_results.py `
  --avg-training "PPO=checkpoints/thesis_ppo_single_s1/thesis_ppo_single_s1_metrics.csv,checkpoints/thesis_ppo_single_s2/thesis_ppo_single_s2_metrics.csv,checkpoints/thesis_ppo_single_s3/thesis_ppo_single_s3_metrics.csv" `
  --avg-training "PPO + DLS=checkpoints/thesis_ppo_dls_single_s1/thesis_ppo_dls_single_s1_metrics.csv,checkpoints/thesis_ppo_dls_single_s2/thesis_ppo_dls_single_s2_metrics.csv,checkpoints/thesis_ppo_dls_single_s3/thesis_ppo_dls_single_s3_metrics.csv" `
  --avg-training "PPO + PLS=checkpoints/thesis_ppo_pls_single_s1/thesis_ppo_pls_single_s1_metrics.csv,checkpoints/thesis_ppo_pls_single_s2/thesis_ppo_pls_single_s2_metrics.csv,checkpoints/thesis_ppo_pls_single_s3/thesis_ppo_pls_single_s3_metrics.csv" `
  --avg-single-test "PPO=results/thesis_ppo_single_s1_test/test_metrics.csv,results/thesis_ppo_single_s2_test/test_metrics.csv,results/thesis_ppo_single_s3_test/test_metrics.csv" `
  --avg-single-test "PPO + DLS=results/thesis_ppo_dls_single_s1_test/test_metrics.csv,results/thesis_ppo_dls_single_s2_test/test_metrics.csv,results/thesis_ppo_dls_single_s3_test/test_metrics.csv" `
  --avg-single-test "PPO + PLS=results/thesis_ppo_pls_single_s1_test/test_metrics.csv,results/thesis_ppo_pls_single_s2_test/test_metrics.csv,results/thesis_ppo_pls_single_s3_test/test_metrics.csv" `
  --avg-shared-test "PPO=results/thesis_ppo_single_s1_shared_test/summary_metrics.csv,results/thesis_ppo_single_s2_shared_test/summary_metrics.csv,results/thesis_ppo_single_s3_shared_test/summary_metrics.csv" `
  --avg-shared-test "PPO + DLS=results/thesis_ppo_dls_single_s1_shared_test/summary_metrics.csv,results/thesis_ppo_dls_single_s2_shared_test/summary_metrics.csv,results/thesis_ppo_dls_single_s3_shared_test/summary_metrics.csv" `
  --avg-shared-test "PPO + PLS=results/thesis_ppo_pls_single_s1_shared_test/summary_metrics.csv,results/thesis_ppo_pls_single_s2_shared_test/summary_metrics.csv,results/thesis_ppo_pls_single_s3_shared_test/summary_metrics.csv" `
  --convergence-threshold 0.70 `
  --convergence-threshold 0.80 `
  --convergence-threshold 0.90 `
  --output-dir vis/thesis_results_avg_3seeds
```

### Shared-policy comparison across methods using 3 seeds each

Use this if you want only the shared-policy comparison figure/table set.

```powershell
python tools/plot_thesis_results.py `
  --avg-shared-test "PPO=results/thesis_ppo_single_s1_shared_test/summary_metrics.csv,results/thesis_ppo_single_s2_shared_test/summary_metrics.csv,results/thesis_ppo_single_s3_shared_test/summary_metrics.csv" `
  --avg-shared-test "PPO + DLS=results/thesis_ppo_dls_single_s1_shared_test/summary_metrics.csv,results/thesis_ppo_dls_single_s2_shared_test/summary_metrics.csv,results/thesis_ppo_dls_single_s3_shared_test/summary_metrics.csv" `
  --avg-shared-test "PPO + PLS=results/thesis_ppo_pls_single_s1_shared_test/summary_metrics.csv,results/thesis_ppo_pls_single_s2_shared_test/summary_metrics.csv,results/thesis_ppo_pls_single_s3_shared_test/summary_metrics.csv" `
  --output-dir vis/thesis_results_shared_avg_3seeds
```

## 8. Notes

- `best_model.zip` is the file to use for final testing.
- for `main.py --checkpoint_path`, pass `.../best_model` without the `.zip` suffix
- for `tools/run_shared_policy_two_agent.py --checkpoint_path`, pass the actual `.zip` file
- Shared-policy testing should use the `*_shared_oracle.pkl` dataset.
- The repo no longer supports the old uncertainty/perfect config split.
- The current joint PLS idea is still a planning item, so there are intentionally no run commands for it yet.
- Any PLS results produced before the ProbLog-backed alignment should be treated as exploratory only.
- `tools/plot_thesis_results.py` now exports TSR convergence summaries from the existing training CSVs, so no retraining is needed just to analyze sample efficiency for PPO or DLS.
