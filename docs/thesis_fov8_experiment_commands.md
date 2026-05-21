# Thesis FOV8 Experiment Commands

These commands assume the current patched setup:

- RL ego runtime FOV = `8`
- rule-based / expert background runtime FOV = `25`
- `IsAhead` capped at `3..4`
- `IsCloseAhead` at `3`
- non-joint PLS uses the original ego-only shield
- joint PLS uses privileged global intersection facts plus local ahead facts

## 0. Shared-Test Annotation

Run this once before any shared-policy evaluation:

```powershell
python tools/annotate_shared_oracle_steps.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --input_path dataset/thesis/main_3cars_1ped/test_50_episodes.pkl --output_path dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --seed 2 --log_dir log_rl_shared_oracle_annotate --exp thesis_real_shared_oracle_annotate_fov8
```

## 1. Non-Joint PLS

### Train 3 seeds

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml --exp thesis_pls_real_fov8_s1 --seed 1 --use_gym
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml --exp thesis_pls_real_fov8_s2 --seed 2 --use_gym
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml --exp thesis_pls_real_fov8_s3 --seed 3 --use_gym
```

### Plot 3-seed training comparison

```powershell
python tools/plot_thesis_results.py --training "PLS s1=checkpoints/thesis_pls_real_fov8_s1/thesis_pls_real_fov8_s1_metrics.csv" --training "PLS s2=checkpoints/thesis_pls_real_fov8_s2/thesis_pls_real_fov8_s2_metrics.csv" --training "PLS s3=checkpoints/thesis_pls_real_fov8_s3/thesis_pls_real_fov8_s3_metrics.csv" --output-dir vis/pls_real_fov8_3seed_train
```

### Single-agent tests

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_eval.yaml --exp thesis_pls_real_fov8_s1_test --seed 1 --checkpoint_path checkpoints\thesis_pls_real_fov8_s1\best_model --use_gym --log_dir log_rl_thesis_pls_real_fov8_s1_test
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_eval.yaml --exp thesis_pls_real_fov8_s2_test --seed 2 --checkpoint_path checkpoints\thesis_pls_real_fov8_s2\best_model --use_gym --log_dir log_rl_thesis_pls_real_fov8_s2_test
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_eval.yaml --exp thesis_pls_real_fov8_s3_test --seed 3 --checkpoint_path checkpoints\thesis_pls_real_fov8_s3\best_model --use_gym --log_dir log_rl_thesis_pls_real_fov8_s3_test
```

### Multi-agent tests

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval.yaml --exp thesis_pls_real_fov8_s1_shared_test --seed 1 --checkpoint_path checkpoints\thesis_pls_real_fov8_s1\best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --log_dir log_rl_thesis_pls_real_fov8_s1_shared_test
```

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval.yaml --exp thesis_pls_real_fov8_s2_shared_test --seed 2 --checkpoint_path checkpoints\thesis_pls_real_fov8_s2\best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --log_dir log_rl_thesis_pls_real_fov8_s2_shared_test
```

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval.yaml --exp thesis_pls_real_fov8_s3_shared_test --seed 3 --checkpoint_path checkpoints\thesis_pls_real_fov8_s3\best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --log_dir log_rl_thesis_pls_real_fov8_s3_shared_test
```

## 2. PPO

### Train 3 seeds

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train.yaml --exp thesis_ppo_real_fov8_s1 --seed 1 --use_gym
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train.yaml --exp thesis_ppo_real_fov8_s2 --seed 2 --use_gym
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train.yaml --exp thesis_ppo_real_fov8_s3 --seed 3 --use_gym
```

### Plot 3-seed training comparison

```powershell
python tools/plot_thesis_results.py --training "PPO s1=checkpoints/thesis_ppo_real_fov8_s1/thesis_ppo_real_fov8_s1_metrics.csv" --training "PPO s2=checkpoints/thesis_ppo_real_fov8_s2/thesis_ppo_real_fov8_s2_metrics.csv" --training "PPO s3=checkpoints/thesis_ppo_real_fov8_s3/thesis_ppo_real_fov8_s3_metrics.csv" --output-dir vis/ppo_real_fov8_3seed_train
```

### Single-agent tests

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_eval.yaml --exp thesis_ppo_real_fov8_s1_test --seed 1 --checkpoint_path checkpoints\thesis_ppo_real_fov8_s1\best_model --use_gym --log_dir log_rl_thesis_ppo_real_fov8_s1_test
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_eval.yaml --exp thesis_ppo_real_fov8_s2_test --seed 2 --checkpoint_path checkpoints\thesis_ppo_real_fov8_s2\best_model --use_gym --log_dir log_rl_thesis_ppo_real_fov8_s2_test
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_eval.yaml --exp thesis_ppo_real_fov8_s3_test --seed 3 --checkpoint_path checkpoints\thesis_ppo_real_fov8_s3\best_model --use_gym --log_dir log_rl_thesis_ppo_real_fov8_s3_test
```

### Multi-agent tests

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --exp thesis_ppo_real_fov8_s1_shared_test --seed 1 --checkpoint_path checkpoints\thesis_ppo_real_fov8_s1\best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --log_dir log_rl_thesis_ppo_real_fov8_s1_shared_test
```

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --exp thesis_ppo_real_fov8_s2_shared_test --seed 2 --checkpoint_path checkpoints\thesis_ppo_real_fov8_s2\best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --log_dir log_rl_thesis_ppo_real_fov8_s2_shared_test
```

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --exp thesis_ppo_real_fov8_s3_shared_test --seed 3 --checkpoint_path checkpoints\thesis_ppo_real_fov8_s3\best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --log_dir log_rl_thesis_ppo_real_fov8_s3_shared_test
```

## 3. PPO vs Non-Joint PLS

### Average training comparison

```powershell
python tools/plot_thesis_results.py --avg-training "PPO=checkpoints/thesis_ppo_real_fov8_s1/thesis_ppo_real_fov8_s1_metrics.csv,checkpoints/thesis_ppo_real_fov8_s2/thesis_ppo_real_fov8_s2_metrics.csv,checkpoints/thesis_ppo_real_fov8_s3/thesis_ppo_real_fov8_s3_metrics.csv" --avg-training "PPO + PLS=checkpoints/thesis_pls_real_fov8_s1/thesis_pls_real_fov8_s1_metrics.csv,checkpoints/thesis_pls_real_fov8_s2/thesis_pls_real_fov8_s2_metrics.csv,checkpoints/thesis_pls_real_fov8_s3/thesis_pls_real_fov8_s3_metrics.csv" --output-dir vis/ppo_vs_pls_real_fov8_train_avg3
```

### Average single-agent test summary

```powershell
python tools/plot_thesis_results.py --avg-single-test "PPO=results/thesis_ppo_real_fov8_s1_test/test_metrics.csv,results/thesis_ppo_real_fov8_s2_test/test_metrics.csv,results/thesis_ppo_real_fov8_s3_test/test_metrics.csv" --avg-single-test "PPO + PLS=results/thesis_pls_real_fov8_s1_test/test_metrics.csv,results/thesis_pls_real_fov8_s2_test/test_metrics.csv,results/thesis_pls_real_fov8_s3_test/test_metrics.csv" --output-dir vis/ppo_vs_pls_real_fov8_test_avg3
```

### Average shared-policy test summary

```powershell
python tools/plot_thesis_results.py --avg-shared-test "PPO=results/thesis_ppo_real_fov8_s1_shared_test/summary_metrics.csv,results/thesis_ppo_real_fov8_s2_shared_test/summary_metrics.csv,results/thesis_ppo_real_fov8_s3_shared_test/summary_metrics.csv" --avg-shared-test "PPO + PLS=results/thesis_pls_real_fov8_s1_shared_test/summary_metrics.csv,results/thesis_pls_real_fov8_s2_shared_test/summary_metrics.csv,results/thesis_pls_real_fov8_s3_shared_test/summary_metrics.csv" --output-dir vis/ppo_vs_pls_real_fov8_shared_avg3
```

### Two-sided bar chart

```powershell
python tools/plot_two_sided_tsr_barchart.py --single_csv vis/ppo_vs_pls_real_fov8_test_avg3/single_agent_test_summary.csv --multi_csv vis/ppo_vs_pls_real_fov8_shared_avg3/shared_policy_test_summary.csv --output vis/ppo_vs_pls_real_fov8_two_sided.png --title "PPO vs PLS TSR: Single vs Multi"
```

## 4. Joint PLS

### Train 3 seeds

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_joint_pls_single_train.yaml --exp thesis_joint_pls_real_fov8_s1 --seed 1 --use_gym
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_joint_pls_single_train.yaml --exp thesis_joint_pls_real_fov8_s2 --seed 2 --use_gym
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_joint_pls_single_train.yaml --exp thesis_joint_pls_real_fov8_s3 --seed 3 --use_gym
```

### Plot 3-seed training comparison

```powershell
python tools/plot_thesis_results.py --training "Joint PLS s1=checkpoints/thesis_joint_pls_real_fov8_s1/thesis_joint_pls_real_fov8_s1_metrics.csv" --training "Joint PLS s2=checkpoints/thesis_joint_pls_real_fov8_s2/thesis_joint_pls_real_fov8_s2_metrics.csv" --training "Joint PLS s3=checkpoints/thesis_joint_pls_real_fov8_s3/thesis_joint_pls_real_fov8_s3_metrics.csv" --output-dir vis/joint_pls_real_fov8_3seed_train
```

### Single-agent tests

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_joint_pls_single_eval.yaml --exp thesis_joint_pls_real_fov8_s1_test --seed 1 --checkpoint_path checkpoints\thesis_joint_pls_real_fov8_s1\best_model --use_gym --log_dir log_rl_thesis_joint_pls_real_fov8_s1_test
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_joint_pls_single_eval.yaml --exp thesis_joint_pls_real_fov8_s2_test --seed 2 --checkpoint_path checkpoints\thesis_joint_pls_real_fov8_s2\best_model --use_gym --log_dir log_rl_thesis_joint_pls_real_fov8_s2_test
```

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_joint_pls_single_eval.yaml --exp thesis_joint_pls_real_fov8_s3_test --seed 3 --checkpoint_path checkpoints\thesis_joint_pls_real_fov8_s3\best_model --use_gym --log_dir log_rl_thesis_joint_pls_real_fov8_s3_test
```

### Multi-agent tests

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_joint_pls_shared_eval.yaml --exp thesis_joint_pls_real_fov8_s1_shared_test --seed 1 --checkpoint_path checkpoints\thesis_joint_pls_real_fov8_s1\best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --log_dir log_rl_thesis_joint_pls_real_fov8_s1_shared_test
```

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_joint_pls_shared_eval.yaml --exp thesis_joint_pls_real_fov8_s2_shared_test --seed 2 --checkpoint_path checkpoints\thesis_joint_pls_real_fov8_s2\best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --log_dir log_rl_thesis_joint_pls_real_fov8_s2_shared_test
```

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_joint_pls_shared_eval.yaml --exp thesis_joint_pls_real_fov8_s3_shared_test --seed 3 --checkpoint_path checkpoints\thesis_joint_pls_real_fov8_s3\best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --log_dir log_rl_thesis_joint_pls_real_fov8_s3_shared_test
```

## 5. PPO vs Joint PLS

### Average training comparison

```powershell
python tools/plot_thesis_results.py --avg-training "PPO=checkpoints/thesis_ppo_real_fov8_s1/thesis_ppo_real_fov8_s1_metrics.csv,checkpoints/thesis_ppo_real_fov8_s2/thesis_ppo_real_fov8_s2_metrics.csv,checkpoints/thesis_ppo_real_fov8_s3/thesis_ppo_real_fov8_s3_metrics.csv" --avg-training "Joint PLS=checkpoints/thesis_joint_pls_real_fov8_s1/thesis_joint_pls_real_fov8_s1_metrics.csv,checkpoints/thesis_joint_pls_real_fov8_s2/thesis_joint_pls_real_fov8_s2_metrics.csv,checkpoints/thesis_joint_pls_real_fov8_s3/thesis_joint_pls_real_fov8_s3_metrics.csv" --output-dir vis/ppo_vs_joint_pls_real_fov8_train_avg3
```

## 6. Final PPO vs PLS vs Joint PLS

### Average single-agent test summary

```powershell
python tools/plot_thesis_results.py --avg-single-test "PPO=results/thesis_ppo_real_fov8_s1_test/test_metrics.csv,results/thesis_ppo_real_fov8_s2_test/test_metrics.csv,results/thesis_ppo_real_fov8_s3_test/test_metrics.csv" --avg-single-test "PPO + PLS=results/thesis_pls_real_fov8_s1_test/test_metrics.csv,results/thesis_pls_real_fov8_s2_test/test_metrics.csv,results/thesis_pls_real_fov8_s3_test/test_metrics.csv" --avg-single-test "Joint PLS=results/thesis_joint_pls_real_fov8_s1_test/test_metrics.csv,results/thesis_joint_pls_real_fov8_s2_test/test_metrics.csv,results/thesis_joint_pls_real_fov8_s3_test/test_metrics.csv" --output-dir vis/ppo_pls_jointpls_real_fov8_test_avg3
```

### Average shared-policy test summary

```powershell
python tools/plot_thesis_results.py --avg-shared-test "PPO=results/thesis_ppo_real_fov8_s1_shared_test/summary_metrics.csv,results/thesis_ppo_real_fov8_s2_shared_test/summary_metrics.csv,results/thesis_ppo_real_fov8_s3_shared_test/summary_metrics.csv" --avg-shared-test "PPO + PLS=results/thesis_pls_real_fov8_s1_shared_test/summary_metrics.csv,results/thesis_pls_real_fov8_s2_shared_test/summary_metrics.csv,results/thesis_pls_real_fov8_s3_shared_test/summary_metrics.csv" --avg-shared-test "Joint PLS=results/thesis_joint_pls_real_fov8_s1_shared_test/summary_metrics.csv,results/thesis_joint_pls_real_fov8_s2_shared_test/summary_metrics.csv,results/thesis_joint_pls_real_fov8_s3_shared_test/summary_metrics.csv" --output-dir vis/ppo_pls_jointpls_real_fov8_shared_avg3
```

### Final two-sided bar chart

```powershell
python tools/plot_two_sided_tsr_barchart.py --single_csv vis/ppo_pls_jointpls_real_fov8_test_avg3/single_agent_test_summary.csv --multi_csv vis/ppo_pls_jointpls_real_fov8_shared_avg3/shared_policy_test_summary.csv --output vis/ppo_pls_jointpls_real_fov8_two_sided.png --title "PPO vs PLS vs Joint PLS TSR: Single vs Multi"
```

## 7. Joint PLS Smoke Quick Start

### Train smoke joint PLS

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_joint_pls_single_train_smoke.yaml --exp thesis_joint_pls_smoke_fov8_s1 --seed 1 --use_gym
```

### Test smoke joint PLS

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_joint_pls_single_eval_smoke.yaml --exp thesis_joint_pls_smoke_fov8_s1_test --seed 1 --checkpoint_path checkpoints\thesis_joint_pls_smoke_fov8_s1\best_model --use_gym --log_dir log_rl_thesis_joint_pls_smoke_fov8_s1_test
```
