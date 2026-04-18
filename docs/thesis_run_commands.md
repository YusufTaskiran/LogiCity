# Thesis Run Commands

This file collects the concrete commands for the frozen thesis setup.

Assumptions:

- run everything from the repo root: `d:\Dev\LogicityFresh\LogiCity`
- active Python environment is the project environment
- final common training budget is `40000` timesteps
- evaluation frequency is `1000`
- each method is trained with `3` seeds

Two scenarios are supported:

- **Perfect-sensor scenario**: the original LogiCity-style setup without injected observation or shield uncertainty
- **Uncertain-sensor scenario**: the thesis setup with observation uncertainty for PPO/DLS/PLS; DLS and PLS consume those observation probabilities directly, without additional shield-side noise

## 1. Dataset Generation

These only need to be run once if the datasets do not already exist.

### Train split

```powershell
python tools/create_single_agent_two_agent_car_ped_region_episode.py --config config/tasks/Nav/thesis/experts/expert_episode_train.yaml --exp thesis_train_100 --seed 111 --max_episodes 100 --output_path dataset/thesis/main_3cars_1ped/train_100_episodes.pkl --log_dir log_rl
```

### Validation split

```powershell
python tools/create_single_agent_two_agent_car_ped_region_episode.py --config config/tasks/Nav/thesis/experts/expert_episode_val.yaml --exp thesis_val_20 --seed 222 --max_episodes 20 --save_worlds --worlds_dir log_rl/thesis_dataset_worlds/val_20_worlds --output_path dataset/thesis/main_3cars_1ped/val_20_episodes.pkl --log_dir log_rl
```

### Test split

```powershell
python tools/create_single_agent_two_agent_car_ped_region_episode.py --config config/tasks/Nav/thesis/experts/expert_episode_test.yaml --exp thesis_test_50 --seed 333 --max_episodes 50 --save_worlds --worlds_dir log_rl/thesis_dataset_worlds/test_50_worlds --output_path dataset/thesis/main_3cars_1ped/test_50_episodes.pkl --log_dir log_rl
```

## 2. Shared-Oracle Annotation For Multi-Agent Evaluation

These produce the corrected shared-policy oracle horizons used in multi-agent evaluation.

### Validation split with shared oracle

```powershell
python tools/annotate_shared_oracle_steps.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --input_path dataset/thesis/main_3cars_1ped/val_20_episodes.pkl --output_path dataset/thesis/main_3cars_1ped/val_20_episodes_shared_oracle.pkl --log_dir log_rl --exp annotate_shared_oracle_val
```

### Test split with shared oracle

```powershell
python tools/annotate_shared_oracle_steps.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --input_path dataset/thesis/main_3cars_1ped/test_50_episodes.pkl --output_path dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl --log_dir log_rl --exp annotate_shared_oracle_test
```

## 3. Expert Reference Runs

### Expert on validation split

```powershell
python main.py --config config/tasks/Nav/thesis/algo/expert_single_val.yaml --exp thesis_expert_val --use_gym --save_steps --log_dir log_rl/thesis_expert_val
```

### Expert on test split

```powershell
python main.py --config config/tasks/Nav/thesis/algo/expert_single_test.yaml --exp thesis_expert_test --use_gym --save_steps --log_dir log_rl/thesis_expert_test
```

## 4. Final Training Runs

### 4A. Perfect-Sensor Scenario

#### PPO, seeds 1-3

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train_perfect.yaml --exp thesis_perfect_ppo_single_s1 --use_gym --log_dir log_rl/thesis_perfect_train_ppo_s1 --seed 111
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train_perfect.yaml --exp thesis_perfect_ppo_single_s2 --use_gym --log_dir log_rl/thesis_perfect_train_ppo_s2 --seed 222
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train_perfect.yaml --exp thesis_perfect_ppo_single_s3 --use_gym --log_dir log_rl/thesis_perfect_train_ppo_s3 --seed 333
```

#### PPO + DLS, seeds 1-3

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_train_perfect.yaml --exp thesis_perfect_ppo_dls_single_s1 --use_gym --log_dir log_rl/thesis_perfect_train_dls_s1 --seed 111
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_train_perfect.yaml --exp thesis_perfect_ppo_dls_single_s2 --use_gym --log_dir log_rl/thesis_perfect_train_dls_s2 --seed 222
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_train_perfect.yaml --exp thesis_perfect_ppo_dls_single_s3 --use_gym --log_dir log_rl/thesis_perfect_train_dls_s3 --seed 333
```

#### PPO + PLS/PLPG, seeds 1-3

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train_perfect.yaml --exp thesis_perfect_ppo_pls_single_s1 --use_gym --log_dir log_rl/thesis_perfect_train_pls_s1 --seed 111
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train_perfect.yaml --exp thesis_perfect_ppo_pls_single_s2 --use_gym --log_dir log_rl/thesis_perfect_train_pls_s2 --seed 222
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train_perfect.yaml --exp thesis_perfect_ppo_pls_single_s3 --use_gym --log_dir log_rl/thesis_perfect_train_pls_s3 --seed 333
```

### 4B. Uncertain-Sensor Scenario

### PPO, seed 1

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train.yaml --exp thesis_uncertain_ppo_single_s1 --use_gym --log_dir log_rl/thesis_uncertain_train_ppo_s1 --seed 111
```

### PPO, seed 2

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train.yaml --exp thesis_uncertain_ppo_single_s2 --use_gym --log_dir log_rl/thesis_uncertain_train_ppo_s2 --seed 222
```

### PPO, seed 3

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_train.yaml --exp thesis_uncertain_ppo_single_s3 --use_gym --log_dir log_rl/thesis_uncertain_train_ppo_s3 --seed 333
```

### PPO + DLS, seed 1

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_train.yaml --exp thesis_uncertain_ppo_dls_single_s1 --use_gym --log_dir log_rl/thesis_uncertain_train_dls_s1 --seed 111
```

### PPO + DLS, seed 2

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_train.yaml --exp thesis_uncertain_ppo_dls_single_s2 --use_gym --log_dir log_rl/thesis_uncertain_train_dls_s2 --seed 222
```

### PPO + DLS, seed 3

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_train.yaml --exp thesis_uncertain_ppo_dls_single_s3 --use_gym --log_dir log_rl/thesis_uncertain_train_dls_s3 --seed 333
```

### PPO + PLS/PLPG, seed 1

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml --exp thesis_uncertain_ppo_pls_single_s1 --use_gym --log_dir log_rl/thesis_uncertain_train_pls_s1 --seed 111
```

### PPO + PLS/PLPG, seed 2

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml --exp thesis_uncertain_ppo_pls_single_s2 --use_gym --log_dir log_rl/thesis_uncertain_train_pls_s2 --seed 222
```

### PPO + PLS/PLPG, seed 3

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml --exp thesis_uncertain_ppo_pls_single_s3 --use_gym --log_dir log_rl/thesis_uncertain_train_pls_s3 --seed 333
```

## 5. Single-Agent Test Runs

### 5A. Perfect-Sensor Scenario

#### PPO single-agent test

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_test_perfect.yaml --exp thesis_perfect_ppo_single_s1_test --use_gym --log_dir log_rl/thesis_perfect_test_ppo_s1 --checkpoint_path checkpoints/thesis_perfect_ppo_single_s1/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_test_perfect.yaml --exp thesis_perfect_ppo_single_s2_test --use_gym --log_dir log_rl/thesis_perfect_test_ppo_s2 --checkpoint_path checkpoints/thesis_perfect_ppo_single_s2/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_test_perfect.yaml --exp thesis_perfect_ppo_single_s3_test --use_gym --log_dir log_rl/thesis_perfect_test_ppo_s3 --checkpoint_path checkpoints/thesis_perfect_ppo_single_s3/best_model.zip
```

#### PPO + DLS single-agent test

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_test_perfect.yaml --exp thesis_perfect_ppo_dls_single_s1_test --use_gym --log_dir log_rl/thesis_perfect_test_dls_s1 --checkpoint_path checkpoints/thesis_perfect_ppo_dls_single_s1/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_test_perfect.yaml --exp thesis_perfect_ppo_dls_single_s2_test --use_gym --log_dir log_rl/thesis_perfect_test_dls_s2 --checkpoint_path checkpoints/thesis_perfect_ppo_dls_single_s2/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_test_perfect.yaml --exp thesis_perfect_ppo_dls_single_s3_test --use_gym --log_dir log_rl/thesis_perfect_test_dls_s3 --checkpoint_path checkpoints/thesis_perfect_ppo_dls_single_s3/best_model.zip
```

#### PPO + PLS/PLPG single-agent test

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_test_perfect.yaml --exp thesis_perfect_ppo_pls_single_s1_test --use_gym --log_dir log_rl/thesis_perfect_test_pls_s1 --checkpoint_path checkpoints/thesis_perfect_ppo_pls_single_s1/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_test_perfect.yaml --exp thesis_perfect_ppo_pls_single_s2_test --use_gym --log_dir log_rl/thesis_perfect_test_pls_s2 --checkpoint_path checkpoints/thesis_perfect_ppo_pls_single_s2/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_test_perfect.yaml --exp thesis_perfect_ppo_pls_single_s3_test --use_gym --log_dir log_rl/thesis_perfect_test_pls_s3 --checkpoint_path checkpoints/thesis_perfect_ppo_pls_single_s3/best_model.zip
```

### 5B. Uncertain-Sensor Scenario

### PPO single-agent test

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_test.yaml --exp thesis_uncertain_ppo_single_s1_test --use_gym --log_dir log_rl/thesis_uncertain_test_ppo_s1 --checkpoint_path checkpoints/thesis_uncertain_ppo_single_s1/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_test.yaml --exp thesis_uncertain_ppo_single_s2_test --use_gym --log_dir log_rl/thesis_uncertain_test_ppo_s2 --checkpoint_path checkpoints/thesis_uncertain_ppo_single_s2/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_single_test.yaml --exp thesis_uncertain_ppo_single_s3_test --use_gym --log_dir log_rl/thesis_uncertain_test_ppo_s3 --checkpoint_path checkpoints/thesis_uncertain_ppo_single_s3/best_model.zip
```

### PPO + DLS single-agent test

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_test.yaml --exp thesis_uncertain_ppo_dls_single_s1_test --use_gym --log_dir log_rl/thesis_uncertain_test_dls_s1 --checkpoint_path checkpoints/thesis_uncertain_ppo_dls_single_s1/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_test.yaml --exp thesis_uncertain_ppo_dls_single_s2_test --use_gym --log_dir log_rl/thesis_uncertain_test_dls_s2 --checkpoint_path checkpoints/thesis_uncertain_ppo_dls_single_s2/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_dls_single_test.yaml --exp thesis_uncertain_ppo_dls_single_s3_test --use_gym --log_dir log_rl/thesis_uncertain_test_dls_s3 --checkpoint_path checkpoints/thesis_uncertain_ppo_dls_single_s3/best_model.zip
```

### PPO + PLS/PLPG single-agent test

```powershell
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_test.yaml --exp thesis_uncertain_ppo_pls_single_s1_test --use_gym --log_dir log_rl/thesis_uncertain_test_pls_s1 --checkpoint_path checkpoints/thesis_uncertain_ppo_pls_single_s1/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_test.yaml --exp thesis_uncertain_ppo_pls_single_s2_test --use_gym --log_dir log_rl/thesis_uncertain_test_pls_s2 --checkpoint_path checkpoints/thesis_uncertain_ppo_pls_single_s2/best_model.zip
python main.py --config config/tasks/Nav/thesis/algo/ppo_pls_single_test.yaml --exp thesis_uncertain_ppo_pls_single_s3_test --use_gym --log_dir log_rl/thesis_uncertain_test_pls_s3 --checkpoint_path checkpoints/thesis_uncertain_ppo_pls_single_s3/best_model.zip
```

## 6. Shared-Policy Multi-Agent Test Runs

### 6A. Perfect-Sensor Scenario

#### PPO shared-policy test

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval_perfect.yaml --exp thesis_perfect_ppo_single_s1_shared_test --log_dir log_rl/thesis_perfect_shared_test_ppo_s1 --checkpoint_path checkpoints/thesis_perfect_ppo_single_s1/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval_perfect.yaml --exp thesis_perfect_ppo_single_s2_shared_test --log_dir log_rl/thesis_perfect_shared_test_ppo_s2 --checkpoint_path checkpoints/thesis_perfect_ppo_single_s2/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval_perfect.yaml --exp thesis_perfect_ppo_single_s3_shared_test --log_dir log_rl/thesis_perfect_shared_test_ppo_s3 --checkpoint_path checkpoints/thesis_perfect_ppo_single_s3/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
```

#### PPO + DLS shared-policy test

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_dls_shared_eval_perfect.yaml --exp thesis_perfect_ppo_dls_single_s1_shared_test --log_dir log_rl/thesis_perfect_shared_test_dls_s1 --checkpoint_path checkpoints/thesis_perfect_ppo_dls_single_s1/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_dls_shared_eval_perfect.yaml --exp thesis_perfect_ppo_dls_single_s2_shared_test --log_dir log_rl/thesis_perfect_shared_test_dls_s2 --checkpoint_path checkpoints/thesis_perfect_ppo_dls_single_s2/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_dls_shared_eval_perfect.yaml --exp thesis_perfect_ppo_dls_single_s3_shared_test --log_dir log_rl/thesis_perfect_shared_test_dls_s3 --checkpoint_path checkpoints/thesis_perfect_ppo_dls_single_s3/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
```

#### PPO + PLS/PLPG shared-policy test

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval_perfect.yaml --exp thesis_perfect_ppo_pls_single_s1_shared_test --log_dir log_rl/thesis_perfect_shared_test_pls_s1 --checkpoint_path checkpoints/thesis_perfect_ppo_pls_single_s1/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval_perfect.yaml --exp thesis_perfect_ppo_pls_single_s2_shared_test --log_dir log_rl/thesis_perfect_shared_test_pls_s2 --checkpoint_path checkpoints/thesis_perfect_ppo_pls_single_s2/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval_perfect.yaml --exp thesis_perfect_ppo_pls_single_s3_shared_test --log_dir log_rl/thesis_perfect_shared_test_pls_s3 --checkpoint_path checkpoints/thesis_perfect_ppo_pls_single_s3/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
```

### 6B. Uncertain-Sensor Scenario

These use the shared-policy runner and the shared-oracle annotated test split.

### PPO shared-policy test

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --exp thesis_uncertain_ppo_single_s1_shared_test --log_dir log_rl/thesis_uncertain_shared_test_ppo_s1 --checkpoint_path checkpoints/thesis_uncertain_ppo_single_s1/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --exp thesis_uncertain_ppo_single_s2_shared_test --log_dir log_rl/thesis_uncertain_shared_test_ppo_s2 --checkpoint_path checkpoints/thesis_uncertain_ppo_single_s2/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml --exp thesis_uncertain_ppo_single_s3_shared_test --log_dir log_rl/thesis_uncertain_shared_test_ppo_s3 --checkpoint_path checkpoints/thesis_uncertain_ppo_single_s3/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
```

### PPO + DLS shared-policy test

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_dls_shared_eval.yaml --exp thesis_uncertain_ppo_dls_single_s1_shared_test --log_dir log_rl/thesis_uncertain_shared_test_dls_s1 --checkpoint_path checkpoints/thesis_uncertain_ppo_dls_single_s1/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_dls_shared_eval.yaml --exp thesis_uncertain_ppo_dls_single_s2_shared_test --log_dir log_rl/thesis_uncertain_shared_test_dls_s2 --checkpoint_path checkpoints/thesis_uncertain_ppo_dls_single_s2/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_dls_shared_eval.yaml --exp thesis_uncertain_ppo_dls_single_s3_shared_test --log_dir log_rl/thesis_uncertain_shared_test_dls_s3 --checkpoint_path checkpoints/thesis_uncertain_ppo_dls_single_s3/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
```

### PPO + PLS/PLPG shared-policy test

```powershell
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval.yaml --exp thesis_uncertain_ppo_pls_single_s1_shared_test --log_dir log_rl/thesis_uncertain_shared_test_pls_s1 --checkpoint_path checkpoints/thesis_uncertain_ppo_pls_single_s1/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval.yaml --exp thesis_uncertain_ppo_pls_single_s2_shared_test --log_dir log_rl/thesis_uncertain_shared_test_pls_s2 --checkpoint_path checkpoints/thesis_uncertain_ppo_pls_single_s2/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
python tools/run_shared_policy_two_agent.py --config config/tasks/Nav/thesis/algo/ppo_pls_shared_eval.yaml --exp thesis_uncertain_ppo_pls_single_s3_shared_test --log_dir log_rl/thesis_uncertain_shared_test_pls_s3 --checkpoint_path checkpoints/thesis_uncertain_ppo_pls_single_s3/best_model.zip --episode_data dataset/thesis/main_3cars_1ped/test_50_episodes_shared_oracle.pkl
```

## 7. Visualizing Saved Episode Worlds

### Visualize every 5th expert test episode

Run the expert test first, then use:

```powershell
$input = "log_rl/thesis_expert_test"
$output = "vis/thesis_expert_test_every5"
$temp = "vis/_tmp_thesis_expert_every5"

Remove-Item $temp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $output | Out-Null
New-Item -ItemType Directory -Force -Path $temp | Out-Null

Get-ChildItem $input -Filter *.pkl |
    Sort-Object {
        if ($_.BaseName -match '_(\d+)$') { [int]$matches[1] } else { 999999 }
    } |
    Where-Object {
        $_.BaseName -match '_(\d+)$' -and ([int]$matches[1] % 5) -eq 0
    } |
    ForEach-Object {
        Copy-Item $_.FullName $temp
    }

python tools/batch_visualize_episodes.py --input_dir $temp --output_dir $output --ego_id 3
```

## 8. Plotting Training And Test Results

Use two complementary plotting perspectives.

### Perspective 1: compare seeds within one method

This is used to inspect seed stability for a single method.

#### Perfect-sensor PPO seed stability

```powershell
python tools/plot_thesis_results.py `
  --training "PPO s1=checkpoints/thesis_perfect_ppo_single_s1/thesis_perfect_ppo_single_s1_metrics.csv" `
  --training "PPO s2=checkpoints/thesis_perfect_ppo_single_s2/thesis_perfect_ppo_single_s2_metrics.csv" `
  --training "PPO s3=checkpoints/thesis_perfect_ppo_single_s3/thesis_perfect_ppo_single_s3_metrics.csv" `
  --output-dir vis/thesis_results_perfect_ppo_3seeds
```

#### Perfect-sensor DLS seed stability

```powershell
python tools/plot_thesis_results.py `
  --training "DLS s1=checkpoints/thesis_perfect_ppo_dls_single_s1/thesis_perfect_ppo_dls_single_s1_metrics.csv" `
  --training "DLS s2=checkpoints/thesis_perfect_ppo_dls_single_s2/thesis_perfect_ppo_dls_single_s2_metrics.csv" `
  --training "DLS s3=checkpoints/thesis_perfect_ppo_dls_single_s3/thesis_perfect_ppo_dls_single_s3_metrics.csv" `
  --output-dir vis/thesis_results_perfect_dls_3seeds
```

#### Perfect-sensor PLS/PLPG seed stability

```powershell
python tools/plot_thesis_results.py `
  --training "PLS s1=checkpoints/thesis_perfect_ppo_pls_single_s1/thesis_perfect_ppo_pls_single_s1_metrics.csv" `
  --training "PLS s2=checkpoints/thesis_perfect_ppo_pls_single_s2/thesis_perfect_ppo_pls_single_s2_metrics.csv" `
  --training "PLS s3=checkpoints/thesis_perfect_ppo_pls_single_s3/thesis_perfect_ppo_pls_single_s3_metrics.csv" `
  --output-dir vis/thesis_results_perfect_pls_3seeds
```

#### PPO seed stability

```powershell
python tools/plot_thesis_results.py `
  --training "PPO s1=checkpoints/thesis_uncertain_ppo_single_s1/thesis_uncertain_ppo_single_s1_metrics.csv" `
  --training "PPO s2=checkpoints/thesis_uncertain_ppo_single_s2/thesis_uncertain_ppo_single_s2_metrics.csv" `
  --training "PPO s3=checkpoints/thesis_uncertain_ppo_single_s3/thesis_uncertain_ppo_single_s3_metrics.csv" `
  --output-dir vis/thesis_results_ppo_3seeds
```

#### DLS seed stability

```powershell
python tools/plot_thesis_results.py `
  --training "DLS s1=checkpoints/thesis_uncertain_ppo_dls_single_s1/thesis_uncertain_ppo_dls_single_s1_metrics.csv" `
  --training "DLS s2=checkpoints/thesis_uncertain_ppo_dls_single_s2/thesis_uncertain_ppo_dls_single_s2_metrics.csv" `
  --training "DLS s3=checkpoints/thesis_uncertain_ppo_dls_single_s3/thesis_uncertain_ppo_dls_single_s3_metrics.csv" `
  --output-dir vis/thesis_results_dls_3seeds
```

#### PLS/PLPG seed stability

```powershell
python tools/plot_thesis_results.py `
  --training "PLS s1=checkpoints/thesis_uncertain_ppo_pls_single_s1/thesis_uncertain_ppo_pls_single_s1_metrics.csv" `
  --training "PLS s2=checkpoints/thesis_uncertain_ppo_pls_single_s2/thesis_uncertain_ppo_pls_single_s2_metrics.csv" `
  --training "PLS s3=checkpoints/thesis_uncertain_ppo_pls_single_s3/thesis_uncertain_ppo_pls_single_s3_metrics.csv" `
  --output-dir vis/thesis_results_pls_3seeds
```

### Perspective 2: average 3 seeds per method and compare methods

This is used for the final thesis comparison between methods.

#### Perfect-sensor averaged training curves and averaged single-agent test summaries

```powershell
python tools/plot_thesis_results.py `
  --avg-training "PPO=checkpoints/thesis_perfect_ppo_single_s1/thesis_perfect_ppo_single_s1_metrics.csv,checkpoints/thesis_perfect_ppo_single_s2/thesis_perfect_ppo_single_s2_metrics.csv,checkpoints/thesis_perfect_ppo_single_s3/thesis_perfect_ppo_single_s3_metrics.csv" `
  --avg-training "PPO + DLS=checkpoints/thesis_perfect_ppo_dls_single_s1/thesis_perfect_ppo_dls_single_s1_metrics.csv,checkpoints/thesis_perfect_ppo_dls_single_s2/thesis_perfect_ppo_dls_single_s2_metrics.csv,checkpoints/thesis_perfect_ppo_dls_single_s3/thesis_perfect_ppo_dls_single_s3_metrics.csv" `
  --avg-training "PPO + PLS=checkpoints/thesis_perfect_ppo_pls_single_s1/thesis_perfect_ppo_pls_single_s1_metrics.csv,checkpoints/thesis_perfect_ppo_pls_single_s2/thesis_perfect_ppo_pls_single_s2_metrics.csv,checkpoints/thesis_perfect_ppo_pls_single_s3/thesis_perfect_ppo_pls_single_s3_metrics.csv" `
  --avg-single-test "PPO=results/thesis_perfect_ppo_single_s1_test/test_metrics.csv,results/thesis_perfect_ppo_single_s2_test/test_metrics.csv,results/thesis_perfect_ppo_single_s3_test/test_metrics.csv" `
  --avg-single-test "PPO + DLS=results/thesis_perfect_ppo_dls_single_s1_test/test_metrics.csv,results/thesis_perfect_ppo_dls_single_s2_test/test_metrics.csv,results/thesis_perfect_ppo_dls_single_s3_test/test_metrics.csv" `
  --avg-single-test "PPO + PLS=results/thesis_perfect_ppo_pls_single_s1_test/test_metrics.csv,results/thesis_perfect_ppo_pls_single_s2_test/test_metrics.csv,results/thesis_perfect_ppo_pls_single_s3_test/test_metrics.csv" `
  --output-dir vis/thesis_results_perfect_avg_3seeds
```

#### Perfect-sensor averaged shared-policy test summaries

```powershell
python tools/plot_thesis_results.py `
  --avg-shared-test "PPO=results/thesis_perfect_ppo_single_s1_shared_test/summary_metrics.csv,results/thesis_perfect_ppo_single_s2_shared_test/summary_metrics.csv,results/thesis_perfect_ppo_single_s3_shared_test/summary_metrics.csv" `
  --avg-shared-test "PPO + DLS=results/thesis_perfect_ppo_dls_single_s1_shared_test/summary_metrics.csv,results/thesis_perfect_ppo_dls_single_s2_shared_test/summary_metrics.csv,results/thesis_perfect_ppo_dls_single_s3_shared_test/summary_metrics.csv" `
  --avg-shared-test "PPO + PLS=results/thesis_perfect_ppo_pls_single_s1_shared_test/summary_metrics.csv,results/thesis_perfect_ppo_pls_single_s2_shared_test/summary_metrics.csv,results/thesis_perfect_ppo_pls_single_s3_shared_test/summary_metrics.csv" `
  --output-dir vis/thesis_results_perfect_shared_avg_3seeds
```

#### Averaged training curves and averaged single-agent test summaries

```powershell
python tools/plot_thesis_results.py `
  --avg-training "PPO=checkpoints/thesis_uncertain_ppo_single_s1/thesis_uncertain_ppo_single_s1_metrics.csv,checkpoints/thesis_uncertain_ppo_single_s2/thesis_uncertain_ppo_single_s2_metrics.csv,checkpoints/thesis_uncertain_ppo_single_s3/thesis_uncertain_ppo_single_s3_metrics.csv" `
  --avg-training "PPO + DLS=checkpoints/thesis_uncertain_ppo_dls_single_s1/thesis_uncertain_ppo_dls_single_s1_metrics.csv,checkpoints/thesis_uncertain_ppo_dls_single_s2/thesis_uncertain_ppo_dls_single_s2_metrics.csv,checkpoints/thesis_uncertain_ppo_dls_single_s3/thesis_uncertain_ppo_dls_single_s3_metrics.csv" `
  --avg-training "PPO + PLS=checkpoints/thesis_uncertain_ppo_pls_single_s1/thesis_uncertain_ppo_pls_single_s1_metrics.csv,checkpoints/thesis_uncertain_ppo_pls_single_s2/thesis_uncertain_ppo_pls_single_s2_metrics.csv,checkpoints/thesis_uncertain_ppo_pls_single_s3/thesis_uncertain_ppo_pls_single_s3_metrics.csv" `
  --avg-single-test "PPO=results/thesis_uncertain_ppo_single_s1_test/test_metrics.csv,results/thesis_uncertain_ppo_single_s2_test/test_metrics.csv,results/thesis_uncertain_ppo_single_s3_test/test_metrics.csv" `
  --avg-single-test "PPO + DLS=results/thesis_uncertain_ppo_dls_single_s1_test/test_metrics.csv,results/thesis_uncertain_ppo_dls_single_s2_test/test_metrics.csv,results/thesis_uncertain_ppo_dls_single_s3_test/test_metrics.csv" `
  --avg-single-test "PPO + PLS=results/thesis_uncertain_ppo_pls_single_s1_test/test_metrics.csv,results/thesis_uncertain_ppo_pls_single_s2_test/test_metrics.csv,results/thesis_uncertain_ppo_pls_single_s3_test/test_metrics.csv" `
  --output-dir vis/thesis_results_avg_3seeds
```

#### Averaged shared-policy test summaries

```powershell
python tools/plot_thesis_results.py `
  --avg-shared-test "PPO=results/thesis_uncertain_ppo_single_s1_shared_test/summary_metrics.csv,results/thesis_uncertain_ppo_single_s2_shared_test/summary_metrics.csv,results/thesis_uncertain_ppo_single_s3_shared_test/summary_metrics.csv" `
  --avg-shared-test "PPO + DLS=results/thesis_uncertain_ppo_dls_single_s1_shared_test/summary_metrics.csv,results/thesis_uncertain_ppo_dls_single_s2_shared_test/summary_metrics.csv,results/thesis_uncertain_ppo_dls_single_s3_shared_test/summary_metrics.csv" `
  --avg-shared-test "PPO + PLS=results/thesis_uncertain_ppo_pls_single_s1_shared_test/summary_metrics.csv,results/thesis_uncertain_ppo_pls_single_s2_shared_test/summary_metrics.csv,results/thesis_uncertain_ppo_pls_single_s3_shared_test/summary_metrics.csv" `
  --output-dir vis/thesis_results_shared_avg_3seeds
```

### Example: compare one seed per method

```powershell
python tools/plot_thesis_results.py `
  --training "PPO=checkpoints/thesis_uncertain_ppo_single_s1/thesis_uncertain_ppo_single_s1_metrics.csv" `
  --training "PPO + DLS=checkpoints/thesis_uncertain_ppo_dls_single_s1/thesis_uncertain_ppo_dls_single_s1_metrics.csv" `
  --training "PPO + PLS=checkpoints/thesis_uncertain_ppo_pls_single_s1/thesis_uncertain_ppo_pls_single_s1_metrics.csv" `
  --single-test "PPO=results/thesis_uncertain_ppo_single_s1_test/test_metrics.csv" `
  --single-test "PPO + DLS=results/thesis_uncertain_ppo_dls_single_s1_test/test_metrics.csv" `
  --single-test "PPO + PLS=results/thesis_uncertain_ppo_pls_single_s1_test/test_metrics.csv" `
  --output-dir vis/thesis_results_seed1
```

### Example: compare one shared-policy seed per method

```powershell
python tools/plot_thesis_results.py `
  --shared-test "PPO=results/thesis_uncertain_ppo_single_s1_shared_test/summary_metrics.csv" `
  --shared-test "PPO + DLS=results/thesis_uncertain_ppo_dls_single_s1_shared_test/summary_metrics.csv" `
  --shared-test "PPO + PLS=results/thesis_uncertain_ppo_pls_single_s1_shared_test/summary_metrics.csv" `
  --output-dir vis/thesis_results_shared_seed1
```

## 9. Notes

- `best_model.zip` is the file that should be used for final test and shared-policy evaluation.
- The shared-policy test should use the `*_shared_oracle.pkl` dataset, not the plain cached test split.
- The current runbook reflects the frozen uncertain-sensor thesis setup.
