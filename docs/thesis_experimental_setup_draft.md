# Experimental Setup Draft

## 1. Experimental Goal

The experiments evaluate whether a decentralized probabilistic logic shield improves reinforcement learning performance and safety in a structured multi-agent driving task.

The main comparison is:

- `PPO`
- `PPO + decentralized PLS`

The central research questions are:

1. does decentralized PLS improve task success over plain PPO?
2. does decentralized PLS reduce rule-based and physical safety violations?
3. does decentralized PLS better match the expert stopping policy under the same observation and action interface?

## 2. Environment

### 2.1 Map

All experiments are run on:

- `config/maps/square_2x2.yaml`

This is a compact grid-based urban map containing roads, sidewalks, and labeled intersection regions.

### 2.2 Agents

All thesis runs use:

- `config/agents/thesis/main_3cars_1ped.yaml`

Each episode contains:

- `Car_1`: RL-controlled ego vehicle
- `Car_2`: rule-based background car
- `Car_3`: rule-based background car
- `Pedestrian_1`: rule-based pedestrian

The RL agent’s operating region is:

- `agent_region: 100`

### 2.3 Rules and Ontology

All experiments use:

- ontology: `config/rules/ontology_simple_weak.yaml`
- expert rule file: `config/rules/Nav/easy/expert_minimal_4action.yaml`

The environment local planner uses:

- `rule_type: Z3_Expert`

This means background-agent behavior and expert rollouts are generated from the same rule system used as the normative reference during evaluation.

## 3. Observation and Action Configuration

### 3.1 Action Space

The ego policy uses the 4-action macro-action space:

- `0 = Fast`
- `1 = Normal`
- `2 = Slow`
- `3 = Stop`

### 3.2 Observation

The RL agent uses:

- `grounding_mode: thesis_minimal`
- `obs_fov: 25`
- `fov_entities: Entity = 4`

The resulting observation is the 6-bit decentralized symbolic vector:

- `ego_in_inter`
- `other_in_inter`
- `ego_at_inter`
- `close_ahead`
- `ahead`
- `higher_pri`

### 3.3 Horizon and Rewards

The current healthy configuration uses:

- `max_horizon: 150`
- `step_cost: 0`
- `progress_reward: 0.1`
- `goal_reward: 10`
- `overtime_cost: -6`

The intention is to reward forward progress and goal completion while penalizing excessive stalling.

## 4. Dataset Generation

### 4.1 Generators

Two generators are used:

- `tools/create_single_agent_two_agent_car_ped_region_episode.py`
- `tools/create_single_agent_two_agent_car_ped_close_ahead_episode.py`

The first creates generic rich interaction episodes. The second creates structured same-lane following cases to stress short-range stopping behavior.

### 4.2 Real Split Composition

The real thesis dataset is composed of:

- train: `100` episodes
- validation: `20` episodes
- test: `50` episodes

The intended composition is:

- train: `80 rich + 20 close_ahead`
- validation: `16 rich + 4 close_ahead`
- test: `40 rich + 10 close_ahead`

The split is produced by generating the component datasets separately and then merging them into final cached episode files:

- `dataset/thesis/main_3cars_1ped/train_100_episodes.pkl`
- `dataset/thesis/main_3cars_1ped/val_20_episodes.pkl`
- `dataset/thesis/main_3cars_1ped/test_50_episodes.pkl`

### 4.3 Expert-Guided Generation Configs

The real split uses the `fov25` expert generation configs:

- `config/tasks/Nav/thesis/experts/expert_episode_train_fov25.yaml`
- `config/tasks/Nav/thesis/experts/expert_episode_val_fov25.yaml`
- `config/tasks/Nav/thesis/experts/expert_episode_test_fov25.yaml`

This keeps the generation pipeline consistent with the final decentralized observation design.

## 5. Methods Compared

### 5.1 PPO Baseline

The PPO baseline uses:

- `config/tasks/Nav/thesis/algo/ppo_single_train.yaml`
- `config/tasks/Nav/thesis/algo/ppo_single_eval.yaml`

This baseline receives the same environment, reward structure, action space, observation design, and cached episodes as the shielded method, but without symbolic policy shaping.

### 5.2 PPO + Decentralized PLS

The decentralized shielded method uses:

- `config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml`
- `config/tasks/Nav/thesis/algo/ppo_pls_single_eval.yaml`

The shield is enabled during both:

- training
- test-time evaluation

The final healthy PLS configuration uses:

- ProbLog backend
- graded safety enabled
- action order aligned with env action ids
- shield temperature `1.0`
- safety loss enabled

### 5.3 Optional Additional Baselines

The repository also contains DLS and joint-PLS related configs. However, the primary finalized comparison in this draft setup is:

- PPO vs decentralized PLS

This keeps the evaluation focused on the thesis core claim before introducing more variants.

## 6. Training Setup

### 6.1 PPO Hyperparameters

Both PPO and PPO+PLS currently use the same PPO backbone:

- algorithm: `PPO` or `PLSPPO`
- policy: `MlpPolicy`
- feature extractor: `MLPFeatureExtractor`
- feature dimension: `32`
- learning rate: `3e-4`
- clip range: `0.2`
- value loss coefficient: `0.5`
- batch size: `64`
- rollout steps: `256`
- PPO epochs per update: `10`
- entropy coefficient: `0.001`
- number of environments: `1`

### 6.2 Training Budget

The real runs use:

- `total_timesteps: 40000`

### 6.3 Evaluation Frequency

Checkpoint evaluation is performed every:

- `1000` steps

and checkpoints are also saved every:

- `1000` steps

This yields learning-curve metrics throughout training and allows final best-model selection.

### 6.4 Seeds

The real experiments use three seeds:

- seed `111`
- seed `222`
- seed `333`

These correspond to:

- `s1`
- `s2`
- `s3`

for both PPO and PPO+PLS.

## 7. PLS Configuration

### 7.1 Regimes

The current decentralized PLS uses:

- `must_stop`
- `warning`
- `fast_zone`
- `normal_zone`

### 7.2 Rule Set

The current implemented PLS rules are:

#### Must-stop

- `close_ahead`
- `ego_at_inter && other_in_inter`
- `ego_at_inter && higher_pri`
- `ego_in_inter && other_in_inter && higher_pri`

#### Warning

- `ahead`
- `ego_at_inter`
- `ego_in_inter && other_in_inter`

### 7.3 Final Graded Safety Weights

The final healthy decentralized PLS weights are:

#### Must-stop

- `fast: 0.0`
- `normal: 0.01`
- `slow: 0.05`
- `stop: 1.0`

#### Warning

- `fast: 0.10`
- `normal: 0.65`
- `slow: 1.00`
- `stop: 0.45`

#### Fast-zone

- `fast: 1.00`
- `normal: 0.80`
- `slow: 0.55`
- `stop: 0.25`

#### Normal-zone

- `fast: 0.35`
- `normal: 1.00`
- `slow: 0.75`
- `stop: 0.40`

### 7.4 Final Implementation Notes

The final decentralized PLS implementation includes the following critical fixes:

- corrected action-order mapping between environment and shield
- correct handling of pedestrians in `other_in_inter`
- correct handling of pedestrians in `higher_pri`
- corrected local FOV entity extraction so visible pedestrians are not dropped from local grounding

These fixes are essential to the validity of the final PPO vs PLS comparison.

## 8. Failure and Safety Evaluation

### 8.1 Episode Termination Types

Episodes can end by:

- success
- fail
- overtime / timeout
- truncation

### 8.2 Hard-Fail Semantics

Hard failures are tracked in three categories:

- `rule_based_fail_events`
- `deadzone_fail_events`
- `simultaneous_entry_fail_events`

The total is reported as:

- `hard_fail_events`

### 8.3 Interpretation

These metrics distinguish between:

- symbolic failure to obey the stop-required rules
- physically unsafe short-range collisions or deadzone incursions
- coordination failure at intersections

This breakdown is important because not all failures are of the same type. A model may obey symbolic stop rules but still fail in close-following dynamics, or conversely may violate stop rules without causing a physical collision.

## 9. Main Evaluation Metrics

### 9.1 TSR

`TSR` is the task success rate:

- the fraction of evaluation episodes that terminate successfully

This is the main task-level performance metric.

### 9.2 DSR

`DSR` is the stop-decision success rate:

- the proportion of expert-required stop decisions that are matched by the learned policy

This is a key diagnostic metric for rule alignment.

### 9.3 Reward

The evaluation pipeline also reports:

- `mean_reward`

This summarizes task progress, goal completion, and penalties over the evaluation set.

### 9.4 Violation Metrics

The evaluation CSV and training CSV record:

- `fail`
- `hard_fail_events`
- `rule_based_fail_events`
- `deadzone_fail_events`
- `simultaneous_entry_fail_events`
- `timeout`
- `truncated`

### 9.5 Action Histograms

The metrics pipeline additionally logs:

- `policy_action_hist_*`
- `expert_action_hist_*`
- `policy_stop_count`
- `expert_stop_count`
- `matched_stop_count`

These make it possible to diagnose over-stopping, under-stopping, and policy-expert divergence.

### 9.6 Training-Side Safety Metrics

The callback also tracks cumulative and interval-based training metrics:

- `train_fail`
- `train_timeout`
- `train_success`
- `train_truncated`
- `train_hard_fail_events`
- `train_rule_based_fail_events`
- `train_deadzone_fail_events`
- `train_simultaneous_entry_fail_events`

and corresponding:

- `*_since_last_eval`

This allows both cumulative and local-window learning-curve analysis.

## 10. Model Selection and Testing

### 10.1 Checkpoint Selection

The evaluation callback writes:

- per-training-run metrics CSVs
- periodic checkpoints

Final testing should use:

- `best_model.zip`

### 10.2 Single-Agent Test Evaluation

After training, each seed is evaluated on:

- `dataset/thesis/main_3cars_1ped/test_50_episodes.pkl`

The test runner also supports:

- saving rollout world pickles
- rendering all test episodes

### 10.3 Visualization

Visual inspection is part of the evaluation methodology, especially for:

- deadzone failures
- incorrect close-ahead follow-up behavior
- stop-rule mismatches
- overly conservative stopping

Visualizations therefore serve as a qualitative complement to TSR and violation summaries.

## 11. Plotting and Reporting

The main comparison plots are produced by:

- `tools/plot_thesis_results.py`

The intended report set includes:

- TSR learning curves
- mean reward learning curves
- training failure curves
- hard-fail breakdown curves
- final single-agent evaluation summaries
- optional multi-seed averages

For the PPO vs decentralized PLS comparison, the most relevant outputs are:

- averaged training curves over 3 seeds
- averaged single-agent test summaries over 3 seeds

## 12. Reproducibility Notes

The final setup is reproducible because:

- dataset generation is scripted
- training and evaluation use frozen YAML configs
- random seeds are fixed
- cached episode files are reused across methods

Care must be taken to ensure:

- dataset pickles are regenerated after generator changes
- runs are restarted after changing cached datasets
- train and eval logs are interpreted only after the final shield and observation fixes

## 13. Summary

The experimental setup is designed to provide a controlled, end-to-end comparison between plain PPO and decentralized PLS under:

- identical map and scenario families
- identical action space and reward structure
- identical real train/val/test splits
- identical PPO backbone and optimization settings

The only intended methodological difference between the two methods is the presence of the probabilistic logic shield. This makes the resulting comparison suitable for isolating the effect of symbolic decentralized safety shaping on learning and evaluation performance.

