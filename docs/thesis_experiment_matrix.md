# Thesis Experiment Matrix

## Methodology Draft

This thesis evaluates whether logic-based shielding improves the safety and learning behaviour of reinforcement learning for urban navigation, and whether probabilistic shielding provides an advantage over deterministic shielding under uncertain sensing. All experiments are conducted in LogiCity using a fixed benchmark setup so that differences between methods can be attributed to the shielding mechanism rather than to changes in map layout, ontology, or agent composition.

### Environment

All experiments use the same city map, agent lineup, and rule set. The map is defined in `config/maps/square_2x2.yaml`. The agent configuration is defined in `config/agents/thesis/main_3cars_1ped.yaml`. The logical state abstraction uses `config/rules/ontology_simple.yaml`, and expert behaviour is defined by `config/rules/Nav/easy/expert_simple.yaml`.

The task contains four agents in total. `Car_1` is the main ego vehicle in single-agent training and testing. The remaining agents are `Car_2`, `Car_3`, and `Pedestrian_1`. In single-agent experiments, only `Car_1` is controlled by the learned policy and the other agents remain rule-based. In shared-policy multi-agent evaluation, both `Car_1` and `Car_2` are controlled by the same learned checkpoint, while `Car_3` and `Pedestrian_1` remain rule-based.

The observation setting uses `fov_entities.Entity = 4`, so the observation dimensionality is kept fixed across single-agent and shared-policy evaluation.

### Multi-Agent Requirement From the Thesis Assignment

The thesis assignment explicitly targets **dynamic, logic-driven environments with multiple agents** and asks for an extension of probabilistic logic shields to **coordinate multiple agents**. In the current thesis design, this requirement is addressed in two complementary ways:

- the environment itself is always multi-agent:
  - multiple vehicles are present simultaneously
  - one pedestrian is also present
  - logical predicates and safety constraints depend on interactions between agents
- the final evaluation explicitly includes a shared-policy multi-agent setting:
  - the same learned policy is used for both `Car_1` and `Car_2`
  - this is therefore a required final evaluation mode, not merely an optional add-on

This means the thesis does **not** implement full multi-agent RL training with multiple simultaneously learning agents. Instead, it uses single-agent training inside a genuinely multi-agent urban environment and then tests whether the resulting learned policy and shielding mechanism generalize to a setting where **two RL-controlled cars must act together in the same environment**. For the current thesis scope, this is the intended interpretation of the assignment’s multi-agent requirement.

### Action Space

The thesis uses a discrete 4-action control space:

- `0 = slow`
- `1 = normal`
- `2 = fast`
- `3 = stop`

This action space replaces the earlier `go/stop` abstraction. The binary action space made deterministic shielding too similar to the expert policy, because the shield effectively reduced to an expert-like stop/go switch. The 4-action setup gives the policy and the shields more room to differ. In particular, it allows movement actions to be treated with different levels of caution and makes probabilistic shielding more meaningful.

The expert policy remains intentionally simple. It uses only `normal` and `stop`, and does not actively choose `slow` or `fast`. This keeps the expert interpretable while still allowing PPO, PPO+DLS, and PPO+PLS to exploit the richer 4-action space.

The current action-cost design is:

- `slow = -3`
- `normal = 0`
- `fast = -4`
- `stop = -5`

This reward structure is intended to make `normal` the default movement mode, `slow` a cautious fallback, `fast` an aggressive option, and `stop` the explicit halt action.

### Methods

Three methods are compared:

- `PPO`: unshielded reinforcement learning baseline
- `PPO + DLS`: PPO combined with a deterministic logic shield
- `PPO + PLS`: PPO combined with a probabilistic logic shield

The corresponding implementation files are:

- PPO baseline through the standard PPO path
- `logicity/rl_agent/alg/dls_ppo.py`
- `logicity/rl_agent/alg/pls_ppo.py`
- `logicity/shields/dls.py`
- `logicity/shields/pls.py`

Both shielded methods operate at the policy level. Let `pi(a|s)` denote the base PPO policy and `P(safe|s,a)` the safety value assigned by the shield. The executed shielded policy is:

`pi_plus(a|s) propto P(safe|s,a) * pi(a|s)`

This means the shield does not only override a sampled action afterward. Instead, it reshapes the policy distribution itself before action selection.

### Deterministic Logic Shield

The deterministic logic shield is implemented in `logicity/shields/dls.py`. Under the current setup, DLS uses uncertain observations when the uncertain-sensor scenario is active, but converts those uncertain predicate values into hard booleans through thresholding. It then applies deterministic stop logic. If the stop condition holds for a candidate movement action, that action is masked out. The `stop` action remains the deterministic safe fallback.

Thus, DLS is deterministic in its reasoning, even when the incoming observation values are uncertain. The uncertainty is discarded at the thresholding stage.

In the current implementation, DLS no longer uses a single undifferentiated safe-movement fallback. Instead, it applies hard risk bands:

- low risk: allow `normal`, `slow`, and sometimes `fast`
- medium risk: allow `slow` and `stop`
- high risk: allow only `stop`

This change was introduced to reduce the earlier tendency of shielded methods to collapse into `slow + stop` behaviour.

The resulting deterministic shielded distribution is:

`pi_plus(a|s) propto P(safe|s,a) * pi(a|s)`

with:

`P(safe|s,a) in {0,1}`

So DLS performs hard masking. Unsafe actions are assigned probability `0`, safe actions retain their PPO probability mass, and the remaining distribution is renormalized.

### Probabilistic Logic Shield

The probabilistic logic shield is implemented in `logicity/shields/pls.py`. PLS uses the same observation information as DLS, but it does not threshold uncertain predicate values into booleans. Instead, it keeps them as probabilities and computes a soft conflict probability. This conflict estimate is then used to assign soft safety scores to `slow`, `normal`, `fast`, and `stop`, after which the PPO policy is renormalized into the shielded policy.

The current implementation now includes a PLPG-style training objective in `logicity/rl_agent/alg/pls_ppo.py`. Concretely, PPO training uses:

- a shielded policy gradient through the shielded policy `pi_plus`
- an additional safety loss term based on `-log P_{pi_plus}(safe | s)`
- a safety coefficient `alpha` that weights this extra safety term

So the current training procedure is substantially closer to the PLPG formulation in the paper than the earlier simplified PLS implementation.

In the current implementation, PLS is also shaped so that action preference depends on the risk regime:

- very low risk: `fast` may be preferred
- low to moderate risk: `normal` should be preferred
- elevated risk: `slow` should be preferred
- high risk: `stop` should dominate

This was introduced because earlier shielded runs tended to overuse `slow` and underuse `normal`.

The resulting probabilistic shielded distribution is again:

`pi_plus(a|s) propto P(safe|s,a) * pi(a|s)`

but now:

`P(safe|s,a) in [0,1]`

So PLS performs soft multiplicative reweighting rather than hard masking. Risky actions are downweighted, safer actions are upweighted, and the final action distribution is renormalized.

### Exact Remaining Differences Between the Current PLS and the Papers

Even after adding the PLPG-style safety term, the current implementation still differs from the papers in several specific ways.

#### 1. No ProbLog-based probabilistic logic program

In the papers, the safety probability is derived from a probabilistic logic representation of the domain. In the current implementation, there is no ProbLog program and no external probabilistic logic engine. Instead, the safety model is implemented directly in Python inside `logicity/shields/pls.py`.

#### 2. No differentiable logic circuit compilation

The papers describe probabilistic shielding in a form that supports differentiable reasoning over the probabilistic logic structure. The current implementation does not compile the logic into a differentiable probabilistic circuit. Instead, it computes conflict probabilities through hand-written numerical formulas over the uncertain predicate values.

#### 3. Hand-crafted `P(safe | s, a)` model

In the papers, `P(safe | s, a)` follows from the probabilistic logic semantics. In the current implementation, `P(safe | s, a)` is manually engineered from the task predicates:

- `IsAtInter`
- `IsInInter`
- `HigherPri`
- `CollidingClose`

These are combined into a custom conflict probability and then mapped into custom action-specific safety weights for `slow`, `normal`, `fast`, and `stop`.

#### 4. Task-specific action-safety shaping

The current implementation uses explicit heuristic shaping so that:

- very low risk can favour `fast`
- low to moderate risk should favour `normal`
- elevated risk should favour `slow`
- high risk should favour `stop`

This is useful for the LogiCity thesis task, but it is more hand-crafted and task-specific than the generic formulation in the paper.

#### 5. Current alpha handling

The current PLPG-style implementation includes a safety coefficient `alpha` through the shield configuration and uses it during PPO training. However, the exact optimization pipeline remains an implementation-specific adaptation of SB3 PPO rather than a line-by-line reproduction of the original paper codebase.

### Honest Thesis Description

The most accurate description of the current method is:

- it implements the key PLPG training structure
  - shielded policy gradient
  - plus a safety gradient term weighted by `alpha`
- but it still uses a hand-designed probabilistic safety model instead of a full ProbLog-based differentiable probabilistic logic system

So it should be described as a PLPG-style implementation that is structurally aligned with the papers, but not a full low-level reproduction of their probabilistic logic machinery.

### Sensor Scenarios

The thesis distinguishes two scenarios.

#### Scenario 1: Perfect sensors

In the perfect-sensor scenario, PPO, DLS, and PLS all receive the true logical grounding vector. No observation uncertainty is injected.

#### Scenario 2: Uncertain sensors

In the uncertain-sensor scenario, PPO, DLS, and PLS all receive noisy predicate observations. This is implemented so that all methods face the same perception uncertainty. PPO receives the noisy observation vector directly. DLS receives the same noisy observation vector and thresholds it into booleans. PLS receives the same noisy observation vector and keeps the predicate values probabilistic.

This design is more realistic than giving uncertainty only to the shield, because in a real driving system the learned policy also acts under uncertain sensing.

### Observation Uncertainty Model

Observation uncertainty is implemented through the sensor model in `logicity/shields/sensor_model.py` and is injected into the observation pipeline in `logicity/utils/gym_wrapper.py`.

The uncertain observation model is based on a confusion-matrix style abstraction. For a true binary predicate `z`, the sensor reports a positive reading with probability `tpr` if `z = 1` and with probability `fpr` if `z = 0`. The observed reading is then converted into a soft confidence value.

Only a subset of predicates is made uncertain:

- `IsAtInter`
- `IsInInter`
- `CollidingClose`

The following predicates remain deterministic:

- `HigherPri`
- `IsCar`
- `IsPedestrian`

The uncertainty is action-conditioned. This means sensing quality depends on the candidate action being evaluated. In the current model, `slow` and `stop` have better sensing quality than `fast`. This is intended to approximate the idea that cautious driving yields more reliable perception than aggressive motion.

The current default profiles are milder than earlier exploratory versions. For `IsAtInter` and `IsInInter`, the default profiles are:

- `slow`: `tpr = 0.97`, `fpr = 0.03`
- `normal`: `tpr = 0.94`, `fpr = 0.06`
- `fast`: `tpr = 0.90`, `fpr = 0.10`
- `stop`: `tpr = 0.99`, `fpr = 0.01`

For `CollidingClose`, the default profiles are:

- `slow`: `tpr = 0.92`, `fpr = 0.08`
- `normal`: `tpr = 0.88`, `fpr = 0.12`
- `fast`: `tpr = 0.82`, `fpr = 0.18`
- `stop`: `tpr = 0.97`, `fpr = 0.03`

This uncertainty model is intentionally simple and controllable. It is not meant to be a full geometric perception simulator, but rather a compact and reproducible way to introduce realistic sensing errors and to create a meaningful distinction between deterministic and probabilistic shielding.

The current uncertainty level should be viewed as moderate rather than extreme. `slow` and `stop` remain fairly reliable, `normal` is still reasonably informative, and `fast` is the most uncertain, especially for `CollidingClose`. So the uncertain-sensor scenario introduces meaningful sensing degradation without making the task unrealistically noisy.

Under the current uncertain-sensor setup, the shielded methods do not re-query perfect simulator truth for risk estimation. Instead:

- PPO uses the noisy observation vector directly
- DLS computes hard risk decisions from the noisy observation values
- PLS computes soft risk values from the same noisy observation values

So the current uncertain-sensor scenario is a shared-observation uncertainty setting, not a perfect-policy-observation plus noisy-shield-only setting.

### Training and Evaluation Protocol

The main real training configs are:

- `config/tasks/Nav/thesis/algo/ppo_single_train.yaml`
- `config/tasks/Nav/thesis/algo/ppo_dls_single_train.yaml`
- `config/tasks/Nav/thesis/algo/ppo_pls_single_train.yaml`

The corresponding single-agent test configs are:

- `config/tasks/Nav/thesis/algo/ppo_single_test.yaml`
- `config/tasks/Nav/thesis/algo/ppo_dls_single_test.yaml`
- `config/tasks/Nav/thesis/algo/ppo_pls_single_test.yaml`

The shared-policy multi-agent evaluation configs are:

- `config/tasks/Nav/thesis/algo/ppo_shared_eval.yaml`
- `config/tasks/Nav/thesis/algo/ppo_dls_shared_eval.yaml`
- `config/tasks/Nav/thesis/algo/ppo_pls_shared_eval.yaml`

The current final training protocol is standardized to:

- `40000` total training timesteps
- evaluation every `1000` timesteps
- checkpoint saving every `1000` timesteps
- `max_horizon = 200`

This same overall setup is used across methods so that the comparison remains controlled.

The final common budget is set to `40000` timesteps. This value was chosen after exploratory runs suggested that the main learning gains for the simplified thesis setup occur well before `100000` timesteps, while a larger budget would mainly increase runtime and make the repeated multi-seed experiments unnecessarily expensive.

### Repeated Runs and Averaging

The final reported thesis results should not rely on a single random seed per method. Reinforcement learning is stochastic, and conclusions based on one run can be misleading. Therefore, after choosing the final training budget from the exploratory runs, each method should be trained with `3` different random seeds and the reported results should be averaged across those runs.

This repetition policy applies to:

- `PPO`
- `PPO + DLS`
- `PPO + PLS`

and should be done separately for:

- the perfect-sensor scenario
- the uncertain-sensor scenario

The exploratory single-seed runs are used only to estimate a sensible timestep budget and to debug the setup. The final thesis tables and figures should be based on the `3`-seed averages.

If the shield logic or reward design changes materially, exploratory shielded runs from before that change should not be treated as final evidence. In that case the affected methods should be restarted under the updated setup before collecting the final multi-seed results.

### Dataset Splits

The final thesis dataset is stored under `dataset/thesis/main_3cars_1ped/` and consists of:

- training split: `train_100_episodes.pkl`
- validation split: `val_20_episodes.pkl`
- test split: `test_50_episodes.pkl`

Visualizable world traces are stored under `log_rl/thesis_dataset_worlds/`, including:

- `val_20_worlds/`
- `test_50_worlds/`

The same cached test split is used for both single-agent and shared-policy evaluation. This means both settings are evaluated on the same initial benchmark scenarios, although the resulting trajectories can differ because the control regime differs.

Some validation and test episodes contain no expert-required `Stop` action. These episodes are kept in the benchmark and are not filtered out. In such episodes:

- per-episode `dsr` is left blank
- `expert_stop_count = 0`

This is intentional. These episodes still matter for overall success, reward, timeout behaviour, and over-cautious stopping. Aggregate `dsr` is computed only over stop-relevant decisions across the full split.

### Multi-Agent Oracle Horizon

For shared-policy evaluation, the episode horizon should not be derived from the single-agent oracle alone. In the multi-agent setting, the correct comparison is against an expert rollout in which both learned-policy vehicles are controlled by the expert. The evaluator therefore now supports `joint_oracle_step`, and annotated datasets can be produced using `tools/annotate_shared_oracle_steps.py`. This ensures that multi-agent timeout limits are based on the correct oracle reference.

### Metrics

The thesis reports the following main metrics:

- `TSR` (trajectory success rate)
- `DSR` (decision success rate for expert-required `Stop` decisions)
- `fail`
- `timeout`
- `truncated`
- `mean_reward`

Additional behaviour and agreement metrics are:

- `policy_stop_count`
- `expert_stop_count`
- `matched_stop_count`

Shield-specific diagnostics are:

- `shield_intervention_count`
- `shield_intervention_rate`

Computational metrics are:

- elapsed wall-clock training time
- seconds per `1000` training timesteps

Training-safety metrics are also logged during learning. These include cumulative and interval counts such as:

- `train_fail`
- `train_timeout`
- `train_success`
- `train_truncated`
- `train_fail_since_last_eval`

These training metrics are important because one of the thesis claims is that shielding can make the learning process itself safer, not only the final evaluated policy.

### Interpretation of Metrics

`TSR` measures whether the agent reaches the goal successfully. `DSR` measures whether the policy agrees with the expert on safety-critical stop decisions. `fail` captures safety violations. `timeout` and `truncated` capture inefficiency and incomplete behaviour. `mean_reward` reflects the aggregate reinforcement learning objective but is not sufficient on its own to evaluate safety.

Sample efficiency and computational overhead are treated separately. Sample efficiency is measured by the number of training timesteps needed to reach a target `TSR` or `DSR`. Computational overhead is measured through wall-clock time and normalized runtime such as seconds per `1000` timesteps.

### Expert Reference and Upper Bound

The expert policy serves as the logic-based upper-bound reference for the current ontology and rule set. It is not expected to represent globally optimal driving in every sense, but it does provide the intended safety logic for the simplified task. PPO, PPO+DLS, and PPO+PLS are therefore compared against the expert primarily through stop-decision agreement and stop-count diagnostics.

### Shield Dependence Ablation

An additional ablation is included for the shielded methods. After training PPO+DLS and PPO+PLS, the learned checkpoint can be evaluated again with the shield disabled. This tests whether safer behaviour was internalized by the learned policy or whether safety at test time depends mainly on the presence of the active shield.

If performance and safety collapse when the shield is removed, the learned policy is shield-dependent. If safety remains improved even without the shield, then training under shielding has shaped a safer base policy.

### Thesis Positioning

The intended contribution of the thesis is not to claim that probabilistic shielding must always outperform deterministic shielding in all settings. In the perfect-sensor scenario, deterministic shielding may already be very strong. The core hypothesis is instead that under uncertain sensing, probabilistic shielding should provide a more robust and better calibrated alternative to deterministic threshold-based shielding.

The experimental comparison is therefore:

- PPO as the unshielded baseline
- PPO+DLS as the deterministic shield baseline
- PPO+PLS as the probabilistic shield method

evaluated in both perfect-sensor and uncertain-sensor scenarios, and in both single-agent and shared-policy settings.

## Planned Result Presentation

The results should be presented in a way that separates learning behaviour, final evaluation performance, and computational cost. The clearest organization is to group results first by sensor scenario and then by evaluation setting.

### Recommended Structure

Results should be organized in the following order:

1. Perfect-sensor scenario
2. Uncertain-sensor scenario

Within each scenario, the same sequence should be used:

1. training behaviour
2. final single-agent test results
3. final shared-policy multi-agent results

This structure makes it easier to compare DLS and PLS under clean sensing and under uncertainty without mixing the two stories together.

### Main Training Graphs

The core training plots should be learning curves over training timesteps.

#### 1. TSR learning curve

- x-axis: training timesteps
- y-axis: `TSR`
- one plot for the perfect-sensor scenario
- one plot for the uncertain-sensor scenario
- lines:
  - `PPO`
  - `PPO + DLS`
  - `PPO + PLS`

This should be one of the main figures because it shows task-learning progress directly.

#### 2. Training safety curve

- x-axis: training timesteps
- y-axis:
  - `train_fail_since_last_eval`
  - or cumulative `train_fail`
- one plot for the perfect-sensor scenario
- one plot for the uncertain-sensor scenario
- lines:
  - `PPO`
  - `PPO + DLS`
  - `PPO + PLS`

This figure is especially important because it directly supports or rejects the thesis claim that shielding makes training safer.

#### 3. DSR learning curve

- x-axis: training timesteps
- y-axis: `DSR`
- one plot per sensor scenario
- lines:
  - `PPO`
  - `PPO + DLS`
  - `PPO + PLS`

This should be treated as a secondary training figure. It is still important because the thesis focuses on expert-consistent stop decisions.

#### 4. Mean reward learning curve

- x-axis: training timesteps
- y-axis: `mean_reward`
- one plot per sensor scenario
- lines:
  - `PPO`
  - `PPO + DLS`
  - `PPO + PLS`

This is useful because reward convergence may differ from safety convergence. It should support the interpretation of whether shielded methods are learning useful behaviour or becoming too conservative.

### Final Test Figures

For final single-agent and shared-policy evaluation, grouped bar charts are the clearest presentation.

#### 5. Final single-agent test chart

Use grouped bars for:

- `TSR`
- `DSR`
- `fail`
- `timeout`
- optionally `mean_reward`

Create separate panels or separate charts for:

- perfect-sensor scenario
- uncertain-sensor scenario

Methods:

- `PPO`
- `PPO + DLS`
- `PPO + PLS`

#### 6. Final shared-policy test chart

Use the same structure as the single-agent chart, but for shared-policy multi-agent evaluation:

- `TSR`
- `DSR`
- `fail`
- `timeout`
- optionally `mean_reward`

Again, keep perfect and uncertain sensor scenarios visually separated.

### Overhead Figures

Computational cost should be shown separately from learning quality.

#### 7. Runtime overhead chart

Use a bar chart for:

- total wall-clock training time
- `seconds_per_1k_timesteps`

Compare:

- `PPO`
- `PPO + DLS`
- `PPO + PLS`

This should be shown separately from sample efficiency because runtime cost and learning speed answer different questions.

### Diagnostic Figures

These are not the main headline figures, but they are useful supplementary plots.

#### 8. Stop behaviour vs expert

Use grouped bars for:

- `policy_stop_count`
- `expert_stop_count`
- `matched_stop_count`

This is useful for understanding whether a method is:

- under-stopping
- over-stopping
- or matching expert stop behaviour closely

These charts are especially valuable for final test results.

#### 9. Shield activity

Use a line plot or bar chart for:

- `shield_intervention_count`
- `shield_intervention_rate`

This applies only to:

- `PPO + DLS`
- `PPO + PLS`

These are diagnostic rather than primary metrics. They help interpret whether the shield is active often, but they should not be the main result figure.

### Tables

Plots should be complemented by compact summary tables.

#### Table 1. Final single-agent test summary

Recommended columns:

- Method
- Sensor scenario
- `TSR`
- `DSR`
- `fail`
- `timeout`
- `mean_reward`

#### Table 2. Final shared-policy test summary

Recommended columns:

- Method
- Sensor scenario
- `TSR`
- `DSR`
- `fail`
- `timeout`
- `mean_reward`

#### Table 3. Training safety and overhead summary

Recommended columns:

- Method
- Sensor scenario
- Seed aggregation
- cumulative `train_fail`
- total wall-clock training time
- `seconds_per_1k_timesteps`

For the final reported tables, the displayed value should be the mean across `3` seeds. If space permits, the table should also report variability, for example standard deviation in parentheses.

### Minimal Strong Figure Set

If the thesis needs a compact but strong results section, the minimum recommended set is:

- TSR learning curve
- training safety curve using `train_fail_since_last_eval` or cumulative `train_fail`
- final test bar chart with `TSR`, `DSR`, `fail`, and `timeout`
- runtime overhead bar chart
- one compact summary table

This set is sufficient to show:

- whether the method learns
- whether training is safer
- whether final behaviour is successful and safe
- and what the computational cost is

### Interpretation Guidance

The most important narrative is not whether one method wins every metric. The intended interpretation is:

- `TSR` shows task success
- `DSR` shows expert-consistent stop behaviour
- `fail` shows safety violations
- training fail metrics show whether shielding makes learning safer
- runtime metrics show the engineering cost of shielding

This means the final results section should avoid treating reward alone as the main performance indicator. Safety, success, and computational overhead must be interpreted together.

For final reporting, every main conclusion should be based on the averaged multi-seed results rather than on a single run. Single-seed plots remain useful for exploratory diagnosis, but the thesis claims should be grounded in the `3`-seed averages.

In particular, when analysing the 4-action setup, the results should also be checked qualitatively for action-collapse patterns such as:

- `slow + stop` collapse for shielded methods
- `fast + stop` collapse for unshielded PPO

These behavioural patterns are useful diagnostics when interpreting TSR, DSR, and reward curves.
