# PLPG Thesis Experimental Plan for LogiCity

## Goal

This document defines the experimental plan for evaluating **Probabilistic Logic Policy Gradient (PLPG)** on LogiCity **Safe Path Following (SPF)** tasks.

The thesis goal is not only to show that PLPG can work, but to investigate:

- whether PLPG improves **task success rate**
- whether PLPG improves **sample efficiency**
- whether PLPG improves **safety**
- whether PLPG remains effective under **harder tasks**, **sensor noise**, and **multi-agent settings**
- which parts of the PLPG shield actually matter

The main baseline is **unshielded PPO**.

## Main Research Questions

### RQ1: Single-Agent Effectiveness

Does PLPG improve Safe Path Following performance relative to PPO in terms of:

- `TSR`
- `failure_rate`
- `timeout_rate`
- `mean_reward`

### RQ2: Training Efficiency

Does PLPG reach strong performance in fewer training timesteps than PPO?

This should be measured through:

- timesteps to reach a target `TSR`
- area under the learning curve
- failure reduction over training

### RQ3: Difficulty Sensitivity

How does the benefit of PLPG change across:

- `easy`
- `medium`
- `hard`

### RQ4: Robustness to Imperfect Safety Information

How sensitive is PLPG to noisy or imperfect safety facts?

### RQ5: Multi-Agent Applicability

Can PLPG remain effective in a **decentralized shared-policy multi-agent RL** setting?

### RQ6: Shield Design Sensitivity

How sensitive is PLPG to the choice of:

- safety weight `alpha`
- shield action granularity
- shield rule strength
- shield fact set
- shield acting-only vs full safety-regularized learning

## Core Hypotheses

- `H1`: PLPG improves `TSR` and lowers failure rate relative to PPO.
- `H2`: PLPG converges faster than PPO because unsafe exploration is reduced.
- `H3`: The gain from PLPG increases with task difficulty until shield mismatch or over-conservatism becomes limiting.
- `H4`: Increasing `alpha` improves safety but may increase conservatism and timeout rate.
- `H5`: PLPG remains beneficial under moderate sensor noise.
- `H6`: In decentralized shared-policy MARL, PLPG reduces unsafe joint behavior and stabilizes learning.
- `H7`: A refined action-sensitive shield improves `TSR` and reduces timeout collapse relative to a coarse stop-vs-move shield.
- `H8`: PLPG remains beneficial as interaction density increases through larger maps and more agents, but shield mismatch and compute overhead may become limiting.

## Methods To Compare

### Primary Methods

- `PPO`
- `PLPG-PPO`

### PLPG Ablations

- `PPO + interaction-biased resets` only
- `PLPG shield-only`
  - shielded acting and learning through `pi+`
  - `alpha = 0`
- `PLPG full`
  - shielded acting and learning
  - `alpha > 0`

### Optional Stronger Baseline

If needed for fairness, include one stronger PPO baseline:

- `PPO + clipped progress reward`

This should only be added if the plain PPO baseline is too weak to make the comparison scientifically useful.

## Experiment Family 1: Single-Agent Main Benchmark

Compare `PPO` and `PLPG-PPO` on:

- `easy`
- `medium`
- `hard`

### Revised Design Principle

For the main benchmark, all three difficulties should train in a **compact high-interaction regime**.

The intended design is:

- train all `easy`, `medium`, and `hard` variants on the compact `2x2` map
- keep the main difficulty progression in:
  - ontology
  - task rules
  - semantic traffic-agent roles
- avoid using sparse large-map training setups for the main comparison

This design is preferred because the thesis focus is on:

- safety-constrained decision making
- logic-guided policy learning
- training efficiency under frequent useful interactions

rather than large-scale route planning.

### Main Benchmark Regime

Use:

- same compact `2x2` training map across difficulties
- same action space
- same observation-space structure family
- same reset policy family
- same validation protocol
- same held-out in-distribution test protocol

Difficulty should mainly differ by:

- task-rule complexity
- ontology richness
- semantic role interactions

### Generalization Regime

Larger maps and denser traffic should be treated as a **separate generalization experiment**, not as part of the main benchmark.

That generalization study should evaluate whether policies trained on compact `2x2` SPF transfer to:

- larger maps such as `5x5`
- denser traffic
- more background agents

This gives two clean claims:

- main benchmark:
  - effectiveness and sample efficiency in dense useful interaction settings
- generalization benchmark:
  - transfer to larger and more complex layouts

### Proposed Difficulty Design Under the Compact Training Regime

#### Easy

- compact `2x2`
- simple stop rule
- minimal semantic traffic roles

#### Medium

- compact `2x2`
- medium ontology and stop-rule structure
- semantic interactions such as:
  - ambulance-related yielding
  - bus / pedestrian interaction
  - old-agent exceptions

#### Hard

- compact `2x2`
- full ontology
- richer semantic relations such as:
  - police-related interaction
  - left/right/close relations
  - the richest stop-rule hazard structure

### Design Goal

Under this revised design, the main difficulty increase should come from:

- **what kinds of safety relations the policy must reason about**

not from:

- a larger map with long low-interaction travel segments

### Outputs

- learning curves
- final validation results
- final test results
- per-difficulty summary table
- compact-regime comparison across `easy/medium/hard`

### Experiment Family 1 Plots and Diagrams

The following plots should be produced for each difficulty:

- `TSR` vs training timesteps
- `failure_rate` vs training timesteps
- `timeout_rate` vs training timesteps
- `mean_reward` vs training timesteps
- `TSR` vs wall-clock training time
- `failure_rate` vs wall-clock training time
- `mean_reward` vs wall-clock training time

These are the core learning curves and directly answer:

- whether PLPG performs better
- whether PLPG converges faster
- whether PLPG remains competitive once extra shield computation time is included

In addition, the following summary diagrams should be produced:

- final test grouped bar chart for:
  - `TSR`
  - `failure_rate`
  - `timeout_rate`
  - `mean_reward`
- sample-efficiency comparison:
  - timesteps to target `TSR`
  - optional `AUC-TSR` comparison across methods
- safety-behavior curves:
  - `action_stop_count` vs timesteps
  - `decision_succ_action_stop` vs timesteps
  - per-rule failure counts vs timesteps

For PLPG runs, also produce shield-diagnostic plots:

- `mean_base_policy_safe_prob` vs timesteps
- `mean_shielded_policy_safe_prob` vs timesteps
- `mean_safety_gain` vs timesteps
- `shield_intervention_rate` vs timesteps
- `mean_policy_kl_base_to_shielded` vs timesteps

For both methods, also produce rollout-side training-signal plots:

- `rollout_failures_since_last_eval` vs timesteps
- `rollout_timeouts_since_last_eval` vs timesteps
- `rollout_successes_since_last_eval` vs timesteps

### Experiment Family 1 Metrics Required in CSV

To support the plots, summary tables, and research questions, each eval CSV row should contain the following information.

#### Run Identity

- `exp_name`
- `method`
- `difficulty`
- `seed`
- `split`
- `timestep`
- `eval_index`

These fields are necessary for merging and comparing runs across methods, seeds, and difficulties.

#### Core Eval Metrics

- `tsr`
- `mean_reward`
- `failure_rate`
- `timeout_rate`
- `mean_episode_length`

These are the minimum metrics required to answer the main effectiveness questions.

#### Action and Decision Metrics

- `action_slow_count`
- `action_normal_count`
- `action_fast_count`
- `action_stop_count`
- `decision_succ_action_slow`
- `decision_succ_action_normal`
- `decision_succ_action_fast`
- `decision_succ_action_stop`
- `mean_decision_succ`

These are especially important for Safe Path Following, because the stopping behavior is a central part of the task.

#### Failure-Type Metrics

- all `fail_rule_*` columns

These are needed to explain not only whether failure occurs, but what kind of failure occurs.

#### Training-Rollout Metrics

- `rollout_failures_since_last_eval`
- `rollout_timeouts_since_last_eval`
- `rollout_successes_since_last_eval`

These are needed for the training-efficiency story because they show what the agent is actually experiencing during training, not only during fixed validation.

#### PLPG-Specific Metrics

These should be logged in the same schema, even if they are zero or empty for plain PPO:

- `mean_base_policy_safe_prob`
- `mean_shielded_policy_safe_prob`
- `mean_safety_gain`
- `shield_intervention_rate`
- `mean_policy_kl_base_to_shielded`
- `mean_l1_shift_base_to_shielded`
- `hazard_step_rate`
- `shield_forced_stop_rate`
- `base_action_slow_count`
- `base_action_normal_count`
- `base_action_fast_count`
- `base_action_stop_count`
- `shielded_action_slow_count`
- `shielded_action_normal_count`
- `shielded_action_fast_count`
- `shielded_action_stop_count`
- `mean_safe_prob_action_slow`
- `mean_safe_prob_action_normal`
- `mean_safe_prob_action_fast`
- `mean_safe_prob_action_stop`

These are required to explain why PLPG works, not only whether it works.

#### Optional But Useful Extra Fields

- `wall_clock_time_sec`
- `best_tsr_so_far`

These are useful for additional efficiency analysis and cleaner plotting. In particular, `wall_clock_time_sec` is required to compare:

- performance vs timesteps
- performance vs real training time

### Minimum Recommended Plot Set

If plotting time is limited, the minimum set that should still be included in the thesis is:

1. `TSR` vs timestep
2. `failure_rate` vs timestep
3. `mean_reward` vs timestep
4. final test bar chart for `TSR`, `failure_rate`, `timeout_rate`, and `mean_reward`
5. PLPG diagnostic plot:
   - `shield_intervention_rate`
   - `mean_safety_gain`

This minimum set is sufficient to answer the main Experiment Family 1 questions while keeping the analysis manageable.

### Experiment Family 1 Execution Commands

The following commands assume:

- fixed validation and test splits are generated first
- each seed uses a distinct `exp` name
- test-time CSV outputs are written into the training run folder by setting `--log_dir` accordingly

### Important Revision

The command block below reflects the **current implementation state**, not yet the final compact-regime redesign.

The intended final Experiment Family 1 setup is:

- `easy`, `medium`, and `hard` all trained and evaluated in-distribution on `2x2`

The larger-map `5x5` evaluation should be moved into a separate generalization experiment.

So these commands should be treated as:

- current runnable baseline commands

not:

- the final thesis execution plan after the redesign

### Redesign Tasks For Experiment Family 1

Before the final full seed sweep, the following redesign tasks should be completed:

1. create `medium-lite` compact `2x2` task configs
2. create `hard-lite` compact `2x2` task configs
3. define compact `2x2` validation and test splits for `medium-lite` and `hard-lite`
4. preserve ontology/rule complexity while keeping the interaction-dense map
5. move larger-map and denser-traffic evaluation into a separate generalization family

### Redesign Sketch

The intended compact-training redesign is:

- `easy-lite-2x2`
  - current setup, already available
- `medium-lite-2x2`
  - keep compact map
  - use medium ontology
  - use medium expert/task rules
  - keep a denser but still manageable background roster
- `hard-lite-2x2`
  - keep compact map
  - use full ontology
  - use hard expert/task rules
  - keep a denser but still manageable background roster

### Generalization Family Preview

After the compact main benchmark, define a separate generalization family such as:

- train on `2x2`
- test on `5x5`
- test with more background agents
- test with denser semantic-role traffic

This should become a separate experiment family rather than being mixed into the main benchmark.

#### 1. Generate Real Validation Splits

Easy:

```powershell
python tools/create_episode.py --config config/tasks/Nav/easy/experts/expert_episode_val_lite.yaml --exp easy_lite_val --max_episodes 40 --num_workers 4 --vis_count 0 --progress_every 5 --seed 101
```

Medium:

```powershell
python tools/create_episode.py --config config/tasks/Nav/medium/experts/expert_episode_val_lite.yaml --exp medium_lite_val --max_episodes 40 --num_workers 4 --vis_count 0 --progress_every 5 --seed 101
```

Hard:

```powershell
python tools/create_episode.py --config config/tasks/Nav/hard/experts/expert_episode_val_lite.yaml --exp hard_lite_val --max_episodes 40 --num_workers 4 --vis_count 0 --progress_every 5 --seed 101
```

#### 2. Generate Real Test Splits

Easy:

```powershell
python tools/create_episode.py --config config/tasks/Nav/easy/experts/expert_episode_test_lite.yaml --exp easy_lite_test --max_episodes 100 --num_workers 4 --vis_count 0 --progress_every 5 --seed 202
```

Medium:

```powershell
python tools/create_episode.py --config config/tasks/Nav/medium/experts/expert_episode_test_lite.yaml --exp medium_lite_test --max_episodes 100 --num_workers 4 --vis_count 0 --progress_every 5 --seed 202
```

Hard:

```powershell
python tools/create_episode.py --config config/tasks/Nav/hard/experts/expert_episode_test_lite.yaml --exp hard_lite_test --max_episodes 100 --num_workers 4 --vis_count 0 --progress_every 5 --seed 202
```

#### 3. Train PPO Baselines

Easy:

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/ppo_lite_eval.yaml --exp easy_lite_ppo_seed101 --seed 101
```

Medium:

```powershell
python main.py --use_gym --config config/tasks/Nav/medium/algo/ppo_lite_eval.yaml --exp medium_lite_ppo_seed101 --seed 101
```

Hard:

```powershell
python main.py --use_gym --config config/tasks/Nav/hard/algo/ppo_lite_eval.yaml --exp hard_lite_ppo_seed101 --seed 101
```

#### 4. Train PLPG-PPO

Easy:

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/plpg_ppo_eval.yaml --exp easy_lite_plpg_seed101 --seed 101
```

Medium:

```powershell
python main.py --use_gym --config config/tasks/Nav/medium/algo/plpg_ppo_eval.yaml --exp medium_lite_plpg_seed101 --seed 101
```

Hard:

```powershell
python main.py --use_gym --config config/tasks/Nav/hard/algo/plpg_ppo_eval.yaml --exp hard_lite_plpg_seed101 --seed 101
```

#### 5. Final Test Evaluation for PPO

Easy:

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/ppo_lite_test.yaml --exp easy_lite_ppo_test_seed101 --seed 101 --checkpoint_path checkpoints/easy_lite_ppo_seed101/best_model.zip --log_dir checkpoints/easy_lite_ppo_seed101
```

Medium:

```powershell
python main.py --use_gym --config config/tasks/Nav/medium/algo/ppo_lite_test.yaml --exp medium_lite_ppo_test_seed101 --seed 101 --checkpoint_path checkpoints/medium_lite_ppo_seed101/best_model.zip --log_dir checkpoints/medium_lite_ppo_seed101
```

Hard:

```powershell
python main.py --use_gym --config config/tasks/Nav/hard/algo/ppo_lite_test.yaml --exp hard_lite_ppo_test_seed101 --seed 101 --checkpoint_path checkpoints/hard_lite_ppo_seed101/best_model.zip --log_dir checkpoints/hard_lite_ppo_seed101
```

#### 6. Final Test Evaluation for PLPG-PPO

Easy:

```powershell
python main.py --use_gym --config config/tasks/Nav/easy/algo/plpg_ppo_test.yaml --exp easy_lite_plpg_test_seed101 --seed 101 --checkpoint_path checkpoints/easy_lite_plpg_seed101/best_model.zip --log_dir checkpoints/easy_lite_plpg_seed101
```

Medium:

```powershell
python main.py --use_gym --config config/tasks/Nav/medium/algo/plpg_ppo_test.yaml --exp medium_lite_plpg_test_seed101 --seed 101 --checkpoint_path checkpoints/medium_lite_plpg_seed101/best_model.zip --log_dir checkpoints/medium_lite_plpg_seed101
```

Hard:

```powershell
python main.py --use_gym --config config/tasks/Nav/hard/algo/plpg_ppo_test.yaml --exp hard_lite_plpg_test_seed101 --seed 101 --checkpoint_path checkpoints/hard_lite_plpg_seed101/best_model.zip --log_dir checkpoints/hard_lite_plpg_seed101
```

#### 7. Seed Repeats

Repeat the training and final-test commands for each seed.

Recommended thesis seed set:

- `101`
- `202`
- `303`

The `exp` name should include the seed so that:

- checkpoints stay separate
- validation CSVs stay separate
- test CSVs stay separate

#### 8. Plotting Commands

After runs are complete, generate the Experiment Family 1 plots from the validation-eval CSVs and held-out test CSVs.

Easy, medium, and hard with one seed each:

```powershell
python tools/plot_experiment_family1.py --eval_csvs checkpoints/easy_lite_ppo_seed101/easy_lite_ppo_seed101_eval_metrics.csv checkpoints/easy_lite_plpg_seed101/easy_lite_plpg_seed101_eval_metrics.csv checkpoints/medium_lite_ppo_seed101/medium_lite_ppo_seed101_eval_metrics.csv checkpoints/medium_lite_plpg_seed101/medium_lite_plpg_seed101_eval_metrics.csv checkpoints/hard_lite_ppo_seed101/hard_lite_ppo_seed101_eval_metrics.csv checkpoints/hard_lite_plpg_seed101/hard_lite_plpg_seed101_eval_metrics.csv --test_csvs checkpoints/easy_lite_ppo_seed101/easy_lite_ppo_test_seed101_test_metrics.csv checkpoints/easy_lite_plpg_seed101/easy_lite_plpg_test_seed101_test_metrics.csv checkpoints/medium_lite_ppo_seed101/medium_lite_ppo_test_seed101_test_metrics.csv checkpoints/medium_lite_plpg_seed101/medium_lite_plpg_test_seed101_test_metrics.csv checkpoints/hard_lite_ppo_seed101/hard_lite_ppo_test_seed101_test_metrics.csv checkpoints/hard_lite_plpg_seed101/hard_lite_plpg_test_seed101_test_metrics.csv --output_dir plots/experiment_family_1
```

For multiple seeds, pass all seed-specific validation and test CSVs to the same command. The script will aggregate by:

- `difficulty`
- `method`
- `timestep`

and produce mean curves with standard-deviation shading across runs.

## Experiment Family 2: Alpha Sweep

Evaluate PLPG for multiple values of the safety term weight.

Recommended values:

- `alpha = 0.0`
- `alpha = 0.01`
- `alpha = 0.05`
- `alpha = 0.1`
- `alpha = 0.2`
- `alpha = 0.5`

### Why this matters

This experiment shows the safety-performance tradeoff:

- low `alpha`: weaker safety pressure
- high `alpha`: stronger safety pressure but possible over-conservatism

`alpha = 0.0` is especially important because it isolates the effect of the shield from the effect of the explicit safety regularizer.

## Experiment Family 3: Sensor Noise Robustness

Introduce controlled noise into the shield facts while leaving the environment dynamics unchanged.

Recommended noise levels:

- `0%`
- `5%`
- `10%`
- `20%`
- `30%`

Recommended noise modes:

- symmetric bit-flip noise
- false-positive biased noise
- false-negative biased noise

### Why this matters

The original PLPG motivation assumes probabilistic abstraction rather than perfect symbolic truth. This experiment tests how much PLPG depends on perfect safety facts.

## Experiment Family 4: Shared-Policy Multi-Agent RL

Train multiple RL-controlled cars with:

- decentralized local observations
- one shared policy

Compare:

- decentralized shared-policy PPO
- decentralized shared-policy PLPG

Recommended progression:

1. `2 RL cars`
2. `3 RL cars`

Start with a simpler `easy-lite` multi-agent setup before extending to harder settings.

### Why this matters

This would be a strong thesis contribution because it shows whether probabilistic logic shielding remains useful when multiple learning agents interact simultaneously.

## Experiment Family 5: Shield Design Ablations

This is the main place to study the shield itself.

### 5.1 Shield On vs Off

Compare:

- `PPO`
- `PLPG shield-only`
- `PLPG full`

This isolates:

- pure shielding effect
- extra safety-loss effect

### 5.2 Safety Rule Strength

This is a valid and interesting experiment, but it should be framed carefully.

The main task rule semantics should stay fixed for the core benchmark. However, the **shield model** can be varied as an ablation to study sensitivity.

Recommended variants:

- `strict shield`
  - current rule-complete shield for the task
- `reduced shield`
  - remove one hazard source at a time
- `partial shield`
  - only `CollidingClose`
  - only intersection occupancy
  - only higher-priority conflicts

This asks:

- which logic components matter most
- whether PLPG gains come from one dominant safety relation or from the full rule set

### 5.3 Shield Action Granularity

Compare two shield designs:

- `coarse shield`
  - current `Stop` vs `Move` semantics
  - all move actions treated equally once stopping is not needed
- `refined shield`
  - action-sensitive semantics for `Stop`, `Slow`, `Normal`, and `Fast`
  - movement actions are not all treated as equivalent

This is an important thesis contribution because it studies whether the symbolic shield should only decide:

- `stop` vs `proceed`

or whether it should also decide:

- `slow` vs `normal` vs `fast`

The motivating hypothesis is:

- coarse shields are easier to specify and strongly protective
- refined shields may better support safe progress and higher `TSR`
- refined shields may reduce timeout-dominant conservatism under uncertainty

This experiment should be evaluated under:

- perfect sensors
- moderate sensor noise

Recommended first implementation order:

1. keep the current `easy` shield as the coarse baseline
2. implement an `easy` refined shield first
3. compare coarse vs refined on `easy`
4. only then extend the refined shield to `medium` and `hard` if the result is promising

### 5.4 Fact-Set Ablation

Test shields built from different subsets of facts:

- full fact set
- no `HigherPri`
- no `CollidingClose`
- no `IsInInter`

This is a stronger structural ablation than simple noise.

### 5.5 Safety-Loss Source

Compare:

- safety loss computed from base policy safety
- safety loss computed from shielded policy safety

This is implementation-specific but scientifically relevant because it changes how aggressively the method regularizes the learner.

### 5.6 Shield Aggressiveness

If needed later, study softer vs harder shields.

For example:

- exact deterministic facts
- noisy probabilistic facts
- deliberately softened fact confidence

This is more principled than manually inventing arbitrary action penalties.

## Experiment Family 6: Larger-Map And Higher-Agent-Count Generalization

Evaluate whether the shielded policies learned in compact interaction-dense regimes remain effective when:

- the map becomes larger
- the number of agents increases
- interaction patterns become more varied and less local

This family should include both:

- `single RL agent + more background agents`
- `multiple RL agents + more total agents`

### 6.1 Single RL Agent Scaling

Train or fine-tune a single RL-controlled ego agent and evaluate on:

- larger maps such as `5x5`
- denser traffic than the compact benchmark
- more background cars, pedestrians, and semantic-role agents

Compare:

- `PPO`
- `PLPG coarse shield`
- `PLPG refined shield` if available

Questions:

- does PLPG still improve `TSR` when the environment is less local and more crowded?
- does the shield remain helpful when interactions are more numerous and diverse?
- does over-conservative stopping become worse as the environment scales?

### 6.2 Multi-RL-Agent Scaling

After the base shared-policy MARL setup is working, increase:

- number of RL-controlled agents
- total number of agents in the map
- map size

Recommended progression:

1. `2 RL cars` on compact map
2. `3 RL cars` on compact map
3. `2 RL cars` on larger map with more background agents
4. `3 RL cars` on larger map with more background agents

Questions:

- can PLPG still coordinate multiple RL agents under denser interaction?
- does the shield help avoid unsafe joint behavior in larger, busier scenes?
- how does performance degrade as interaction complexity rises?

### 6.3 Why This Matters

This experiment family strengthens the thesis because it tests whether the contribution is limited to:

- compact toy interaction settings

or whether it extends to:

- larger urban layouts
- denser traffic
- more simultaneous agents

This is especially important for LogiCity, since the simulator is meant to represent dynamic urban environments rather than only small intersection cases.

## Metrics

### Main Task Metrics

- `tsr`
- `mean_reward`
- `failure_rate`
- `timeout_rate`
- `mean_episode_length`

### Sample Efficiency Metrics

- timesteps to reach `TSR >= threshold`
- timesteps to reach `failure_rate <= threshold`
- area under the `TSR` learning curve
- area under the reward learning curve

### Safety Metrics

- per-rule failure counts
- `decision_succ_action_stop`
- rollout failures since last eval
- rollout timeouts since last eval
- rollout successes since last eval

### PLPG-Specific Metrics

- `mean_base_policy_safe_prob`
- `mean_shielded_policy_safe_prob`
- `mean_safety_gain`
- `shield_intervention_rate`
- `mean_policy_kl_base_to_shielded`
- `mean_l1_shift_base_to_shielded`
- `hazard_step_rate`
- `shield_forced_stop_rate`
- per-action safety means
- base-action counts
- shielded-action counts

### Multi-Agent Metrics

For decentralized shared-policy MARL, add:

- per-agent TSR
- joint success rate
- collision/failure rate across RL agents
- fairness or variance across agents if relevant

### Scaling And Density Metrics

For larger-map and higher-agent-count experiments, also report:

- number of total agents
- number of RL-controlled agents
- map size
- success rate as a function of agent density if possible
- wall-clock slowdown as agent count increases
- shield inference cost as agent count increases

### Systems Metrics

To make the study stronger, also measure:

- wall-clock training time
- steps per second
- average shield inference time
- one-time circuit compilation time
- cache hit rate if memoization is added later

## Experimental Protocol

### Seeds

Use:

- at least `3` seeds
- preferably `5` seeds for the final main benchmark if compute allows

### Data Splits

Use:

- fixed validation split for model selection
- fixed test split for final reporting

Do not tune on test.

### Model Selection

Use one fixed criterion across methods, for example:

- best validation `TSR`

If safety-first selection is preferred, define it clearly and keep it fixed.

### Training Budgets

Use the same budget across compared methods within a task.

If difficulty-specific budgets differ, justify them explicitly.

### Reporting

For final thesis results, report:

- mean across seeds
- standard deviation or confidence interval
- significance testing where appropriate

## Recommended Thesis Experiment Matrix

### Minimum Strong Thesis Version

1. `PPO vs PLPG` on `easy`, `medium`, `hard`
2. `alpha` sweep
3. sensor-noise robustness
4. shield-only vs full PLPG ablation
5. small-scale shared-policy multi-agent benchmark

### Best Version If Time Allows

1. single-agent benchmark across all difficulties
2. alpha sweep
3. noise robustness
4. shield fact/rule ablations
5. multi-agent shared-policy benchmark
6. larger-map and higher-agent-count generalization
7. systems analysis of shield overhead

## Recommended Figures and Tables

### Figures

- learning curves: `TSR` vs timesteps
- learning curves: failure rate vs timesteps
- alpha sweep tradeoff curves
- noise robustness curves
- PLPG intervention-rate curves
- shield safety gain curves
- larger-map / agent-density scaling curves

### Tables

- final `easy/medium/hard` benchmark
- ablation table
- noise robustness table
- multi-agent benchmark table
- larger-map / higher-agent-count generalization table
- systems overhead table

## What Counts As A Scientific Contribution

To make the thesis scientifically valuable, the contribution should not be framed as only:

- "PLPG beats PPO on one LogiCity setup"

The stronger contribution is:

- PLPG improves safe path following under symbolic traffic rules
- PLPG improves training efficiency relative to PPO
- PLPG remains robust under noisy safety abstractions
- shield granularity matters for safe progress versus over-conservative stopping
- PLPG can be analyzed through shield-specific diagnostics
- PLPG can extend to decentralized shared-policy multi-agent RL
- PLPG can be tested for scalability to larger maps and denser multi-agent urban scenes

That combination is much stronger than a single benchmark comparison.

## Practical Recommendation

The best execution order is:

1. finish strong `easy-lite` PPO vs PLPG experiments
2. stabilize the metrics and logging
3. scale to `medium` and `hard`
4. run `alpha` sweep
5. run noise experiments
6. implement and test shared-policy multi-agent PLPG
7. test larger maps and higher-agent-count settings
8. add shield ablations if time remains

## Notes On Shield Experiments

Yes, the shield itself should be studied.

The most scientifically useful shield-related experiment axes are:

- `alpha`
- shield-only vs full PLPG
- fact noise
- fact-set ablation
- rule-component ablation
- base-policy-safety vs shielded-policy-safety regularization

The least useful shield experiments are arbitrary manual tweaks that break the task semantics without a clear interpretation.

So if we vary shield strength, it should be done in interpretable ways:

- remove a fact family
- remove a rule component
- soften fact certainty
- compare no safety term vs safety term

That keeps the experiments defensible in a thesis.
