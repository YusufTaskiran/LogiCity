# Experiment Execution Plans

This folder contains step-by-step execution plans for each thesis experiment family.

Use the files in this order:

1. `01_experiment_family1_main_benchmark.md`
2. `02_experiment_family2_alpha_sweep.md`
3. `03_experiment_family3_sensor_noise.md`
4. `04_experiment_family4_multi_agent.md`
5. `05_experiment_family5_shield_ablations.md`
6. `06_experiment_family6_scaling_generalization.md`

Recommended global execution order:

1. finish the `easy` single-agent baseline comparison
2. finish the `easy` method-analysis experiments
3. freeze the preferred PLPG design
4. run the full single-agent benchmark across `easy`, `medium`, and `hard`
5. extend to multi-agent and scaling experiments

General rules:

- keep validation and test splits fixed within a given experiment family
- use fresh `exp` names for restarted runs
- do not mix CSV rows from different runs in the same file
- keep only one changed variable per method-analysis experiment
- for final benchmark comparisons, keep the training budget fixed across compared methods
