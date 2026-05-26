Abstract

1. Introduction
2. Background
   2.1 Reinforcement Learning and Policy Gradients
   2.2 Safe Reinforcement Learning and Shielding
   2.3 Probabilistic Logic Shields and PLPG
   2.4 LogiCity Simulator

3. Problem Setting
   3.1 Safe Path Following in LogiCity
   3.2 Agents, Actions, and Observations
   3.3 Grounded Observation Space by Difficulty
   3.4 Safety Rules and Failure Conditions
   3.5 Evaluation Metrics
   3.6 Single-Agent and Multi-Agent Task Variants

4. Method
   4.1 Neural RL Baseline
   4.2 LogiCity Probabilistic Logic Shield
   4.3 Shield Construction for Safe Path Following
   4.4 Perfect and Noisy Shield Inputs
   4.5 Multi-Agent Shared-Policy Extension

5. Experimental Setup
   5.1 Compared Methods
   5.2 Experiment Families

6. Results
   6.1 Family 1: Single-Agent Main Benchmark
   6.2 Family 2: Alpha Sweep
   6.3 Family 3: Sensor Noise Robustness
   6.4 Family 4: Shared-Policy Multi-Agent Results
   6.5 Family 5: Shield Granularity Study
   6.6 Family 6: Scaling and Generalization

7. Discussion
   7.1 Safety Versus Progress
   7.2 Shielded Policy Versus Base Policy
   7.3 Deterministic and Probabilistic Shield Behavior
   7.4 Limitations
   7.5 Implications for Neurosymbolic Safe RL in Urban Environments

8. Related Work

9. Conclusion and Future Work

Appendices
