# Experimental Plan Summary

This thesis investigates whether **Probabilistic Logic Shields (PLS)** and **Probabilistic Logic Policy Gradients (PLPG)** improve safe and successful navigation in the LogiCity simulator.

The two main research questions are:

1. Does incorporating probabilistic logic shields improve **Trajectory Success Rate (TSR)**?
2. Does incorporating probabilistic logic shields reduce the number of **training steps** needed to reach strong performance?

To answer these questions, the experimental plan is organized into six experiment families.

## Experiment Family 1: Single-Agent Main Benchmark

**What we evaluate**

- `PPO` versus `PLPG-PPO`
- single-agent Safe Path Following
- `easy`, `medium`, and `hard`

**Main metrics**

- `TSR`
- `failure_rate`
- `timeout_rate`
- `mean_reward`
- training steps to reach strong validation performance
- wall-clock time

**Purpose**

- This is the core benchmark family.
- It directly tests whether PLPG improves task success and learning efficiency relative to a fully neural baseline.

**Contribution**

- Establishes the main thesis result in LogiCity.
- Shows whether probabilistic shielding helps not only safety, but also actual goal-reaching performance.

**Mini research questions**

- Does PLPG improve TSR compared with PPO?
- Does PLPG reduce unsafe failures without causing excessive timeouts?
- Does PLPG reach strong performance in fewer training steps?

## Experiment Family 2: Alpha Sweep

**What we evaluate**

- PLPG with different safety weights `alpha`
- fixed `easy` setup
- fixed seed and fixed shield design

**Main metrics**

- `TSR`
- `failure_rate`
- `timeout_rate`
- `mean_reward`
- `shield_intervention_rate`
- `mean_safety_gain`

**Purpose**

- Tests how strongly the explicit safety term should influence learning.
- Separates the effect of the shielded policy from the effect of the added safety regularization.

**Contribution**

- Clarifies the safety-performance tradeoff inside PLPG.
- Helps justify the final `alpha` used in the main benchmark.

**Mini research questions**

- Does stronger safety regularization improve TSR or only reduce failures?
- At what point does larger `alpha` become over-conservative?
- Is shielded acting alone already sufficient, or does the safety term add value?

## Experiment Family 3: Sensor Noise Robustness

**What we evaluate**

- PLPG under imperfect shield inputs
- soft probabilistic sensor noise
- bit-flip sensor noise
- initially on `easy`

**Main metrics**

- `TSR`
- `failure_rate`
- `timeout_rate`
- `mean_reward`
- base versus shielded safety probabilities
- `shield_intervention_rate`

**Purpose**

- Tests whether PLPG still helps when the symbolic safety facts are uncertain.
- This is important because the original PLPG idea is specifically motivated by probabilistic safety reasoning rather than perfectly deterministic sensing.

**Contribution**

- Moves beyond the ideal perfect-sensor setting.
- Shows the robustness limits of PLPG in LogiCity.
- Also reveals whether a shield collapses into overly conservative stopping under uncertainty.

**Mini research questions**

- Up to what noise level does PLPG remain effective?
- Does uncertainty reduce TSR mainly through failures or through timeout collapse?
- Are some noise models more harmful than others?

## Experiment Family 4: Shared-Policy Multi-Agent RL

**What we evaluate**

- multiple RL-controlled cars
- one shared decentralized policy
- `PPO` versus `PLPG`
- starting with `easy`

**Main metrics**

- `joint_tsr`
- per-agent TSR
- joint failure rate
- joint timeout rate
- failure attribution by RL agent

**Purpose**

- Extends the thesis beyond single-agent safe navigation.
- Tests whether PLPG remains useful when several learning agents interact in the same environment.

**Contribution**

- Directly addresses the multi-agent focus of the thesis assignment.
- Shows whether probabilistic logic shielding helps coordination, not only individual safety.

**Mini research questions**

- Does PLPG improve joint task success when multiple RL agents share the same environment?
- Does PLPG reduce unsafe joint behavior?
- Can a shared policy with decentralized local observations still benefit from symbolic shielding?

## Experiment Family 5: Shield Design Ablations

**What we evaluate**

- different shield designs and ablations, including:
- shield-only versus full PLPG
- coarse `Stop` versus `Move` shield
- refined `Stop` / `Slow` / `Normal` / `Fast` shield
- fact-set ablations
- rule-strength ablations
- safety-loss-source ablations

**Main metrics**

- `TSR`
- `failure_rate`
- `timeout_rate`
- `mean_reward`
- `shield_intervention_rate`
- `mean_safety_gain`

**Purpose**

- Understands why PLPG works or fails.
- Studies how the symbolic shield design affects safe progress.

**Contribution**

- This is a methodological contribution of the thesis.
- In particular, comparing a coarse shield to a refined action-sensitive shield can show whether symbolic action granularity improves TSR and reduces over-conservative stopping.

**Mini research questions**

- Is a coarse stop-versus-move shield sufficient?
- Does a refined action-sensitive shield improve safe progress?
- Which facts and rules are most responsible for PLPG’s gains?

## Experiment Family 6: Larger-Map and Higher-Agent-Count Generalization

**What we evaluate**

- larger maps such as `5x5`
- denser traffic
- more background agents
- both:
- single RL agent
- multiple RL agents

**Main metrics**

- `TSR`
- joint TSR for multi-agent settings
- `failure_rate`
- `timeout_rate`
- wall-clock cost
- scaling of shield overhead

**Purpose**

- Tests whether the benefits of PLPG extend beyond compact benchmark scenarios.
- Evaluates whether the method remains practical as interaction complexity increases.

**Contribution**

- Strengthens the external validity of the thesis.
- Shows whether PLPG remains useful in more realistic urban settings with more agents and more complex interactions.

**Mini research questions**

- Does PLPG still improve TSR on larger maps?
- Does the benefit persist when agent density increases?
- Does shielding remain practical as computational cost grows?

## Overall Structure of the Plan

The six families play different roles:

- **Family 1** answers the two main thesis questions directly.
- **Families 2 and 3** analyze PLPG behavior, calibration, and robustness.
- **Family 4** extends the method to the multi-agent setting required by the thesis assignment.
- **Family 5** studies the shield itself as a scientific design object.
- **Family 6** evaluates scaling and generalization beyond compact setups.

## Expected Thesis Contribution

Together, these experiments aim to show:

- whether PLPG improves `TSR`
- whether PLPG improves sample efficiency
- whether PLPG remains useful under uncertainty
- whether shield design affects safe progress versus over-conservative stopping
- whether PLPG extends to multi-agent urban interaction
- whether the method scales to larger and denser LogiCity scenarios

The overall thesis contribution is therefore not only the implementation of PLPG in LogiCity, but also a structured empirical study of **when**, **why**, and **under which design choices** probabilistic logic shielding helps safe task completion.
