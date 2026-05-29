# Thesis Roadmap

## Scope

The thesis studies **shared-policy multi-RL-agent Safe Path Following (SPF)** in LogiCity.

The **core thesis setting** is intentionally controlled:

- all RL-controlled agents are **normal cars**,
- the main background traffic consists of **normal cars** and **pedestrians**,
- all RL agents share the same action space, observation-space schema, and policy parameters,
- the single-agent case \(K=1\) is treated only as the simplest instantiation of the same multi-agent formulation.

Richer traffic roles such as **ambulance**, **bus**, and **police** are not part of the default thesis core. They are treated as an **optional extension/generalization setting** if time and compute permit.

## Method Focus

The fixed method comparison is:

- **PPO**: fully neural baseline
- **PLPG**: PPO with a ProbLog-based Probabilistic Logic Shield

The main shield-design comparison is:

- **coarse shield**: `Stop` versus `Move`
- **fine-grained shield**: `Stop`, `Slow`, `Normal`, `Fast`

In the current implementation:

- the rules are **crisp logical clauses**,
- the perfect-sensor mode binarizes grounded facts,
- therefore the perfect-sensor shield behaves effectively like a **hard deterministic action mask**,
- probabilistic behavior appears mainly when symbolic fact uncertainty is injected through noisy sensor modes.

## Research Questions

### RQ1

Does PLPG improve safe task completion relative to PPO?

This is first answered in the controlled \(K=1\) setting and then related to the multi-agent results.

### RQ2

Does PLPG remain beneficial as the number of RL-controlled agents increases?

This is the core multi-agent thesis question.

### RQ3

Does shielding remain useful when symbolic inputs are mildly noisy?

This is evaluated using one fixed noisy condition rather than a full robustness sweep.

### RQ4

Does shield granularity matter for safe progress?

This compares the coarse stop-vs-move shield against the fine-grained action-sensitive shield.

## Experiment Families

### Family 1: Controlled \(K=1\) Benchmark

- Compare PPO and PLPG
- Use the core normal-car-plus-pedestrian setting
- Serve as the controlled reference point for learning behavior, failures, and timeouts

### Family 2: Shared-Policy Multi-Agent Benchmark and \(K\)-Scaling

- Compare PPO and PLPG for \(K=2\)
- Extend to \(K=3\) if budget permits
- Explicitly compare performance across \(K=1 \rightarrow K=2 \rightarrow K=3\)

### Family 3: Multi-Agent Perfect vs Noisy Shield Inputs

- Compare perfect symbolic inputs against one fixed noisy mode
- Preferred default: soft symmetric noise with \(\varepsilon = 0.1\)

### Family 4: Shield Granularity Study

- Compare coarse shield against fine-grained shield
- Focus on whether finer symbolic action structure reduces timeout-dominant conservatism

## Metrics

For the multi-agent thesis setting, the primary metric is:

- **joint TSR**

Supporting metrics are:

- **joint failure rate**
- **joint timeout rate**
- **per-agent TSR**
- **mean reward**
- **mean episode length**

For sample efficiency, also track:

- training timesteps required to reach strong performance
- validation learning-curve shape over time

## Fixed Design Choices

To keep the thesis focused:

- use one fixed PLPG safety weight \(\alpha\),
- do **not** run a broad \(\alpha\)-sweep,
- do **not** run a full noise sweep over many levels and models,
- use one fixed mild noisy mode for the main noise question.

The thesis contribution is not to re-derive all PLPG calibration results from the original paper, but to test whether the method remains useful in a richer shared-policy multi-agent symbolic traffic setting.

## Writing Structure

- **Chapter 3**: present the problem as multi-agent by default
- **Chapter 3**: include a concrete grounded-observation example
- **Chapter 5**: organize the experimental setup by research questions
- **Chapter 6**: organize the results by research questions

The optional heterogeneous traffic setting with ambulance, bus, and police agents should be written as an extension or generalization study, not as the default basis of the thesis.

## Practical Priority Order

If time is limited, prioritize the thesis work in this order:

1. PPO vs PLPG in controlled \(K=1\)
2. PPO vs PLPG in multi-agent \(K=2\)
3. \(K=3\) scaling if stable
4. perfect vs noisy comparison
5. coarse vs fine-grained shield comparison
6. heterogeneous-traffic extension

## One-Sentence Thesis Story

This thesis asks whether probabilistic logic shielding improves **joint safe task completion** for multiple shared-policy RL-controlled normal cars in a symbolic traffic environment, and whether that benefit survives scaling, mild symbolic noise, and different shield granularities.
