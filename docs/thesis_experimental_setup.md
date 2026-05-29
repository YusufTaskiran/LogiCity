# Experimental Setup

## 5.1 Compared Methods

We compare two methods throughout the thesis. The first method is unshielded PPO, which serves as our fully neural baseline. This baseline receives the same grounded symbolic observation vector as all other methods, but it learns only from task reward and does not use any explicit probabilistic logic shield during acting or learning.

The second method is PLPG, i.e., PPO augmented with a Probabilistic Logic Shield constructed in ProbLog. In this method, the base policy is transformed into a shielded policy before action selection, and the learning objective additionally includes a safety regularization term. As described in Chapter 4, this makes PLPG a genuinely shielded policy-gradient method rather than a purely post-hoc action filter.

The thesis-level comparison is therefore fixed: PPO versus PLPG under the same shared-policy multi-agent SPF formulation. Variations such as the number of RL-controlled agents \(K\), shield-input quality, and shield granularity are treated as experimental conditions rather than as separate methods.

## 5.2 Scope of the Main Experiments

The **main thesis experiments** are conducted in a controlled LogiCity subset centered on interactions among normal cars and pedestrians. All RL-controlled agents are normal cars. The background traffic consists of additional normal cars and pedestrians controlled by the simulator's expert planner.

This scope is chosen deliberately. The main research questions of the thesis concern:

- whether probabilistic logic shielding improves safe task completion,
- whether that benefit remains present in the shared-policy multi-agent setting,
- whether it scales with the number of RL-controlled agents,
- and whether shield granularity matters for safe progress.

These questions can be answered cleanly in a homogeneous RL-agent setting. Richer vehicle roles such as ambulance, bus, and police cars are therefore not part of the default experimental core. If included, they are treated as an **extension setting** for testing whether the main conclusions generalize to more heterogeneous traffic semantics.

## 5.3 Fixed Design Choices

To keep the empirical study focused on the thesis contribution, we do not perform a broad re-investigation of PLPG hyperparameters. In particular, we do not include a standalone sweep over the safety weight \(\alpha\), and we do not dedicate a full experiment family to many noise models and many noise magnitudes. Those analyses are already motivated in the original PLPG work on simpler domains, whereas our thesis contribution is to study the method in the LogiCity shared-policy multi-agent setting.

Accordingly, we fix one PLPG safety weight \(\alpha\) across the main experiments. We also use a small number of fixed shield-input conditions rather than a broad robustness sweep. In particular, for the multi-agent experiments we compare:

- a **perfect-symbolic** shield setting,
- a **noisy-symbolic** shield setting with one fixed corruption level, e.g. soft symmetric noise with \(\varepsilon = 0.1\).

This design lets us test whether PLPG remains useful under mild symbolic uncertainty without turning noise calibration into a separate thesis objective.

## 5.4 Research Questions

We organize the empirical evaluation around four research questions. The experiment families are then used as concrete protocols for answering those questions.

### RQ1: Does probabilistic logic shielding improve safe task completion relative to PPO?

This question asks whether PLPG improves overall task performance in the controlled LogiCity core setting. We answer it first in the controlled \(K=1\) case and then in the full shared-policy multi-agent setting. The relevant outcome metrics are TSR, failure rate, timeout rate, and training efficiency.

The associated experiments are:

- **Family 1: Controlled \(K=1\) Benchmark**
- **Family 2: Shared-Policy Multi-Agent Benchmark**

The \(K=1\) benchmark is not the thesis problem in itself. Rather, it is the simplest controlled instantiation of the same shared-policy formulation and serves as a reference point before scaling to multiple RL-controlled cars.

### RQ2: Does probabilistic logic shielding remain beneficial as the number of RL-controlled agents increases?

This question focuses on the core multi-agent aspect of the thesis. The key issue is whether the shield remains useful once multiple RL-controlled normal cars interact in the same symbolic traffic environment.

We answer this question by explicitly varying the number of RL-controlled agents under the same shared-policy formulation:

- **Family 2: Shared-Policy Multi-Agent Benchmark and \(K\)-Scaling**

The primary metric for this question is **joint TSR**, supported by per-agent TSR, joint failure rate, joint timeout rate, and training-curve comparisons over \(K=1 \rightarrow K=2 \rightarrow K=3\).

### RQ3: Does shielding remain useful when the symbolic inputs are mildly noisy?

PLPG relies on grounded symbolic inputs. This question asks whether its benefit persists when those symbolic facts are not perfect, but are instead corrupted by one fixed mild noise condition.

We answer this question with:

- **Family 3: Multi-Agent Perfect vs Noisy Shield Inputs**

The purpose here is not to map a full robustness frontier. It is to test whether the method remains behaviorally useful under a plausible noisy-symbolic condition in the shared-policy multi-agent setting.

### RQ4: Does shield granularity matter for safe progress?

The final question is methodological. Even if shielding helps, it may matter how the shield represents action choices. A coarse stop-versus-move shield may prevent failures but remain too blunt to support efficient progress, whereas a more action-sensitive shield may better balance safety and motion.

We answer this question with:

- **Family 4: Shield Granularity Study**

The main metrics are TSR, failure rate, timeout rate, and any observable reduction in over-conservative stopping.

## 5.5 Experiment Families

For clarity, we summarize the concrete experiment families below.

### Family 1: Controlled \(K=1\) Benchmark

We compare PPO and PLPG on the \(K=1\) instantiation of SPF in the core normal-car-plus-pedestrian setting. This family provides the controlled baseline reference for learning behavior, safety failures, timeout collapse, and shield effects.

### Family 2: Shared-Policy Multi-Agent Benchmark and \(K\)-Scaling

We evaluate PPO and PLPG in the thesis-level shared-policy multi-agent setting with \(K>1\). We begin with \(K=2\) RL-controlled normal cars and extend to \(K=3\) if computational budget permits. Within the same family, we explicitly compare performance across \(K=1 \rightarrow K=2 \rightarrow K=3\) so that agent-count scaling is not treated as a side detail.

For this family, the primary evaluation metric is **joint TSR**, i.e., the fraction of episodes in which all RL-controlled agents reach their goals before any failure and before the horizon is exceeded. We also report per-agent TSR, joint failure rate, joint timeout rate, mean reward, and mean episode length.

### Family 3: Multi-Agent Perfect vs Noisy Shield Inputs

We evaluate the shared-policy multi-agent setting under two fixed shield-input conditions: perfect symbolic inputs and one noisy symbolic-input mode. The noisy condition uses a single fixed corruption level, such as soft symmetric noise with \(\varepsilon = 0.1\). The underlying LogiCity dynamics remain unchanged.

### Family 4: Shield Granularity Study

We compare a coarse stop-versus-move shield against a refined action-sensitive shield. The coarse shield distinguishes only between actions that should stop and actions for which movement is appropriate. The refined shield distinguishes more explicitly among `Stop`, `Slow`, `Normal`, and `Fast`.

## 5.6 Optional Extension Setting

If time and computational budget permit, we additionally evaluate an extension setting with richer heterogeneous background traffic, including ambulance, bus, and police agents. The purpose of this extension is not to redefine the main thesis problem, but to test whether the qualitative conclusions from the controlled core setting continue to hold under more complex traffic semantics.

## 5.7 Overall Evaluation Logic

Taken together, these four research questions and four experiment families provide a structured evaluation of probabilistic logic shielding in LogiCity.

- RQ1 establishes whether PLPG improves safe task completion relative to PPO.
- RQ2 tests whether that benefit survives and scales in the thesis-level multi-agent setting.
- RQ3 checks whether the shield remains useful under one fixed mild noisy-symbolic condition.
- RQ4 studies whether shield action granularity matters for achieving safe progress instead of timeout-dominant conservatism.

This structure is more tightly aligned with the thesis scope than an extensive \(\alpha\)-sweep or a broad noise-robustness benchmark. The goal is not to re-derive all calibration findings from the original PLPG paper, but to determine whether the method remains effective, practical, and behaviorally meaningful in a controlled shared-policy multi-agent symbolic traffic setting, and optionally whether those findings extend to richer heterogeneous traffic.
