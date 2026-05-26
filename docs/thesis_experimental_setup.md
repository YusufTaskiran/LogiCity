# Experimental Setup

## 5.1 Compared Methods

We compare two methods throughout the thesis. The first method is unshielded PPO, which serves as our fully neural baseline. This baseline receives the same grounded symbolic observation vector as all other methods, but it learns only from task reward and does not use any explicit probabilistic logic shield during acting or learning.

The second method is PLPG, i.e., PPO augmented with a Probabilistic Logic Shield constructed in ProbLog. In this method, the base policy is transformed into a shielded policy before action selection, and the learning objective additionally includes a safety regularization term. As described in Chapter 4, this makes PLPG a genuinely shielded policy-gradient method rather than a purely post-hoc action filter.

We use these two methods as the fixed comparison backbone of the thesis. Additional variations, such as different shield designs, different safety weights, different noise models, and single-agent versus multi-agent settings, are treated as experimental conditions within the experiment families rather than as separate top-level methods. This keeps the main comparison conceptually clean: we ask whether augmenting PPO with a probabilistic logic shield improves trajectory success and learning efficiency in LogiCity.

## 5.2 Experiment Families

We organize the empirical study into six experiment families. Together, these families address our two main research questions: whether probabilistic logic shielding improves Trajectory Success Rate (TSR), and whether it reduces the number of training steps required to reach strong performance.

### Family 1: Single-Agent Main Benchmark

In the first family, we compare PPO and PLPG on the single-agent Safe Path Following task across the easy, medium, and hard difficulty levels. This family is the core benchmark of the thesis. It addresses directly whether probabilistic shielding improves task success and whether it improves sample efficiency relative to the unshielded neural baseline.

### Family 2: Alpha Sweep

In the second family, we vary the PLPG safety weight \(\alpha\) on a fixed easy configuration. The purpose of this family is to understand how strongly the explicit safety regularizer should influence learning. This allows us to separate the effect of shielded acting from the effect of the additional safety loss and to study the trade-off between safety and over-conservatism.

### Family 3: Sensor Noise Robustness

In the third family, we evaluate PLPG under uncertain shield inputs. We inject noise into the grounded symbolic facts used by the shield while keeping the underlying LogiCity dynamics unchanged. We consider both soft probabilistic noise and bit-flip noise. This family is designed to test whether PLPG remains effective when the symbolic abstraction on which the shield relies is imperfect.

### Family 4: Shared-Policy Multi-Agent Reinforcement Learning

In the fourth family, we extend the study from a single RL-controlled vehicle to multiple RL-controlled vehicles that share one decentralized policy. We then compare PPO and PLPG in this shared-policy multi-agent setting. This family addresses the multi-agent aspect of the thesis assignment and allows us to test whether probabilistic shielding continues to help when several learning agents interact in the same urban environment.

### Family 5: Shield Granularity Study

In the fifth family, we study one specific shield-design question: whether a coarse stop-versus-move shield is sufficient, or whether a more refined action-sensitive shield leads to better safe task completion. Concretely, we compare two shield formulations. The first is the current coarse shield, which distinguishes only between actions that should stop and actions for which movement is appropriate. The second is a refined shield that distinguishes among `Stop`, `Slow`, `Normal`, and `Fast` more explicitly.

The motivation for this family comes directly from our LogiCity setting. A coarse shield is simpler and strongly protective, but it may remain too blunt to support nuanced safe progress in urban traffic. A refined shield, by contrast, may better capture when the agent should move cautiously, move normally, or move aggressively, rather than collapsing all movement into a single category. This family therefore tests whether increasing the symbolic action granularity of the shield improves TSR and reduces timeout-dominant conservatism.

### Family 6: Scaling and Generalization

In the sixth family, we evaluate the methods in larger and denser settings, including larger maps, more background agents, and, where applicable, multiple RL-controlled agents. This family tests whether the benefits of probabilistic logic shielding extend beyond compact benchmark scenarios and remain meaningful as interaction complexity increases.

Taken together, these six families provide a structured evaluation of probabilistic logic shielding in LogiCity. Family 1 establishes the main benchmark result. Families 2 and 3 analyze calibration and robustness. Family 4 addresses the decentralized multi-agent setting. Family 5 studies whether shield action granularity matters for safe progress, and Family 6 tests whether the overall approach generalizes to larger and more demanding scenarios. In this way, the experimental setup is designed not only to measure final performance, but also to explain when, why, and under which conditions probabilistic logic shielding helps safe task completion.
