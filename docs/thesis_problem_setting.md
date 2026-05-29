# Problem Setting

## 3.1 Safe Path Following in LogiCity

In this thesis, we study **Safe Path Following (SPF)** in the LogiCity simulator under a shared-policy **multi-RL-agent** setting. Let \(K\) denote the number of RL-controlled cars. Each RL agent must reach its own destination while respecting traffic-like logical constraints induced by the map structure and by surrounding agents, which may include both expert-controlled traffic participants and other RL-controlled cars. We do not study low-level vehicle control. Instead, we study the decision problem of when each RL-controlled vehicle should move and when it should stop in the presence of rule-relevant traffic interactions.

This distinction matters because the policy can fail in two qualitatively different ways. It can fail **unsafely**, by causing one or more RL agents to violate a logical traffic rule and trigger task failure. It can also fail **conservatively**, by inducing excessive stopping so that the agents do not reach their goals before the horizon expires. We therefore formulate SPF as a safe task-completion problem rather than as a pure safety-filtering problem. The shared policy must not only avoid rule violations, but also guide all RL-controlled agents to make sufficient progress.

The LogiCity simulator is well suited to this problem because it exposes safety-relevant structure through symbolic predicates and explicit task rules. Instead of learning from pixels, each RL agent operates on a grounded logical representation of nearby entities and their relations. This makes it possible to study shielding in a domain where the safety specification is already naturally relational and interpretable.

## 3.2 Shared-Policy Multi-Agent Task

At the thesis level, SPF is defined for \(K \ge 1\) RL-controlled cars that share the same policy architecture, action space, and observation-space schema. At each timestep, every RL agent receives its own local observation and independently chooses an action from the same action set, but all RL agents use one shared parameter vector \(\theta\). Background traffic participants remain controlled by the simulator's logic-based expert planner.

The core thesis setting uses a **controlled LogiCity subset** in which the RL-controlled agents are normal cars and the main non-RL traffic participants are normal cars and pedestrians. This keeps the shared-policy multi-agent problem homogeneous on the RL side while still preserving interactive symbolic traffic structure.

This means that the thesis problem is multi-agent by construction. The case \(K=1\) is not a separate task definition. It is only the simplest instantiation of the same shared-policy formulation and is useful as a controlled implementation setting before training with multiple RL-controlled cars simultaneously.

In the current implementation, we use the following staged instantiations of the same overall problem:

- a controlled benchmark with \(K=1\),
- a shared-policy multi-agent setting with \(K=2\),
- and, if budget permits, an extended setting with \(K=3\).

## 3.3 Agents, Actions, and Observations

### Agents in the core thesis setting

In the main experiments of the thesis, all RL-controlled agents are **normal cars**. The controlled background traffic consists of additional normal cars and pedestrians. This choice is deliberate: it keeps the shared policy semantically homogeneous across RL agents and makes the multi-agent scaling results easier to interpret.

Thus, the core thesis questions are answered in a controlled traffic setting with:

- RL-controlled normal cars,
- expert-controlled normal cars,
- expert-controlled pedestrians.

Richer traffic roles such as ambulance, bus, and police vehicles are not part of the default thesis setting. If included, they are treated as an **extension setting** used to explore whether the same qualitative conclusions continue to hold under more heterogeneous traffic semantics.

### Shared action space

Each RL-controlled car uses the same discrete four-action space:

\[
\mathcal{A} = \{\texttt{Slow}, \texttt{Normal}, \texttt{Fast}, \texttt{Stop}\}.
\]

These four macro-actions are mapped to the simulator's underlying 13-dimensional car action model:

- `Slow`: one-cell movement in one of the four cardinal directions,
- `Normal`: two-cell movement in one of the four cardinal directions,
- `Fast`: three-cell movement in one of the four cardinal directions,
- `Stop`: no movement.

At the car level, the underlying action set is:

\[
\{\texttt{Left\_Slow}, \texttt{Right\_Slow}, \texttt{Up\_Slow}, \texttt{Down\_Slow},
\texttt{Left\_Normal}, \dots, \texttt{Down\_Fast}, \texttt{Stop}\}.
\]

However, the RL policy does not choose directions directly. The direction is constrained by the car's route and local planner. In practice:

- `Slow` means "advance one grid cell along an admissible route direction,"
- `Normal` means "advance two grid cells along an admissible route direction,"
- `Fast` means "advance three grid cells along an admissible route direction,"
- `Stop` means "remain in place."

All RL-controlled agents therefore share the same action semantics. The learning problem is a shared-policy **speed-mode decision problem under safety constraints**.

### Shared observation-space schema

We do not use visual observations. Each RL agent receives its own flattened grounded logical state vector over a local field of view. Within a given difficulty, all RL agents share the same observation-space schema, so the observation space is represented as a continuous box

\[
\mathcal{O} \subseteq [0,1]^d,
\]

where \(d\) depends on the active ontology and on the field-of-view entity budget.

In the core thesis setting, this ontology is intentionally kept compact and centered on normal-car and pedestrian interactions. Each unary predicate contributes one grounded value per local entity, and each binary predicate contributes one grounded value per ordered entity pair. Predicates whose function is `None` in the ontology, such as action predicates, are not included in the observation. Thus, all RL agents share the same feature template, even though the actual grounded values differ across agents and states.

## 3.4 Example of a Grounded Observation

Because the observation is flattened before being passed to the neural policy, it is helpful to illustrate what one grounded state actually looks like. Consider a simple core-setting case with three local entities in the field of view:

- entity slot 1: another normal car,
- entity slot 2: a pedestrian,
- entity slot 3: another normal car.

Suppose the compact ontology contains unary predicates such as `IsPedestrian`, `IsCar`, `IsAtInter`, and `IsInInter`, and binary predicates such as `HigherPri` and `CollidingClose`. One possible grounded observation for the ego agent can then be written schematically as

\[
o =
[
\underbrace{0,1,1,0}_{\text{entity 1 unary}},
\underbrace{1,0,0,0}_{\text{entity 2 unary}},
\underbrace{0,1,0,1}_{\text{entity 3 unary}},
\underbrace{0,1,0,0,0,0,0,0,0}_{\texttt{HigherPri}},
\underbrace{1,0,0,0,0,0,0,0,0}_{\texttt{CollidingClose}}
].
\]

The first three blocks encode unary facts for the three entity slots. In the example above, the first entity is a car at an intersection, the second is a pedestrian, and the third is a car currently in an intersection. The last two blocks encode binary relations over ordered pairs of entity slots. Here, one `HigherPri` relation is active and one `CollidingClose` relation is active, while the remaining ordered pairs are zero.

The exact ordering of features is implementation-dependent, but the important point is that the policy does not observe raw symbolic formulas. It observes a fixed-length vector of grounded predicate values. The shield then reads the same grounded information again, but interprets it symbolically through ProbLog facts rather than only through neural function approximation.

## 3.5 Core Observation Space and Extension Space

In the core thesis setting, the observation space is intentionally compact because the experiments focus on multi-agent scaling and shield design rather than on heterogeneous vehicle roles. The exact observation dimension depends on the chosen compact ontology and field-of-view budget, but the same general principle holds throughout: the observation is a flattened vector of grounded unary and binary predicate values shared across all RL agents.

Difficulty in the core setting can still increase in two ways:

1. more entities are represented locally,
2. more relational structure must be tracked among normal cars and pedestrians.

Thus, harder settings still require the shared policy to reason over richer symbolic interactions, even without introducing ambulance, bus, and police semantics into the main thesis experiments.

If time permits, we additionally consider an **extension setting** with richer heterogeneous background traffic. In that extension, the ontology can be expanded to include ambulance, bus, and police predicates and their associated relations. However, those richer semantics are not required to answer the main multi-agent research questions of the thesis.

## 3.6 Safety Rules and Joint Episode Outcomes

In the core SPF task family, safety is defined through explicit **Task** rules in the navigation rule files. At a high level, the active task failure condition has the form

\[
\texttt{hazard condition} \Rightarrow \texttt{Stop(entity)}.
\]

Thus, when a rule-relevant hazardous condition holds for a given RL-controlled car, that car must choose `Stop`. If it fails to do so, the episode terminates with failure and receives the configured penalty.

In the core thesis setting, the hazard conditions are intentionally centered on normal-car and pedestrian interactions, such as blocked intersection occupancy, priority conflicts, and imminent collision proximity. This is sufficient to create a meaningful symbolic shielding problem while keeping the main multi-agent study focused and interpretable.

If we include the richer extension setting, additional role-dependent rules involving ambulance, bus, or police traffic can be added on top of the same overall task structure. Those rules are treated as an extension of the thesis rather than as the default basis for the main research questions.

### Joint termination and outcome types

At the environment level, an episode terminates if:

1. all RL-controlled agents reach their goals,
2. the horizon is exceeded,
3. any RL-controlled agent violates a task rule.

This yields three outcome types:

- **success**: all RL-controlled agents reach their goals,
- **failure**: at least one RL-controlled agent triggers a task-rule violation,
- **timeout**: the horizon is exceeded before joint success.

This distinction is important because high safety alone does not imply high task performance: a shared policy may avoid failures but still induce timeout-dominant behavior. When we instantiate the special case \(K=1\), these same definitions reduce to the familiar single-agent outcomes.

## 3.7 Evaluation Metrics

Our primary task metric is **Trajectory Success Rate (TSR)**, defined as the fraction of episodes in which the RL-controlled system succeeds. In the general multi-agent case, this means **joint TSR**: all RL-controlled agents reach their goals before any failure and before the horizon is exceeded.

We focus on joint TSR because it captures the actual objective of shared-policy SPF: safe completion of the route set for all RL-controlled agents. To understand why a method succeeds or fails, we also evaluate:

- **joint failure rate**, the fraction of episodes ending in task-rule failure,
- **joint timeout rate**, the fraction of episodes ending due to horizon exhaustion,
- **per-agent TSR**, the fraction of RL-controlled agents that individually reach their goals,
- **mean reward**, the average episode return,
- **mean episode length**, the average number of steps taken per episode.

These metrics are complementary:

- joint failure rate measures unsafe behavior,
- joint timeout rate measures over-conservative or ineffective behavior,
- per-agent TSR separates local progress from joint coordination success,
- joint TSR measures the final balance between safety and coordinated progress.

For the sample-efficiency question, we additionally evaluate:

- the number of training timesteps required to reach strong joint TSR,
- the shape of the validation learning curves over time,
- the trajectory of failures and timeouts during training.

For shielded methods, we also log shield-specific diagnostics such as intervention rate and estimated policy safety. These are explanatory metrics rather than part of the task definition itself.
