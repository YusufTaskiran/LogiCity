# Problem Setting

## 3.1 Safe Path Following in LogiCity

In this thesis, we study **Safe Path Following (SPF)** in the LogiCity simulator. The task is a goal-directed urban navigation problem in which an RL-controlled car must reach its destination while respecting traffic-like logical constraints induced by the map structure and by surrounding agents. We do not study low-level vehicle control. Instead, we study the decision problem of when the ego vehicle should move and when it should stop in the presence of rule-relevant traffic interactions.

This distinction is important. In our setting, a policy can fail in two qualitatively different ways. It can fail **unsafely**, by violating a logical traffic rule and triggering a task failure. It can also fail **conservatively**, by stopping too often and timing out before reaching the goal. We therefore formulate SPF as a safe task-completion problem rather than as a pure safety-filtering problem. The agent must not only avoid rule violations, but also complete its trajectory efficiently enough to succeed within the horizon.

The LogiCity simulator is well suited to this problem because it exposes safety-relevant structure through symbolic predicates and explicit task rules. Instead of learning from pixels, the agent operates on a grounded logical representation of nearby entities and their relations. This makes it possible to study shielding in a domain where the safety specification is already naturally relational and interpretable.

## 3.2 Agents, Actions, and Observations

### Agents in the experiments

Across the main single-agent experiments, the RL-controlled agent is always `Car_1`. In the Lite SPF task family used in this thesis, `Car_1` is a normal car and the remaining agents are background traffic participants controlled by the simulator’s logic-based expert planner. The exact roster varies by difficulty.

In the current training setups, we use the following agent rosters:

- **easy-lite**
  - `Car_1`: RL-controlled normal car
  - `Pedestrian_1`: old pedestrian
  - `Car_2`: ambulance car
  - `Car_3`: normal car

- **medium-lite**
  - `Car_1`: RL-controlled normal car
  - `Pedestrian_1`: old pedestrian
  - `Car_2`: ambulance car
  - `Car_3`: bus car
  - `Car_4`: normal car

- **hard-lite**
  - `Car_1`: RL-controlled normal car
  - `Pedestrian_1`: old pedestrian
  - `Car_2`: ambulance car
  - `Car_3`: police car
  - `Car_4`: bus car
  - `Car_5`: normal car

Thus, difficulty is not increased merely by adding more objects to the map. It is increased by adding more semantically distinct traffic participants, which in turn activates richer rule conditions.

### Actions

The RL policy acts in a discrete four-action space:

\[
\mathcal{A} = \{\texttt{Slow}, \texttt{Normal}, \texttt{Fast}, \texttt{Stop}\}.
\]

These four macro-actions are mapped to the simulator’s underlying 13-dimensional car action model:

- `Slow`: one-cell movement in one of the four cardinal directions,
- `Normal`: two-cell movement in one of the four cardinal directions,
- `Fast`: three-cell movement in one of the four cardinal directions,
- `Stop`: no movement.

At the car level, the underlying action set is:

\[
\{\texttt{Left\_Slow}, \texttt{Right\_Slow}, \texttt{Up\_Slow}, \texttt{Down\_Slow},
\texttt{Left\_Normal}, \dots, \texttt{Down\_Fast}, \texttt{Stop}\}.
\]

However, the RL policy does not choose directions directly. The direction is implicitly constrained by the car’s route and local planner. In practice:

- `Slow` means “advance one grid cell along an admissible route direction,”
- `Normal` means “advance two grid cells along an admissible route direction,”
- `Fast` means “advance three grid cells along an admissible route direction,”
- `Stop` means “remain in place.”

This makes the RL problem fundamentally a **speed-mode decision problem under safety constraints**.

### Observations

We do not use visual observations. The policy receives a flattened grounded logical state vector over a local field of view. The observation space is represented as a continuous box:

\[
\mathcal{O} \subseteq [0,1]^d,
\]

where \(d\) depends on the active ontology and the field-of-view entity budget.

The field-of-view entity budgets are:

- **easy-lite**: `fov_entities.Entity = 3`
- **medium-lite**: `fov_entities.Entity = 4`
- **hard-lite**: `fov_entities.Entity = 5`

Each unary predicate contributes one grounded value per local entity, and each binary predicate contributes one grounded value per ordered entity pair. Predicates whose function is `None` in the ontology, such as action predicates, are not included in the observation.

## 3.3 Grounded Observation Space by Difficulty

The observation dimension differs by difficulty because the predicate set becomes richer and the field-of-view entity budget increases.

### Easy-lite observation space

The `easy` ontology contains:

- 7 grounded unary predicates:
  - `IsPedestrian`
  - `IsCar`
  - `IsAmbulance`
  - `IsOld`
  - `IsTiro`
  - `IsAtInter`
  - `IsInInter`
- 2 grounded binary predicates:
  - `HigherPri`
  - `CollidingClose`

With `3` entities in the field of view, the observation dimension is:

\[
7 \cdot 3 + 2 \cdot 3^2 = 21 + 18 = 39.
\]

So the easy-lite observation space is:

\[
\mathcal{O}_{\text{easy}} \subseteq [0,1]^{39}.
\]

### Medium-lite observation space

The `medium` ontology contains:

- 8 grounded unary predicates:
  - `IsPedestrian`
  - `IsCar`
  - `IsAmbulance`
  - `IsOld`
  - `IsTiro`
  - `IsBus`
  - `IsAtInter`
  - `IsInInter`
- 4 grounded binary predicates:
  - `HigherPri`
  - `CollidingClose`
  - `NextTo`
  - `RightOf`

With `4` entities in the field of view, the observation dimension is:

\[
8 \cdot 4 + 4 \cdot 4^2 = 32 + 64 = 96.
\]

So the medium-lite observation space is:

\[
\mathcal{O}_{\text{medium}} \subseteq [0,1]^{96}.
\]

### Hard-lite observation space

The `hard` ontology contains:

- 11 grounded unary predicates:
  - `IsPedestrian`
  - `IsCar`
  - `IsAmbulance`
  - `IsBus`
  - `IsPolice`
  - `IsTiro`
  - `IsReckless`
  - `IsOld`
  - `IsYoung`
  - `IsAtInter`
  - `IsInInter`
- 6 grounded binary predicates:
  - `IsClose`
  - `HigherPri`
  - `CollidingClose`
  - `LeftOf`
  - `RightOf`
  - `NextTo`

With `5` entities in the field of view, the observation dimension is:

\[
11 \cdot 5 + 6 \cdot 5^2 = 55 + 150 = 205.
\]

So the hard-lite observation space is:

\[
\mathcal{O}_{\text{hard}} \subseteq [0,1]^{205}.
\]

### Interpretation

The important point is that the observation difficulty grows in two ways:

1. more entities are represented locally,
2. more semantic and relational predicates are grounded.

This means that harder settings require the policy to reason over a richer symbolic state, not merely over longer trajectories.

## 3.4 Safety Rules and Failure Conditions

In the SPF task family, safety is defined through explicit **Task** rules in the navigation rule files. In all three current Lite difficulties, the active task failure condition is a stop-style implication:

\[
\texttt{hazard condition} \Rightarrow \texttt{Stop(entity)}.
\]

Thus, when a rule-relevant hazardous condition holds, the RL-controlled car must choose `Stop`. If it fails to do so, the episode terminates with failure and receives the configured penalty.

### Easy-lite safety rule

In the `easy` difficulty, the task rule requires the ego car to stop if there exists another entity such that at least one of the following is true:

1. the ego is at an intersection and the other entity is in the intersection,
2. the ego is at an intersection, the other entity is also at the intersection, and the other entity has higher priority,
3. the ego is collision-close to the other entity.

So the easy-lite rule set captures three core urban safety relations:

- blocked intersection occupancy,
- intersection priority,
- imminent collision proximity.

### Medium-lite safety rule

In the `medium` difficulty, the stop condition is extended. The ego car must stop if there exists another entity such that at least one of the following holds:

1. a non-ambulance, non-old ego car is at an intersection while another entity is in the intersection,
2. a non-ambulance, non-old ego car is at an intersection while another entity at the intersection has higher priority,
3. a non-ambulance, non-old ego car is in an intersection while another entity in the intersection is an ambulance,
4. a bus ego vehicle is not at or in an intersection and a pedestrian is both to its right and next to it,
5. an ambulance ego vehicle has an old entity to its right,
6. a non-ambulance, non-old ego car is collision-close to another entity.

Compared with `easy`, the medium stop rule therefore adds:

- role-dependent exceptions,
- ambulance-related yielding,
- bus-pedestrian interaction,
- richer spatial conditions.

### Hard-lite safety rule

In the `hard` difficulty, the stop condition is extended again. The ego car must stop if there exists another entity such that at least one of the following holds:

1. a non-ambulance, non-old ego car is at an intersection while another entity is in the intersection,
2. a non-ambulance, non-old ego car is at an intersection while another entity at the intersection has higher priority,
3. a non-ambulance, non-old ego car is in an intersection while another entity in the intersection is an ambulance,
4. a non-ambulance, non-police ego car that is a car is not at or in an intersection, while a police vehicle is to its left and close to it,
5. a bus ego vehicle is not at or in an intersection and a pedestrian is both to its right and next to it,
6. an ambulance ego vehicle has an old entity to its right,
7. a non-ambulance, non-old ego car is collision-close to another entity.

Thus, the hard-lite setting adds police-related interaction logic on top of the medium conditions.

### Failure and episode termination

At the environment level, an episode terminates if:

1. the agent reaches its goal,
2. the horizon is exceeded,
3. the task rule is violated.

This yields three outcome types:

- **success**: goal reached,
- **failure**: task-rule violation,
- **timeout**: horizon exceeded before success.

This distinction is important for the thesis because high safety alone does not imply high task performance: an agent may avoid failure but still time out frequently.

## 3.5 Evaluation Metrics

Our primary task metric is **Trajectory Success Rate (TSR)**, defined as the fraction of episodes in which the agent reaches its goal successfully. We focus on TSR because it captures the actual objective of SPF: safe completion of the route.

To understand *why* a method succeeds or fails, we also evaluate:

- **failure rate**, the fraction of episodes ending in task-rule failure,
- **timeout rate**, the fraction of episodes ending due to horizon exhaustion,
- **mean reward**, the average episode return,
- **mean episode length**, the average number of steps taken per episode.

These metrics are complementary. In particular:

- failure rate measures unsafe behavior,
- timeout rate measures over-conservative or ineffective behavior,
- TSR measures the final balance between safety and progress.

For the sample-efficiency question, we additionally evaluate:

- the number of training timesteps required to reach strong TSR,
- the shape of the validation learning curves over time,
- the trajectory of failures and timeouts during training.

For shielded methods, we also log shield-specific diagnostics such as intervention rate and estimated policy safety. However, these are explanatory metrics rather than part of the task definition itself.

## 3.6 Single-Agent and Multi-Agent Task Variants

We consider two task variants.

### Single-agent SPF

The main setting is the single-agent task, in which one RL-controlled ego vehicle interacts with expert-controlled background traffic. This is the core setting used to answer our two main research questions about:

- whether probabilistic logic shields improve TSR,
- whether they improve training efficiency relative to PPO.

It is also the cleanest setting in which to analyze safety failures, timeout collapse, and the effect of different shield designs.

### Shared-policy multi-agent SPF

We also consider a multi-agent extension in which multiple RL-controlled cars act simultaneously while sharing one policy. Each RL-controlled agent receives its own local grounded observation and produces its own action independently through the shared policy. Background non-RL agents remain expert-controlled.

In the current implementation, we begin with an `easy` shared-policy setup and use:

- 2 RL cars first,
- then 3 RL cars if time permits.

For this setting, we distinguish between **per-agent success** and **joint success**. Joint success holds only if all RL-controlled agents reach their goals before any one of them fails and before the global horizon is exceeded. Joint failure occurs if any RL agent violates a task rule.

This extension is important because it lets us study whether symbolic shielding remains useful not only for individual safe navigation, but also for coordination in a shared traffic environment.
