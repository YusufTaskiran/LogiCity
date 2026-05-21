# LogiCity Simulator Draft

## 1. Introduction

This thesis uses the LogiCity simulator as the experimental platform for studying reinforcement learning with symbolic safety shielding in multi-agent urban navigation. The choice of simulator is not incidental. LogiCity is particularly well suited to the thesis because it combines:

- explicit symbolic rule grounding
- multi-agent interaction
- interpretable grid-based traffic geometry
- reproducible scenario generation
- direct support for expert-rule baselines and shielded RL

These properties make it a strong match for a thesis whose goal is not only to compare learning curves, but also to analyze why a policy succeeds or fails under structured safety constraints.

## 2. What LogiCity Provides

### 2.1 Structured Urban World

LogiCity constructs a city from:

- road layers
- building layers
- agent layers

using an explicit spatial representation in `logicity/core/city.py`. The map is assembled from streets and buildings, and intersections are derived programmatically rather than being treated as black-box visual regions.

This is important for the thesis because it enables formal definitions such as:

- being at an intersection
- being inside an intersection
- entering the same labeled intersection block
- sharing an intersection with another entity

These are exactly the kinds of structured concepts required for symbolic safety reasoning.

### 2.2 Labeled Intersection Geometry

A major advantage of LogiCity is that intersection geometry is not implicit. The simulator computes an `intersection_matrix` with separate semantics for:

- car entry lines
- pedestrian entry lines
- interior intersection blocks

The relevant construction appears in [city.py](/d:/Dev/LogicityFresh/LogiCity/logicity/core/city.py:126).

This makes the simulator especially appropriate for the thesis, because the safety questions being studied are intersection-centric:

- when should the ego vehicle stop at an intersection?
- when is another entity already inside the intersection?
- when is a simultaneous entry event a failure?

In many generic simulators, these questions are difficult to operationalize cleanly. In LogiCity, they are first-class constructs.

### 2.3 Multi-Agent Dynamics

LogiCity is inherently multi-agent. At each world update:

- all agents obtain local decisions from the local planner
- all agents then move in the shared world
- the world state is updated jointly

This update structure is implemented in [city.py](/d:/Dev/LogicityFresh/LogiCity/logicity/core/city.py:30) and extended for RL in [city_env.py](/d:/Dev/LogicityFresh/LogiCity/logicity/core/city_env.py:10).

This is ideal for the thesis because the central problem is not isolated motion planning, but interactive safety:

- ego must respond to cars
- ego must respond to pedestrians
- failures depend on coordinated occupancy and motion

The simulator therefore exposes the learning agent to the kind of coupled dynamics that symbolic shielding is meant to help with.

## 3. Symbolic Reasoning Compatibility

### 3.1 Predicate-Level World Access

LogiCity is built around explicit symbolic predicates such as:

- `IsAtInter`
- `IsInInter`
- `HigherPri`
- `IsAhead`
- `IsCloseAhead`
- `CollidingClose`

These predicates are grounded over the current local world and exposed to planners and RL wrappers.

This is a crucial reason the simulator is ideal for the thesis. The thesis investigates probabilistic logic shielding, so the environment must support:

- extracting symbolic facts from the world
- evaluating logical rules over those facts
- comparing symbolic expert behavior with learned policy behavior

LogiCity provides exactly that interface.

### 3.2 Native Expert Rule Support

The simulator already supports rule-driven local planners through:

- `Z3_Expert`

and loads them through [CityLoader](/d:/Dev/LogicityFresh/LogiCity/logicity/utils/load.py:14).

This means the same simulator can provide:

- background rule-based agents
- expert demonstrations or expert rollout filtering
- normative stop requirements for evaluation

That is ideal for this thesis because it avoids the usual mismatch where:

- one system generates data
- another unrelated system defines the rules
- and a third system evaluates the agent

In LogiCity, these pieces live in a single coherent formal environment.

### 3.3 Direct RL Integration

The simulator also includes a dedicated RL environment layer:

- [CityEnv](/d:/Dev/LogicityFresh/LogiCity/logicity/core/city_env.py:10)
- [GymCityWrapper](/d:/Dev/LogicityFresh/LogiCity/logicity/utils/gym_wrapper.py:16)

These bridge the symbolic world and modern RL training by:

- flattening symbolic observations
- exposing discrete action spaces
- computing reward and failure outcomes
- keeping expert actions available for diagnostics

This direct coupling between symbolic planning and RL is precisely what the thesis requires.

## 4. Why LogiCity Is Better Than a Generic RL Simulator for This Thesis

### 4.1 Interpretability

A generic simulator with continuous state vectors or image-only input would make it much harder to analyze:

- why the expert stopped
- why the shield intervened
- why a failure occurred

LogiCity makes these explanations inspectable. The current setup can log:

- shield facts
- expert stop witnesses
- grounded predicate dictionaries
- action histograms
- per-failure-type counts

This is a major methodological advantage. The thesis is not only about performance, but also about understanding the mechanism of improvement or failure.

### 4.2 Formal Safety Semantics

The simulator supports safety definitions that are formal rather than heuristic. For example:

- rule-based stop-required failure
- deadzone failure
- simultaneous same-intersection entry failure

These are encoded in world- and predicate-level terms, not ad hoc human judgment. That makes the evaluation clearer, stronger, and easier to defend academically.

### 4.3 Controlled Abstraction

The thesis requires a middle ground:

- enough realism to make interaction meaningful
- enough abstraction to keep logic design and analysis tractable

LogiCity provides this balance well.

It is more structured and controllable than a photorealistic traffic simulator, which is helpful for:

- precise rule grounding
- dataset generation
- debugging symbolic-RL mismatch

At the same time, it is richer than a toy tabular intersection environment because it includes:

- multiple agents
- route following
- different priorities
- pedestrians
- spatial intersection structure
- forward proximity and deadzone behavior

This makes it an especially strong fit for a thesis on structured safe RL.

## 5. Why LogiCity Is Ideal for the Decentralized PLS Story

### 5.1 Supports Local Symbolic Observation Design

The thesis uses a decentralized 6-bit symbolic observation:

- `ego_in_inter`
- `other_in_inter`
- `ego_at_inter`
- `close_ahead`
- `ahead`
- `higher_pri`

This kind of observation is easy to construct in LogiCity because the simulator already grounds the underlying predicates locally. The RL wrapper can therefore aggregate them into a small decentralized observation without inventing arbitrary approximations.

This is important because the thesis argument depends on the difference between:

- full expert symbolic access
- compact decentralized symbolic access

LogiCity naturally supports both within one consistent framework.

### 5.2 Makes Mismatches Observable

One of the most valuable aspects of the recent debugging process is that LogiCity allowed mismatches to be identified precisely. For example, it was possible to observe that:

- the expert required `Stop`
- the shield saw only `ego_at_inter`
- the stop witness pointed to a pedestrian or priority interaction

This kind of diagnosis is only possible because the simulator exposes:

- predicate-level state
- expert rule witnesses
- local RL grounding

That is exactly the kind of transparency needed in a thesis that studies the boundary between symbolic structure and local information loss.

### 5.3 Supports Decentralized and Joint Extensions

The simulator is also well suited for later extensions beyond the decentralized setting. Because multiple agents coexist in the same formally structured world, the same platform can later support:

- decentralized independent local shields
- shared-policy multi-agent evaluation
- joint shields with additional coordination information

This makes LogiCity not only useful for the current chapter, but also a strong long-term platform for the broader thesis trajectory.

## 6. Why LogiCity Is Ideal for Evaluation

### 6.1 Offline Cached Episode Datasets

LogiCity supports cached episode generation through expert-driven rollouts and reusable episode files. This is highly valuable for evaluation because it ensures:

- PPO and PLS are tested on the same worlds
- train/val/test splits are reproducible
- controlled scenario families can be compared fairly

This is better than purely online random environment sampling when the goal is method comparison rather than only large-scale training throughput.

### 6.2 Visual and Quantitative Evaluation

The simulator supports both:

- quantitative metrics
- rollout visualization

This is ideal for the thesis because some failures, especially close-ahead and intersection coordination failures, are easier to understand when visualized. The ability to connect:

- TSR and failure counts
- with GIF or step-by-step world renderings

is a strong methodological asset.

### 6.3 Expert Comparison

Because expert actions are available during evaluation, LogiCity allows metrics such as:

- stop decision success rate
- matched stop counts
- expert-vs-policy action histograms

This provides a much richer evaluation than success rate alone, and it is especially appropriate for a thesis on safety shielding.

## 7. Practical Research Advantages

Beyond conceptual fit, LogiCity also offers practical advantages for thesis work:

- the simulator is lightweight compared to large external driving stacks
- rules and observations can be edited quickly
- bugs in symbolic grounding can be traced directly
- episode generators can be specialized for diagnostic cases
- training and evaluation use the same underlying world model

This makes iterative methodological refinement feasible, which is particularly important in research that combines RL, symbolic logic, and multi-agent safety.

## 8. Limitations and Why They Are Acceptable Here

LogiCity is not a photorealistic traffic simulator and does not model full real-world driving complexity. For example:

- the action space is discrete and coarse
- road geometry is grid-based
- perception is symbolic rather than raw sensory

However, these limitations are acceptable, and even beneficial, for this thesis. The goal is not high-fidelity autonomous driving benchmarking. The goal is to study:

- how symbolic safety structure interacts with RL
- how decentralized abstraction loses information
- how failures can be categorized and analyzed formally

For these goals, interpretability and formal controllability are more important than photorealism.

## 9. Conclusion

LogiCity is ideal for this thesis because it sits at the right point in the design space:

- more structured and interpretable than generic deep RL simulators
- richer and more interactive than toy symbolic planning environments
- directly compatible with both expert symbolic rules and RL training

Its key advantages for the thesis are:

- explicit intersection semantics
- multi-agent interaction
- symbolic predicate grounding
- native expert-rule support
- direct RL integration
- reproducible cached dataset generation
- strong debugging and visualization support

Taken together, these properties make LogiCity an especially appropriate platform for evaluating decentralized probabilistic logic shielding in urban multi-agent navigation.

