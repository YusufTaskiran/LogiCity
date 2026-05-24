# PLPG for LogiCity Safe Path Following

## What PLPG Is

PLPG stands for **Probabilistic Logic Policy Gradient**.

It is a policy-gradient reinforcement learning method in which the neural policy is not used directly. Instead, the policy is passed through a **probabilistic logic shield** that reweights actions according to their estimated safety.

The key idea is:

- the neural policy proposes action probabilities
- a logic program estimates how safe each action is in the current state
- the original policy is transformed into a **shielded policy**
- the agent acts using the shielded policy
- the agent also learns through the shielded policy

So PLPG is not just action filtering. It changes both:

- the distribution used for acting
- the gradient used for learning

## Shielded Policy

PLPG defines the shielded policy as:

```text
π⁺(a|s) = [ P(safe | s, a) / Pπ(safe | s) ] π(a|s)
```

where:

- `π(a|s)` is the base neural policy
- `P(safe | s, a)` is the probability that action `a` is safe in state `s`
- `Pπ(safe | s)` is the policy-level safety probability under `π`
- `π⁺(a|s)` is the shielded policy

Interpretation:

- actions safer than the policy average get upweighted
- actions less safe than the policy average get downweighted
- the resulting distribution is renormalized

## ProbLog Safety Model

PLPG uses a **ProbLog** program to compute safety probabilities.

The program has three parts.

### 1. Policy Facts

The current policy is injected as an annotated disjunction:

```prolog
0.1::act(nothing);
0.5::act(accel);
0.1::act(brake);
0.1::act(left);
0.2::act(right).
```

This represents the current action distribution produced by the neural policy.

### 2. State Abstraction

The state is converted into probabilistic facts that contain only safety-relevant information:

```prolog
0.8::obstc(front).
0.2::obstc(left).
0.5::obstc(right).
```

These facts need not represent the full MDP state. They only need to encode what the safety logic cares about.

### 3. Safety Rules

Safety is defined logically, for example:

```prolog
crash :- obstc(front), act(accel).
safe :- not(crash).
```

From the same ProbLog program, one can query:

- `P(safe | s, a)`
- `Pπ(safe | s)`
- the shielded action probabilities

## Inference Efficiency and Knowledge Compilation

An important practical point in PLPG is that the probabilistic logic program does not need to be solved from scratch in a naive way at every step.

Probabilistic logic programs can be **compiled into circuits** using knowledge compilation, in the same spirit as compiled Bayesian-network inference.

The relevance of this is:

- exact probabilistic inference is hard in general
- but once the program has been compiled into a circuit, inference becomes linear in the size of the compiled circuit
- the same compiled object can also be used for gradient computation
- the safety model only needs to be compiled once, provided the program structure stays fixed

So the intended PLPG workflow is not:

- rebuild and fully re-solve a large symbolic program from scratch each time

but rather:

- define a fixed probabilistic logic program template
- compile it once
- reuse the compiled circuit for repeated action-safety queries and gradients

This efficiency argument is one of the reasons PLPG is practically interesting rather than only theoretically appealing.

Two constraints are important:

- the action space is assumed to be discrete
- the logic program structure should remain fixed while only the fact probabilities change from state to state

That fits our current Safe Path Following setup well, because:

- our action space is discrete: `Slow`, `Normal`, `Fast`, `Stop`
- the rule structure is fixed
- only the state-dependent predicate values change

## Learning Objective

PLPG replaces the normal policy-gradient update through `π` with a gradient through the shielded policy `π⁺`.

Instead of learning from:

```text
∇ log π(a|s)
```

it learns from:

```text
∇ log π⁺(a|s)
```

The full objective also adds a safety term:

```text
E_{π⁺θ} [ Σ_t Ψ_t ∇θ log π⁺θ(a_t|s_t) - α ∇θ log P_{π⁺θ}(safe | s_t) ]
```

This has two parts:

- reward term: increase shielded actions that lead to high return
- safety term: penalize policies whose overall safety probability is low

`α` controls the reward-safety tradeoff.

## PLPG Training Loop

At each decision step:

1. Observe state `s`
2. Compute the base policy `π(a|s)`
3. Build the ProbLog state abstraction
4. Insert the policy and state facts into the ProbLog program
5. Query action-level safety probabilities
6. Construct the shielded policy `π⁺(a|s)`
7. Sample the environment action from `π⁺`
8. Receive reward and next state
9. Update parameters using:
   - the policy-gradient term through `π⁺`
   - the safety-gradient term

The important property is:

- the acting policy and the learning policy are the same shielded policy

This avoids the policy mismatch that appears in rejection-based shields.

## Difference from Rejection-Based Shielding

Rejection-based shielding typically works like:

1. sample from `π`
2. test safety
3. reject unsafe action
4. resample or replace action

That creates a mismatch:

- learning assumes actions came from `π`
- the environment actually receives actions from another distribution

PLPG avoids this by directly constructing `π⁺` and training with `π⁺`.

## Mapping PLPG to LogiCity Safe Path Following

PLPG is a good conceptual fit for LogiCity Safe Path Following.

### Policy

The base policy is the RL policy over the 4 macro-actions:

- `Slow`
- `Normal`
- `Fast`
- `Stop`

### Safety Abstraction

In LogiCity, the safety-relevant abstraction already exists in logical form through predicates such as:

- `IsAtInter`
- `IsInInter`
- `HigherPri`
- `CollidingClose`
- `IsAmbulance`
- `IsBus`
- `IsPolice`

For our setting, these facts are effectively **binary** rather than probabilistic:

- `0` = false
- `1` = true

So compared with the original PLPG setting:

- the sensor probabilities collapse to deterministic facts
- the logic shield remains probabilistic because the policy itself is probabilistic

### Safety Rules

The task-rule logic already defines what counts as unsafe.

For example in `easy`:

- entering an intersection when another entity is already there
- violating intersection priority
- taking an action that leads to `CollidingClose`

These task rules can be re-expressed as ProbLog `unsafe` conditions or as action-conditional safety queries.

### Shielded Action Distribution

For each state:

1. compute the neural action probabilities over `Slow/Normal/Fast/Stop`
2. evaluate `P(safe | s, a)` for each action using the logic shield
3. construct `π⁺`
4. sample from `π⁺`

### Reward and Safety Learning

For Safe Path Following, the reward side can remain:

- goal reward
- progress reward
- failure penalty
- timeout penalty

PLPG would add a safety-aware optimization term on top of that.

## What Simplifies in Our Setting

Compared with the general PLPG setup, LogiCity SPF is simpler in several ways:

- the action space is small: 4 macro-actions
- the state abstraction is already logic-oriented
- the safety rules are already explicit
- the sensor facts are binary rather than uncertain

This means the main implementation burden is not state abstraction. It is:

- building a differentiable or gradient-compatible safety computation around the existing logic rules
- integrating the shielded policy into the PPO-style action-selection and update pipeline

## Feasibility in LogiCity

The efficiency argument from PLPG is feasible in principle for LogiCity, but with an important distinction between **conceptual feasibility** and **engineering feasibility**.

### Why It Looks Feasible

Several properties of LogiCity match the assumptions under which PLPG is attractive:

- the action space is discrete and small
- the rule set is fixed by difficulty
- the state abstraction is already symbolic
- the safety predicates are already explicit

This means there is a plausible path to:

- define a fixed PLP template per task difficulty
- feed current state facts into that template
- query `P(safe | s, a)` for each action
- construct the shielded policy from those values

### What Makes It Non-Trivial

The main uncertainty is not whether a logic safety model can be written. It can. The harder part is whether the full PLPG computation can be made efficient enough inside the existing training loop.

In this repo, the current logic stack is based on:

- deterministic symbolic predicates
- Z3-style rule checking

PLPG, by contrast, assumes:

- a probabilistic logic program
- reusable compiled inference objects
- gradient-compatible safety queries

So the feasibility question is really:

- do we wrap the existing logic into a ProbLog-based safety layer
- or do we approximate the PLPG shield using the current rule engine

### Practical Assessment

For `easy-lite`, I would judge PLPG as:

- **conceptually feasible:** yes
- **implementable as a research prototype:** yes
- **drop-in with the current codebase:** no

The likely effort is moderate rather than trivial, because it requires a new safety-inference layer rather than only a PPO config change.

### Most Realistic Implementation Strategy

The most feasible path is:

1. start with `easy-lite`
2. build a fixed ProbLog safety template for the 4 actions
3. use binary state facts from the current predicates
4. compute action-safety values outside the existing Z3 planner
5. reweight the policy into a shielded discrete distribution
6. add the safety term after the shielded-action path works

That is much more feasible than trying to replace the whole planner stack at once.

## Practical Interpretation for This Repo

For this project, PLPG can be viewed as:

```text
policy gradient for LogiCity
+ probabilistic logic safety model
+ policy reweighting by action safety
+ safety regularization during learning
```

In thesis language:

> The RL car does not act directly from its neural policy. Instead, the policy is transformed by a logic-based safety layer that increases the probability of safe actions and decreases the probability of unsafe actions. Training then proceeds through this shielded policy, with an additional safety-driven learning signal.

## Conclusion

Yes, PLPG is applicable to LogiCity Safe Path Following in principle.

The strongest reasons are:

- LogiCity already has symbolic safety predicates
- LogiCity already has explicit task rules
- the action space is small
- the SPF task is exactly the kind of safety-constrained sequential control problem that PLPG targets

The main open question is not conceptual fit. It is implementation detail:

- how to compute `P(safe | s, a)` from the current Logicity rule machinery in a way that can be used inside policy-gradient training

## Suggested Next Step

If we implement PLPG in this repo, the next design task should be:

1. define the exact ProbLog safety program for `easy`
2. define how `P(safe | s, a)` is computed for the 4 SPF actions
3. decide whether to:
   - replace PPO with a PLPG-specific policy-gradient loop
   - or build a PPO-compatible shielded action distribution and safety loss
