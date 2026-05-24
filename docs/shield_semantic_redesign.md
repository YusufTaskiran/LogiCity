# Shield Semantic Redesign

## Why The Redesign Is Needed

The first PLPG shield design for LogiCity Safe Path Following used a semantics close to:

- movement becomes unsafe when hazard is present
- `Stop` remains safe unless explicitly made unsafe

This worked well in the perfect-sensor setting, but under soft sensor uncertainty it produced a strong failure mode:

- uncertainty reduced the safety of movement actions
- `Stop` remained near-maximally safe
- the shielded policy collapsed to repeated `Stop`
- evaluation produced `timeout_rate = 1.0` with `failure_rate = 0.0`

So the problem was not that PLPG could not execute safely under uncertainty. The problem was that the shield semantics were too asymmetric.

## Problem With The Old Semantics

The old query effectively asked:

```text
P(safe | action)
```

where “safe” was interpreted mainly as:

- not violating the stop rule
- not taking an unsafe move under hazard

In that formulation, `Stop` became too strong under uncertainty because:

- mild uncertainty penalized movement
- but did not penalize unnecessary stopping

As a result, the shield behaved more like:

```text
always stop unless completely certain movement is safe
```

That is too conservative for Safe Path Following.

## New Design Principle

The redesigned shield should estimate:

```text
P(appropriate | action)
```

instead of only:

```text
P(safe | action)
```

The goal is to model:

- when stopping is necessary
- when movement is appropriate
- safe progress rather than pure movement suppression

In this redesign:

- `Stop` should dominate when the probability of needing to stop is high
- movement should dominate when the probability of needing to stop is low

So the shield becomes a model of **appropriate safe action selection**, not only a hazard-avoidance filter.

## Easy-Difficulty Redesign

For `easy`, the redesign keeps the task’s stopping conditions but changes the queried semantics.

### Stop-Need Predicates

We retain the existing stopping logic:

```prolog
need_to_stop :- is_at_inter_ego, is_in_inter_i.
need_to_stop :- is_at_inter_ego, higher_pri_i.
need_to_stop :- colliding_close_i.
```

### Move-Permissibility Predicate

We then define:

```prolog
okay_to_move :- not(need_to_stop).
```

### Action Appropriateness

The key semantic change is:

```prolog
appropriate :- act(stop), need_to_stop.
appropriate :- act(slow), okay_to_move.
appropriate :- act(normal), okay_to_move.
appropriate :- act(fast), okay_to_move.
```

The shield then evaluates:

```text
P(appropriate | act(a))
```

for each action `a`.

## Why This Is Better

Under this semantics:

- if `P(need_to_stop)` is high, `Stop` becomes appropriate
- if `P(need_to_stop)` is low, movement becomes appropriate
- `Stop` no longer remains universally dominant under mild uncertainty

This preserves the safety logic from the task while avoiding the pathological:

```text
any uncertainty -> always stop
```

behavior observed with the previous soft-probability shield.

## Important Terminology

In this redesign:

- `Stop` is not treated as “unsafe” when stopping is unnecessary
- instead, it is treated as **inappropriate for safe progress**

This is the cleaner scientific framing for Safe Path Following.

## Scope

This redesign is first implemented for:

- `easy`

and should be validated there before extending the same semantic shift to:

- `medium`
- `hard`
