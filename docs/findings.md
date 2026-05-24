# Findings

## PLPG Can Keep Execution Safe While The Base Policy Learns Slowly

In the current LogiCity SPF PLPG setup, the shield can enforce safe executed behavior even when the underlying base policy does not yet internalize the safety behavior strongly.

This was observed in the `easy_lite_plpg_seed101` run:

- `failure_rate` stayed at `0.0`
- `decision_succ_action_stop` stayed at `1.0`
- `mean_shielded_policy_safe_prob` stayed near `1.0`
- but `mean_base_policy_safe_prob` stayed much lower, around `0.68`
- the base policy chose `Stop` essentially never
- the shielded policy inserted many `Stop` actions

So the shield was doing the main corrective work during execution.

### Why the base policy does not learn as fast

The main reason is that the current PLPG instantiation is very strong and very certain:

- state facts are binary
- action-safety queries are conditioned on one-hot actions
- task rules are deterministic hard stop rules

Together, this often makes `P(safe | s, a)` effectively deterministic.

In many hazard states, the safety vector is close to:

```text
[0, 0, 0, 1]
```

which means:

- `Slow`, `Normal`, and `Fast` are unsafe
- `Stop` is safe

After reweighting and renormalization, the shielded policy can therefore become effectively forced to `Stop`.

### Why the safety term does not fully fix this

The current implementation uses a safety loss based on the **shielded policy safety**:

```text
safety_loss = -log P_{pi+}(safe | s)
```

with a weighted objective of the form:

```text
ppo_loss + alpha * safety_loss
```

where `alpha` is currently configured as `0.1`.

Because the shield is already very effective, the shielded policy safety quickly becomes almost perfect:

```text
P_{pi+}(safe | s) ~= 1
```

Once that happens:

- `-log P_{pi+}(safe | s)` becomes very small
- the safety-loss gradient becomes weak
- the base policy gets little direct pressure to become safe on its own

So the shield solves the execution problem before the raw policy has to fully absorb the safety behavior.

### Interpretation

This is not a contradiction of PLPG. It is a property of the current deterministic LogiCity setup.

The original PLPG paper emphasizes a more probabilistic regime where:

- state abstractions can be uncertain
- unsafe actions may still retain some probability mass
- the safety term continues to exert meaningful pressure

In the current LogiCity SPF implementation:

- the rules should remain deterministic
- the current facts are also deterministic
- therefore the shield behaves much closer to a hard corrective shield in many states

### Thesis-useful conclusion

For the current setup, PLPG provides very strong safety at execution time, but the base policy may learn the safe action structure more slowly than expected because:

- the shield corrects unsafe behavior before failures occur
- the shielded-policy safety becomes nearly perfect early
- the current safety-loss formulation then becomes weak

This suggests an important methodological point for later experiments:

- compare base-policy safety against shielded-policy safety
- monitor shield intervention rate over time
- consider evaluating alternative safety-loss sources, such as using base-policy safety instead of shielded-policy safety

## PLPG Is Optimizing The Shielded Policy, Not Necessarily A Safe Base Policy

The PLPG paper explicitly notes that the base policy learned under PLPG is generally **not expected to match** the policy that would have been learned without a shield.

The important implication is:

- PLPG learns parameters so that the **shielded policy** `pi+` performs well under the imposed safety constraints
- it does **not** guarantee that the raw base policy `pi` itself becomes the same kind of safe policy that unshielded learning would produce

This matters directly for the current LogiCity SPF results.

In the current deterministic setup:

- the shielded policy can be near-perfectly safe
- the base policy can remain substantially unsafe
- the gap between `pi` and `pi+` can stay large over training

So the current results should not be interpreted as:

- “PLPG failed because the base policy did not become safe on its own”

Instead, they should be interpreted as:

- PLPG successfully optimized the shielded policy
- but in the current binary deterministic regime, the shield may do much more of the safety work than the base policy

### Thesis-useful interpretation

A useful thesis conclusion is:

> In deterministic symbolic Safe Path Following, PLPG may function primarily as an execution-time safety mechanism that optimizes a safe shielded policy, while the underlying base policy internalizes the safety structure only weakly.

This is consistent with the paper, but it is especially pronounced in the current LogiCity setup because:

- facts are binary
- safety rules are deterministic
- action-safety queries are effectively hard `0/1`
- the shield can therefore behave close to a hard mask

## Soft Sensor Probabilities Can Make The Current Shield Collapse To Timeout-Dominant Conservatism

When soft symmetric sensor noise was introduced into the shield facts for the `easy` task, the current PLPG formulation became extremely conservative.

Observed pattern:

- `eps = 0.00` reached `TSR = 1.0` quickly
- `eps = 0.05`, `0.10`, and `0.20` all collapsed to:
  - `TSR = 0.0`
  - `failure_rate = 0.0`
  - `timeout_rate = 1.0`
  - almost all executed actions becoming `Stop`
  - `shield_intervention_rate = 1.0`

So the method did not fail by producing unsafe behavior. Instead, it failed by becoming too conservative to make progress.

### Interpretation

This suggests that the current soft-noise shield semantics are making `Stop` dominate too easily under uncertainty.

In the current safety program:

- `Stop` remains safe unless explicitly made unsafe
- moving actions become less safe whenever hazard has nonzero probability

With soft fact probabilities, hazard may rarely become exactly zero. As a result:

- moving actions are always somewhat penalized
- `Stop` remains the safest action
- deterministic evaluation collapses to repeated stopping

### Important implication

This result should not be interpreted simply as:

- “PLPG is not robust to noisy sensors”

More precisely, it indicates:

- the current **soft-probability shield formulation** is overly conservative in this domain

So this is a finding about the interaction between:

- the current safety program structure
- soft fact probabilities
- deterministic action selection from the shielded policy

### Methodological consequence

Before scaling soft-noise experiments further, the current soft-noise setup should be reconsidered.

Reasonable next alternatives are:

- test bit-flip noise first
- redesign the soft-noise safety semantics so `Stop` is not trivially dominant under any uncertainty
- compare how this failure mode differs from the perfect-sensor regime

## Redesigning The Easy Shield Semantics Restored Useful Behavior Under Soft Uncertainty

After the `easy` shield semantics were redesigned from:

- pure movement-safety filtering

to:

- action appropriateness for safe progress

the previous soft-noise collapse disappeared at `eps = 0.05`.

Observed pattern in the redesigned `easy_lite_plpg_soft005_seed101_retry` run:

- `TSR = 1.0` at `2500` steps
- `failure_rate = 0.0`
- `timeout_rate = 0.0`
- movement and stopping were both used
- the shield still intervened heavily
- the agent no longer collapsed to universal stopping

This strongly suggests that the earlier soft-noise failure was caused primarily by the **old shield semantics**, not by soft uncertainty itself.

### Interpretation

The redesign changed the shield from approximating:

```text
P(safe | action)
```

to a semantics closer to:

```text
P(appropriate safe action | action)
```

This made a critical difference:

- `Stop` no longer remained trivially dominant under mild uncertainty
- movement remained preferable when stopping was not needed
- the shield could still enforce stopping when stop-need probability was high

### Methodological consequence

This is a strong indication that:

- uncertainty-aware PLPG in LogiCity is feasible
- but the queried shield property must reflect **appropriate safe progress**, not only **movement unsafety**

So for later noisy-sensor experiments, the redesigned shield semantics should be preferred over the original soft-probability formulation.
