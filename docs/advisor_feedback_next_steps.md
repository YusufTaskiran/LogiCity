# Advisor Feedback: Next Implementation Phase

Date captured: 2026-04-29

## Current Position

- The current PLS implementation is considered good enough to keep as the main shielded method.
- DLS can be dropped from the thesis narrative.
- The thesis comparison should now focus on:
  - PPO
  - PPO + PLS

## Main Feedback

### 1. Remove DLS from the thesis focus

The thesis should no longer emphasize DLS as a main method or comparison track.

Implication:
- experimental design should focus on PPO vs PLS
- documentation and thesis framing should be updated accordingly

### 2. Add a joint PLS shield

Current status:
- the implemented PLS is decentralized
- each agent is shielded individually from its own local observation

Requested extension:
- implement a joint PLS shield in addition to the decentralized one

Goal:
- compare decentralized PLS against joint PLS
- show where joint safety reasoning is stronger than per-agent safety reasoning

### 3. Weaken the observation space

Advisor concern:
- the current observation is too strong
- especially:
  - `CollidingClose`
  - `IsCloseAhead`
- these predicates already encode nontrivial logic and make the local observation too informative

Requested direction:
- weaken the observation so that it contains less pre-computed safety structure

Reason:
- this should make the limitations of decentralized shielding clearer
- weakening local observation should reduce the amount of global information available per agent
- this creates room for a joint shield to show its benefit

### 4. Increase time horizon / reduce overtime pressure

Advisor concern:
- PLS currently incurs too much overtime
- the current setup may overemphasize time efficiency rather than safety

Requested direction:
- increase the allowed time horizon
- make safety the main focus rather than overtime behavior

## Intended Research Shift

The next version of the work should move toward:

- weaker local observations
- decentralized PLS as the baseline shielded method
- joint PLS as the stronger collective safety method
- evaluation focused more clearly on safety rather than overtime effects

## Implementation Plan

### A. Scope / cleanup

- remove DLS from the active thesis experiment story
- keep DLS code only if needed for archival/reference, but stop treating it as a core thesis method
- update docs and experiment matrix to focus on PPO vs PLS

### B. Observation redesign

Review the current observation ontology and grounding.

Current concern:
- `CollidingClose`
- `IsCloseAhead`

Potential direction:
- remove these predicates from the active observation ontology
- replace them with weaker, more primitive predicates if needed
- make sure the new observation still supports training, but with less baked-in logic

Key requirement:
- the new observation should be weak enough that decentralized agents lose some useful global safety information

### C. Decentralized PLS baseline

Keep the current per-agent PLS as the decentralized baseline.

Tasks:
- adapt it to the weaker observation
- revalidate that the decentralized shield still works correctly
- retrain PPO + decentralized PLS under the weakened observation

### D. Joint PLS implementation

Add a joint shield variant where safety is computed jointly across agents rather than independently per agent.

Questions to resolve during implementation:
- what is the joint state abstraction
- what policy inputs are needed for joint shielding
- whether the joint shield reweights each agent independently from a joint safety model, or computes a joint action safety object first
- how joint ProbLog facts and queries should be structured

### E. Timing / environment setup

Adjust the environment configuration so that overtime is less dominant.

Possible directions:
- increase the maximum number of steps / allowed time
- reduce overtime pressure if needed

Main intent:
- safety comparison should be primary
- unnecessary punishment from short horizons should not dominate results

## Concrete Next Steps

1. Decide the weakened observation ontology.
2. Identify which current predicates should be removed or replaced.
3. Adapt the decentralized PLS shield to the weakened observation.
4. Design the joint PLS state abstraction and ProbLog program.
5. Implement joint PLS.
6. Update training/eval configs with the revised time horizon.
7. Regenerate datasets or annotations if the new setup requires it.
8. Retrain PPO and PLS under the new setup.

## Notes

- The core PLS path already works and is a good foundation.
- The next phase is not about proving DLS vs PLS anymore.
- The main new scientific story is:
  - local/decentralized shielding under weaker observation
  - versus joint shielding under the same weaker observation

