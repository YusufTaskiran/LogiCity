Experiment 1 clean restart with a 4-car compact roster.

Design choice:
- 4 total cars on the map
- 2 pedestrians retained
- K in {1, 2, 3}

Rationale:
- K=3 now means the RL-controlled population covers most of the car traffic
- one background car remains, so the benchmark is still mixed-population rather than fully RL-only
- this should expose stronger RL-RL interaction pressure than the previous 6-car roster

Important:
- these configs are a clean branch and do not modify the earlier dense experiment setup
- validation/test episodes should be regenerated for this roster

