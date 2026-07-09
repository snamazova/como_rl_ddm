"""PRLT RLDDM simulator (placeholder).

Conceptual plan for this file:
------------------------------
1. Generate a PRLT reward timeline using create_task_environment.generate_timeline()
   and convert it to an (n_trials, 2) outcome matrix with timeline_to_matrix().

2. Simulate one or many synthetic participants with rlddm.rlddm_simulate(),
   using a parameter dictionary that controls learning (alpha), drift scaling
   (v_intercept, v_scale), and DDM parameters (a, w, t0, [sv, sw, st0]).

3. Optionally sweep across parameter grids or generate a counterbalanced set
   of timelines (e.g., half with reversed_state=True, half False).

4. Save the synthetic datasets (choices, RTs, outcomes, true parameters) to CSV
   so they can be used for parameter-recovery experiments later.

5. Provide a small CLI so the group can run:
       python simulator.py --n-trials 140 --n-subjects 10 --out-dir data/

This file is intentionally empty for now; the core model is already implemented
in rlddm.py and the task generator is in create_task_environment.py.
"""
