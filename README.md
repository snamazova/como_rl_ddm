# como_rl_ddm

Group project for "Cognitive Modelling: Generative Models of Behaviour".

Implements a Reinforcement Learning Drift Diffusion Model (RLDDM) for a
Probabilistic Reversal Learning Task (PRLT).

## Files

| File | Description |
|---|---|
| `create_task_environment.py` | Generates the PRLT reward timeline (binary rewards, fixed reversal points) and converts it to an outcome matrix. |
| `rlddm.py` | Core RLDDM model: RL value learning (Rescorla-Wagner), DDM sampling (via `ssms`), analytic DDM log-density (Navarro & Fuss, 2009), likelihood, priors, posterior, and diagnostic plots. |
| `simulator.py` | Multi-subject simulation, PRLT behavioural metrics (accuracy, post-reversal recovery, perseveration, win-stay/lose-shift), and group comparison plots. |
| `parameter_recovery.py` | Parameter recovery: simulate data with known parameters, fit the model, and check recovery. |
| `parameter_sweep.py` | Systematically varies one parameter at a time (alpha, a, v_scale) and shows the effect on PRLT behaviour. |
| `model_comparison.py` | Compares the full RLDDM against a DDM-only model (no learning) and an RL-only model (no RT) via BIC. |

## Dependencies

- `numpy`, `scipy`, `pandas`, `matplotlib`
- `ssms` (for DDM sampling via `full_ddm`)

## Quick start

```bash
# Simulate one subject on the PRLT
python rlddm.py

# Simulate two groups and compare behaviour
python simulator.py

# Parameter recovery
python parameter_recovery.py

# Parameter sweep (alpha, a, v_scale)
python parameter_sweep.py

# Model comparison (full RLDDM vs DDM-only vs RL-only)
python model_comparison.py
```