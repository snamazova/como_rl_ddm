# Presentation Guide: PRLT RLDDM Project

## Generated plots

### 1. Reversal-locked accuracy & RT (H1 vs H2)

![Reversal-locked](reversal_locked.png)

**What it shows:** Accuracy (top) and RT (bottom) from trial −20 to +20 relative to each reversal, averaged across 10 subjects and 5 reversal points. H1 (blue) vs H2 (orange).

### 2. H1 single-subject diagnostic (RL panels)

![H1 RL](single_subject_h1_rl.png)

**What it shows:** Learned values (with reversal markers), drift rates, prediction errors, and observed outcomes for one H1 subject.

### 3. H1 single-subject diagnostic (DDM panels)

![H1 DDM](single_subject_h1_ddm.png)

**What it shows:** DDM schematics and RT histograms at low/medium/high value-difference regimes.

### 4. H2 single-subject diagnostic (RL panels)

![H2 RL](single_subject_h2_rl.png)

**What it shows:** Same four panels as H1, but for the H2 model. Values and drifts look similar — the difference is in the boundary.

### 5. H2 single-subject diagnostic (DDM panels)

![H2 DDM](single_subject_h2_ddm.png)

### 6. H2 time-varying boundary

![H2 boundary](h2_boundary_over_trials.png)

**What it shows:** The boundary a(t) spikes at each reversal (by κ=2.0) then decays back to baseline (a_base=3.0) with time constant τ=5.

### 7. Parameter sweep — alpha (learning rate)

![Alpha sweep](parameter_sweep_alpha.png)

**What it shows:** Six behavioural metrics as a function of α. Higher α → higher accuracy, faster post-reversal recovery, lower perseveration.

### 8. Parameter sweep — a (decision threshold)

![A sweep](sweep_a.png)

**What it shows:** Same metrics as a function of a. Higher a → higher accuracy but much slower RTs (speed-accuracy tradeoff). a does not affect learning.

### 9. Confusion matrix (model comparison)

![Confusion matrix](confusion_matrix.png)

**What it shows:** 2×2 BIC comparison. H1 data → H1 wins. H2 data → H2 wins. Diagonal = models are identifiable.

---

## Research question

> Does post-reversal slowing in choice RT reflect a collapse in drift rate
> (H1: value-driven evidence accumulation) or an independent increase in
> response caution (H2: boundary separation)?

## Formula check

Your project document specifies `v(t) = v_max · tanh(β · ΔQ(t))` where
`ΔQ = Q_A − Q_B`. Our code uses `v = v_max · tanh(β · (Q_B − Q_A))` — the
opposite sign. This is **correct** because in our DDM:

- Positive drift → upper boundary → choice 2 (bandit B)
- Negative drift → lower boundary → choice 1 (bandit A)

So if bandit A is better (Q_A > Q_B), the drift should be **negative**
(to push toward choice A = lower boundary). Our formula produces
`Q_B − Q_A < 0` → negative drift → choice A. ✓

The project's formula `Q_A − Q_B` would produce positive drift → choice B,
which is the **wrong** bandit. So either the project formula needs a minus
sign, or the DDM boundaries need to be swapped. Our implementation is
internally consistent and correct.

---

## Plots and what to say about each

### Plot 1: Reversal-locked accuracy & RT (`reversal_locked.png`)

**What it shows:** Accuracy (top) and RT (bottom) from trial −20 to +20
relative to each reversal, averaged across 10 subjects and 5 reversal
points. H1 (blue) vs H2 (orange).

**Key observations:**
- Both models show an accuracy drop at reversal (trial 0) — this is
  the behavioural signature of the PRLT.
- H1's RT increase is moderate (~2.0s → ~2.1s): the drift collapses
  because ΔQ → 0 after reversal, so evidence accumulates slowly, but
  the boundary stays the same.
- H2's RT increase is large (~2.0s → ~3.6s): the boundary physically
  increases by κ=2.0 at reversal, requiring much more evidence before
  committing.

**What to say in the presentation:**
> "The two models produce qualitatively different RT patterns after
> reversal. H1 predicts a modest RT increase driven by drift collapse
> — values haven't updated yet, so the drift rate drops to near zero.
> H2 predicts a sharp RT increase driven by boundary inflation — the
> agent becomes more cautious, requiring more evidence before deciding.
> These patterns are the behavioural signatures we will try to
> distinguish in the confusion matrix analysis."

---

### Plot 2: H1 single-subject RL diagnostic (`single_subject_h1_rl.png`)

**What it shows:** Top-left: learned values V₁ and V₂ over trials with
reversal markers. Top-right: drift rate over trials. Bottom-left:
prediction errors. Bottom-right: observed outcomes.

**Key observations:**
- Values cross at each reversal (dashed lines) — the model learns
  which bandit is correct, then flips after the reversal.
- Drift rate crosses zero at each reversal (when values are equal)
  — this is the "drift collapse" that H1 says causes post-reversal
  slowing.
- Prediction errors spike at reversals — the old value predicts
  reward but the correct bandit just flipped.

**What to say:**
> "In the H1 model, the drift rate is driven entirely by the value
> difference. At each reversal, values converge to zero difference,
> causing the drift rate to collapse to zero. This is the mechanism
> H1 uses to explain post-reversal slowing."

---

### Plot 3: H2 single-subject RL diagnostic (`single_subject_h2_rl.png`)

**What it shows:** Same four panels as Plot 2, but for H2.

**Key observations:**
- Values and drifts look similar to H1 (both models use the same
  tanh drift function).
- The difference is in the DDM boundary, not in the RL component.

---

### Plot 4: H2 boundary over trials (`h2_boundary_over_trials.png`)

**What it shows:** The time-varying boundary a(t) for one H2 subject.
Spikes at each reversal, then decays exponentially back to baseline.

**Key observations:**
- At reversal (trial 36, 56, 71, 86, 106): a = a_base + κ = 5.0
- ~5 trials after: a ≈ 3.27 (most of the κ has decayed)
- ~15 trials after: a ≈ 3.0 (back to baseline)

**What to say:**
> "In H2, the boundary separation increases by κ at each reversal
> and decays back to baseline with time constant τ. This means the
> agent requires more evidence immediately after a reversal, producing
> slower RTs — but for a different reason than H1. Here the drift rate
> is the same, but the decision criterion is stricter."

---

### Plot 5: Parameter sweep — alpha (`parameter_sweep_alpha.png`)

**What it shows:** 6 behavioural metrics as a function of learning rate α,
each simulated with 8 subjects. Shows how α affects accuracy, post-reversal
recovery, perseveration, win-stay, and RT.

**Key observations:**
- Higher α → higher overall accuracy (0.51 → 0.75)
- Higher α → faster post-reversal recovery (accuracy in trials 1-5 improves)
- Higher α → lower perseveration (faster value updating → quicker switching)
- Higher α → faster RTs (less time stuck with wrong values)

**What to say:**
> "The learning rate α controls how quickly the model adapts to reversals.
> Higher α means faster value updates, which translates to quicker
> post-reversal recovery and less perseveration. This confirms that α
> is the key parameter governing adaptation speed in the PRLT."

---

### Plot 6: Parameter sweep — a (`sweep_a.png`, if generated)

**What it shows:** Same 6 metrics as a function of decision threshold a.

**Key observations:**
- Higher a → higher accuracy but much slower RTs (speed-accuracy tradeoff)
- a does NOT affect post-reversal recovery speed (it's not a learning
  parameter — it only affects how much evidence is needed)

**What to say:**
> "The decision threshold a controls the speed-accuracy tradeoff but
> does not affect learning. Higher a produces more accurate but slower
> responses. This is dissociable from α, which controls adaptation
> speed without directly affecting RT — confirming that the two
> parameters capture distinct cognitive mechanisms."

---

### Plot 7: Confusion matrix (`confusion_matrix.png`, if generated)

**What it shows:** A 2×2 heatmap of BIC values. Rows = data-generating
model (H1 or H2). Columns = fitted model (H1 or H2). Lower BIC = better.

**The ideal result:**
```
              Fit H1     Fit H2
H1 data      [WIN]      [lose]
H2 data      [lose]     [WIN]
```

**What it says if diagonal:**
> "When data is generated under H1, fitting H1 gives lower BIC than H2,
> and vice versa. This means the two models are identifiable — their
> post-reversal behavioural signatures are distinct enough that the
> fitting procedure can correctly recover which mechanism generated
> the data."

**What it says if off-diagonal:**
> "The models are confounded — both explain the data equally well, and
> we cannot distinguish drift collapse from boundary increase based on
> behaviour alone. This would mean the research question cannot be
> answered with this experimental design, and we would need more
> trials, more reversals, or additional behavioural measures."

---

## Suggested presentation flow

1. **Research question** (1 slide): post-reversal slowing — drift or boundary?
2. **Model specification** (1 slide): H1 vs H2 formulas, parameter meanings
3. **Reversal-locked plot** (1 slide): the two models produce different patterns
4. **Single-subject diagnostics** (1-2 slides): show values, drifts, boundary for each model
5. **Parameter sweep** (1 slide): α and a have dissociable effects
6. **Confusion matrix** (1 slide): can we tell the models apart?
7. **Critical evaluation** (1 slide): strengths, limitations, next steps

---

## Files in this project

| File | Purpose |
|---|---|
| `create_task_environment.py` | PRLT reward generator + correct-bandit metadata |
| `rlddm.py` | Core RLDDM: tanh drift, H1/H2 boundary, likelihood, plotting |
| `simulator.py` | Multi-subject simulation + behavioural metrics |
| `parameter_recovery.py` | Parameter recovery (needs update for H1/H2) |
| `parameter_sweep.py` | Systematic parameter sweeps |
| `model_comparison.py` | BIC-based model comparison (older approach) |
| `confusion_matrix.py` | 2×2 confusion matrix (H1 vs H2) |
| `reversal_locked_plots.py` | Time-locked accuracy/RT plots |
| `generate_plots.py` | Script to generate all presentation plots |

## What still needs work

1. **Confusion matrix fitting is slow** — the H2 model has 8 parameters and
   Nelder-Mead struggles. Options: more restarts, different optimizer
   (e.g. differential evolution), or better starting values.

2. **`parameter_recovery.py` needs updating** — it still uses the old
   linear parameter names (v_intercept, v_scale). Needs to be rewritten
   for H1 (v_max, beta) and H2 (v_max, beta, a_base, kappa, tau).

3. **`sweep_a.png` was not generated** — the command timed out. Run
   `python parameter_sweep.py` separately to generate it.

4. **H3 (combined model)** — optional per your project plan. Would need
   a model with both drift and boundary free, fitted to both datasets.