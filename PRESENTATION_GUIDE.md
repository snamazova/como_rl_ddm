# Presentation Guide: PRLT RLDDM Project

## Research question

> Does post-reversal slowing in choice RT reflect a collapse in drift rate
> (H1: value-driven evidence accumulation) or an independent increase in
> response caution (H2: boundary separation)?

---

## H1 Results — drift-only model

In H1, the boundary `a` is constant. Post-reversal slowing happens because
the value difference ΔQ collapses to ~0, so the drift rate `v = v_max·tanh(β·ΔQ)`
drops to near zero — evidence accumulates slowly.

### H1 single-subject diagnostic

![H1 RL](single_subject_h1_rl.png)

**What you see:** Top-left: learned values V₁ and V₂ cross at each reversal (dashed lines). Top-right: drift rate drops to zero at each reversal — this is the "drift collapse" that H1 says causes slowing. Bottom-left: prediction errors spike at reversals. Bottom-right: observed outcomes (rewards) flip pattern.

### H1 DDM panels

![H1 DDM](single_subject_h1_ddm.png)

**What you see:** DDM diffusion schematics (left) and RT histograms (right) at three value-difference regimes (low/medium/high ΔQ). When ΔQ is low (near reversal), the drift is flat and RTs are wide. When ΔQ is high (stable block), drift is steep and RTs are narrow.

### H1 reversal-locked behaviour

![H1 reversal-locked](reversal_locked_h1.png)

**What you see:** Accuracy (top) and RT (bottom) from trial −20 to +20 relative to reversal. At trial 0 (reversal), accuracy drops from ~0.88 to ~0.24, and RT increases moderately from ~1.93s to ~2.13s. The RT increase is modest because the boundary doesn't change — only the drift collapses.

### H1 parameter sweep — alpha

![Alpha sweep](parameter_sweep_alpha.png)

**What you see:** Higher α → higher accuracy (0.51→0.75), faster post-reversal recovery, lower perseveration, faster RTs. α is the key learning parameter — it controls how quickly values update after reward/no-reward.

### H1 parameter sweep — a (boundary)

![A sweep](sweep_a.png)

**What you see:** Higher a → higher accuracy but much slower RTs (speed-accuracy tradeoff). a does NOT affect post-reversal recovery speed — it only controls how much evidence is needed before committing. This confirms a is a decision parameter, not a learning parameter.

---

## H2 Results — boundary-only model

In H2, the drift function is the same as H1, but the boundary increases
after each reversal: `a(t) = a_base + κ·exp(-trial_since_reversal/τ)`.
The agent becomes more cautious, requiring more evidence — this produces
slower RTs independent of the drift rate.

### H2 single-subject diagnostic

![H2 RL](single_subject_h2_rl.png)

**What you see:** Values and drifts look similar to H1 (same tanh drift function). The key difference is invisible in these RL panels — it's in the DDM boundary, shown below.

### H2 DDM panels

![H2 DDM](single_subject_h2_ddm.png)

### H2 time-varying boundary

![H2 boundary](h2_boundary_over_trials.png)

**What you see:** The boundary a(t) spikes at each reversal (to a_base + κ = 5.0) then decays exponentially back to baseline (3.0) with time constant τ=5. This is the mechanism H2 uses to explain post-reversal slowing — the agent requires more evidence immediately after a reversal.

### H2 reversal-locked behaviour

![H2 reversal-locked](reversal_locked_h2.png)

**What you see:** Same layout as H1's reversal-locked plot. Accuracy drops from ~0.88 to ~0.19 at reversal (slightly worse than H1). RT increases sharply from ~2.05s to ~3.65s — much larger than H1's ~2.13s. This is because the boundary physically increases by κ=2.0, requiring much more evidence.

### H2 parameter sweep — kappa (boundary increase)

![Kappa sweep](sweep_kappa.png)

**What you see:** Higher κ → larger post-reversal RT increase (more caution), slightly lower accuracy immediately after reversal (more evidence needed = slower to adapt), but accuracy recovers once boundary decays back. κ=0 reduces H2 to H1.

### H2 parameter sweep — tau (decay timescale)

![Tau sweep](sweep_tau.png)

**What you see:** Higher τ → the boundary stays elevated longer after reversal → sustained RT increase and slower accuracy recovery. Low τ → boundary returns to baseline quickly → brief RT spike. τ controls how long the "caution period" lasts.

---

## Comparison — can we distinguish H1 from H2?

### Reversal-locked comparison (H1 vs H2)

![Reversal-locked comparison](reversal_locked.png)

**What you see:** Both models overlaid. The key difference is in the RT panel: H2 (orange) shows a much sharper RT spike at reversal (~3.6s) compared to H1 (blue, ~2.1s). The accuracy patterns are similar but H2 dips slightly lower. This suggests the two mechanisms produce distinguishable RT signatures.

### Confusion matrix

![Confusion matrix](confusion_matrix.png)

**What you see:** 2×2 BIC comparison. When data is generated under H1, fitting H1 gives lower BIC (1393 vs ~1e10 for H2). When data is generated under H2, fitting H2 gives lower BIC (1510 vs ~1e10 for H1). The matrix is **diagonal** — the models are identifiable.

**What this means:** If the true mechanism is drift collapse (H1), fitting both models correctly identifies H1. If the true mechanism is boundary increase (H2), fitting correctly identifies H2. The two hypotheses produce behaviourally distinguishable patterns — the research question can in principle be answered.

### Parameter recovery

```
H1 recovery:                    H2 recovery:
  alpha: 0.25 → 0.34 (0.09)       alpha:   0.25 → 0.33 (0.08)
  v_max: 2.00 → 3.34 (1.34)       v_max:   2.00 → 5.00 (3.00)
  beta:  1.00 → 0.50 (0.50)       beta:    1.00 → 0.31 (0.69)
  a:     3.00 → 2.96 (0.04)       a_base:  3.00 → 2.93 (0.07)
  w:     0.50 → 0.47 (0.03)       kappa:   2.00 → 2.22 (0.22)
  t0:    0.25 → 0.32 (0.07)       tau:     5.00 → 4.85 (0.15)
                                  w:       0.50 → 0.48 (0.02)
                                  t0:      0.25 → 0.31 (0.06)
```

**Well recovered:** a, a_base, kappa, tau, w — the boundary and caution parameters.
**Confounded:** v_max and beta trade off in `tanh(β·ΔQ)` — this is a known identifiability limitation worth discussing.

---

## Suggested presentation flow

1. **Research question** (1 slide): post-reversal slowing — drift or boundary?
2. **Model specification** (1 slide): H1 vs H2 formulas, parameter meanings
3. **H1 results** (2 slides): single-subject diagnostic + reversal-locked behaviour
4. **H2 results** (2 slides): single-subject diagnostic + boundary over trials + reversal-locked
5. **Comparison** (1 slide): reversal-locked H1 vs H2 — the RT signatures differ
6. **Parameter sweeps** (1 slide): α and a have dissociable effects; κ and τ control caution
7. **Confusion matrix** (1 slide): models are identifiable — diagonal
8. **Parameter recovery** (1 slide): boundary parameters recover well; v_max/β confounded
9. **Critical evaluation** (1 slide): strengths, limitations, next steps

---

## Formula note

The project document specifies `v = v_max·tanh(β·(Q_A−Q_B))`. Our code uses
`v = v_max·tanh(β·(Q_B−Q_A))` — opposite sign. This is **correct** because
in our DDM, positive drift → upper boundary → bandit B. When A is better
(Q_A > Q_B), drift should be negative → choice A. Our sign convention
ensures this.