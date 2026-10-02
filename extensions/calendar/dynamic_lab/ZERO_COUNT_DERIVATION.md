# Sparse-count boundary correction and its remaining limit

The response measurement uses `h = raw_events × estimated_human_fraction`, weighted exposure E, and log value `log((h+0.5)/E)`. It is an approximate Gaussian measurement, not an exact transformed NB/audit likelihood.

The initial variance used

```
V_old = (h + raw_events² Var(p_human)) / (h+0.5)² + V_NB
```

At raw=h=0 its sampling term was0, leaving the small numerical floor0.02. That incorrectly made a zero-count cell a highly precise log-rate observation.

The corrected rule consistently uses the half-count in the count-variance approximation:

```
V_new = 1/(h+0.5)
        + raw_events² Var(p_human)/(h+0.5)²
        + V_NB
V_NB = [Σ_d exposure_d²/(Σ_d exposure_d)²] / r
```

One interpretation is a first-order delta approximation to a positive count-intensity quantity with mean and variance `h+0.5`, as in a unit-rate Gamma half-count reference. The derivative of log at that mean is1/(h+0.5), so its approximate log variance is1/(h+0.5). At h=0 it is2 before NB uncertainty. The audit term adds uncertainty in the nested audited human fraction, rather than treating labels and raw counts as independent repeated outcomes.

This repairs a boundary inconsistency. It does **not** make the Gaussian approximation reliable for sparse bins. Even for that simple Gamma reference, exact log variance is trigamma(h+0.5), which at zero isπ²/2≈4.935, not2. Its exact mean log is digamma(h+0.5), rather than log(h+0.5). The audit mixture, clipping and NB process make the implemented likelihood less exact still. These reference moments are a diagnostic comparison, not the true posterior for the whole model.

`zero_count_reference.csv` records those exact reference quantities beside the delta approximation. The persistent regression checks ensure the zero-bin variance is2, rather than returning to false precision. The final held-out coverage failure remains visible; none of the priors or scale supports was tuned to repair coverage. A future exact/marginalized count-and-audit likelihood is separate work and is not claimed by this implementation.
