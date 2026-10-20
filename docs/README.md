# Reading guide

## Current four-model release

1. [Methodology and configuration explorer](visualizations/README.md):
   workload, analytical SOL, capacity envelopes, prefill provisioning and scoring.
2. [Sensitivity to methodology choices](research/methodology-sensitivity/README.md):
   model baskets, linear/log speed weighting, speed bands, prefill ratios and
   single-speed alternatives. Includes full CSVs and interactive tables.
3. [InferenceX validation](research/inferencex-validation/README.md):
   empirical evidence, matched configurations, curve agreement and limitations.

The current headline uses equal GLM-5.2, Kimi-K3, Qwen3-Coder 480B and
MAI-Base-2 2.8T proxy weights, log 50–400 SOL TPS/user and separate prefill/decode
provisioning. The sensitivity study does not replace that definition.

## Historical archive — not the current methodology

- `METHODOLOGY.txt` and `METHODOLOGY.html` describe the **9 October two-model
  decode-only release**. “Current” inside them refers to that date.
- `REPRODUCTION.txt` documents historical packaging checks.
- [Reproduction history](REPRODUCTION_HISTORY.md) preserves older instructions.

The historical methodology files are hash-listed in the old release manifest;
their original bytes are retained for auditability. Use the current reading
order above, not the historical filenames, for present setup and explanations.
