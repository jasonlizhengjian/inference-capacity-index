# Reading guide

## Report snapshot: 20 October 2026 Rubin revision

1. [Markdown methodology](METHODOLOGY.md), [report figures](report/README.md) and
   [interactive explorer](visualizations/README.md):
   workload, analytical SOL, capacity envelopes, prefill provisioning and scoring.
2. [Sensitivity to methodology choices](research/methodology-sensitivity/README.md):
   model baskets, linear/log speed weighting, speed bands, prefill ratios and
   single-speed alternatives. Includes full CSVs and interactive tables.
3. [InferenceX validation](research/inferencex-validation/README.md):
   empirical evidence, matched configurations, curve agreement and limitations.

See the [snapshot and command guide](SNAPSHOTS.md) before comparing result files.

The current headline uses equal GLM-5.2, Kimi-K3, Qwen3-Coder 480B and
MAI-Base-2 2.8T proxy weights, log 50–400 SOL TPS/user and separate prefill/decode
provisioning. The sensitivity study does not replace that definition.

The [20 October Rubin revision](research/rubin-spec-update/README.md) updates
VR200 to 19.2 TB/s HBM and 1.5 TB/s outbound NVLink, giving **2.177 GB300e**.
Current visualization tables and methodology sensitivity include it. The
InferenceX study retains its 19 October observations and now uses revised
Rubin inputs, with the five affected layout recipes retimed on 29 October. Original release reproduction
remains available separately from `python3 reproduce.py rubin-update`.

## Separate hardware extensions

- [Trainium2 and Trainium3 scenarios](research/trainium/README.md): matched-workload
  scores, topology sensitivities, eight-chip controls, resource sanity checks and
  the conditional effect on the inference-capacity growth estimate. This study
  does not change the headline roster or published growth chart.

## Historical archive — not the current methodology

- `METHODOLOGY.txt` and `METHODOLOGY.html` describe the **9 October two-model
  decode-only release**. “Current” inside them refers to that date.
- `REPRODUCTION.txt` documents historical packaging checks.
- [Reproduction history](REPRODUCTION_HISTORY.md) preserves older instructions.

The historical methodology files are hash-listed in the old release manifest;
their original bytes are retained for auditability. Use the current reading
order above, not the historical filenames, for present setup and explanations.
