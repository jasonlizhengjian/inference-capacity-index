# Sensitivity to methodology choices

The headline is unchanged: equal GLM-5.2, Kimi-K3, Qwen3-Coder 480B and
MAI-Base-2 2.8T proxy weights; log 50–400 SOL TPS/user; separate prefill/decode
provisioning at the saved new-input/output ratio of 1.513342523752522.

This study re-integrates the **same frozen 23-accelerator, four-model release**
from 15 October 2026. No timing, precision, memory, topology, model-dimension or
layout-search assumptions change. Later standalone hardware additions, including
A100, are not silently merged into this frozen basket.

Open [the interactive tables](index.html), or download [all scores](score_matrix.csv).
The detailed [long table](scenario_results.csv) includes percentage and rank
changes; [per-model contributions](per_model_sensitivity.csv) expose architecture
dependence. [Family summaries](family_robustness.csv) keep distinct choices separate.

## What changes?

- **Axis × band:** both linear and log integration over 50–200, 50–300,
  50–400 and 50–800. Same-band comparisons isolate weighting.
- **Basket:** GLM + Kimi, every model alone, and every leave-one-out basket.
- **Proxy weight:** 0%, 10%, 25%, 50%; remaining weight shared equally.
- **Prefill:** decode-only, half, current and double the input/output ratio.
- **Point alternative:** capacity at 50, 100, 200 or 400 SOL TPS/user.

There are 28 named scenarios, including the headline and the explicit
25%-proxy control (which reproduces it). Basket, ratio and point scenarios hold
all other headline choices fixed. This is not a full factorial of every choice.

## Linear versus logarithmic weighting

Linear integrates capacity against **ds**: equal absolute speed intervals have
equal weight. Log integrates against **ds/s**: equal multiplicative intervals
have equal weight. Changing only a display axis does not change a score.

For VR200, log 50–400 gives **2.389 GB300e**, while linear over the
same band gives **2.703**. Log 50–200 gives
**2.252** and log 50–800 gives **2.495**.
These are sensitivity results, not a reason to pick the methodology that
produces a preferred accelerator ratio.

| Accelerator | log_50_400 | log_50_200 | log_50_300 | log_50_800 | linear_50_200 | linear_50_300 | linear_50_400 | linear_50_800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GB300 NVL72 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Meta MTIA 500 (512 GB) | 3.198 | 3.032 | 3.119 | 3.331 | 3.109 | 3.314 | 3.548 | 4.114 |
| Meta MTIA 500 (384 GB) | 2.993 | 2.833 | 2.920 | 3.124 | 2.935 | 3.138 | 3.357 | 3.924 |
| MI455X | 2.893 | 2.736 | 2.820 | 3.017 | 2.813 | 3.007 | 3.225 | 3.744 |
| VR200 | 2.389 | 2.252 | 2.328 | 2.495 | 2.348 | 2.523 | 2.703 | 3.155 |
| Meta MTIA 450 | 2.014 | 1.917 | 1.969 | 2.083 | 1.972 | 2.092 | 2.225 | 2.510 |
| Maia 300 rack | 1.593 | 1.520 | 1.559 | 1.645 | 1.560 | 1.651 | 1.752 | 1.968 |
| Jalapeño | 1.291 | 1.241 | 1.267 | 1.325 | 1.269 | 1.331 | 1.399 | 1.541 |
| Meta MTIA 400 | 1.208 | 1.184 | 1.198 | 1.215 | 1.194 | 1.224 | 1.248 | 1.271 |
| Ascend 960DT | 0.934 | 0.910 | 0.923 | 0.944 | 0.921 | 0.952 | 0.984 | 1.025 |
| TPU8i · corrected · hops | 0.912 | 0.910 | 0.911 | 0.905 | 0.910 | 0.912 | 0.913 | 0.885 |
| GB200 NVL72 | 0.900 | 0.889 | 0.895 | 0.903 | 0.893 | 0.905 | 0.915 | 0.925 |
| TPU7x Ironwood · FP8 | 0.742 | 0.748 | 0.743 | 0.739 | 0.741 | 0.732 | 0.728 | 0.721 |
| TPU8t · FP4 equivalent | 0.720 | 0.717 | 0.719 | 0.721 | 0.718 | 0.724 | 0.728 | 0.728 |
| Maia 200 rack | 0.677 | 0.693 | 0.680 | 0.673 | 0.678 | 0.650 | 0.643 | 0.632 |
| MI355X · 400G | 0.552 | 0.579 | 0.561 | 0.545 | 0.562 | 0.527 | 0.511 | 0.494 |
| B300 · 800G | 0.541 | 0.569 | 0.550 | 0.534 | 0.552 | 0.516 | 0.498 | 0.481 |
| B200 · 800G | 0.392 | 0.405 | 0.397 | 0.388 | 0.391 | 0.376 | 0.365 | 0.355 |
| Ascend 950DT | 0.357 | 0.370 | 0.361 | 0.354 | 0.363 | 0.343 | 0.334 | 0.324 |
| Ascend 960PR | 0.257 | 0.282 | 0.263 | 0.251 | 0.251 | 0.219 | 0.207 | 0.191 |
| Ascend 950PR | 0.126 | 0.140 | 0.129 | 0.122 | 0.120 | 0.103 | 0.096 | 0.087 |
| H200 SXM · FP8 · 400G | 0.123 | 0.130 | 0.125 | 0.118 | 0.121 | 0.112 | 0.109 | 0.091 |
| H100 SXM · FP8 · 400G | 0.040 | 0.040 | 0.041 | 0.038 | 0.039 | 0.042 | 0.038 | 0.030 |

## Model-basket sensitivity

All columns below retain log 50–400 and the observed input/output ratio. Individual
models are architecture diagnostics, not a proposed single-model headline.
For VR200 the two-model GLM + Kimi basket gives **1.989**, versus
**2.389** for the four-model headline. Removing the MAI proxy gives
**2.199**; increasing its weight to 50% gives **2.579**.
The per-model range is wider: **1.236–2.959**.
This shows why a basket score should not be presented as an architecture-independent
hardware multiplier. See the full tables for other accelerators and rank changes.

| Accelerator | Headline | GLM + Kimi | GLM only | Kimi only | Qwen only | MAI proxy only | Without proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| GB300 NVL72 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Meta MTIA 500 (512 GB) | 3.198 | 2.825 | 2.154 | 3.496 | 3.334 | 3.810 | 2.995 |
| Meta MTIA 500 (384 GB) | 2.993 | 2.514 | 1.639 | 3.389 | 3.295 | 3.649 | 2.774 |
| MI455X | 2.893 | 2.546 | 1.879 | 3.214 | 2.896 | 3.584 | 2.663 |
| VR200 | 2.389 | 1.989 | 1.236 | 2.743 | 2.619 | 2.959 | 2.199 |
| Meta MTIA 450 | 2.014 | 1.726 | 1.189 | 2.263 | 2.226 | 2.377 | 1.893 |
| Maia 300 rack | 1.593 | 1.390 | 1.103 | 1.677 | 1.933 | 1.660 | 1.571 |
| Jalapeño | 1.291 | 1.051 | 0.869 | 1.232 | 1.675 | 1.386 | 1.259 |
| Meta MTIA 400 | 1.208 | 1.169 | 1.087 | 1.252 | 1.189 | 1.303 | 1.176 |
| Ascend 960DT | 0.934 | 0.869 | 0.952 | 0.786 | 1.053 | 0.944 | 0.930 |
| TPU8i · corrected · hops | 0.912 | 0.928 | 0.980 | 0.875 | 0.944 | 0.849 | 0.933 |
| GB200 NVL72 | 0.900 | 0.825 | 0.678 | 0.972 | 1.000 | 0.950 | 0.883 |
| TPU7x Ironwood · FP8 | 0.742 | 0.708 | 0.646 | 0.771 | 0.888 | 0.661 | 0.768 |
| TPU8t · FP4 equivalent | 0.720 | 0.718 | 0.723 | 0.712 | 0.763 | 0.684 | 0.733 |
| Maia 200 rack | 0.677 | 0.650 | 0.660 | 0.640 | 0.814 | 0.593 | 0.704 |
| MI355X · 400G | 0.552 | 0.544 | 0.707 | 0.382 | 0.858 | 0.263 | 0.649 |
| B300 · 800G | 0.541 | 0.541 | 0.711 | 0.371 | 0.826 | 0.255 | 0.636 |
| B200 · 800G | 0.392 | 0.290 | 0.386 | 0.194 | 0.819 | 0.169 | 0.466 |
| Ascend 950DT | 0.357 | 0.344 | 0.390 | 0.298 | 0.444 | 0.297 | 0.377 |
| Ascend 960PR | 0.257 | 0.294 | 0.390 | 0.198 | 0.296 | 0.142 | 0.295 |
| Ascend 950PR | 0.126 | 0.137 | 0.195 | 0.079 | 0.184 | 0.044 | 0.153 |
| H200 SXM · FP8 · 400G | 0.123 | 0.054 | 0.086 | 0.022 | 0.368 | 0.016 | 0.158 |
| H100 SXM · FP8 · 400G | 0.040 | 0.014 | 0.023 | 0.006 | 0.125 | 0.005 | 0.051 |

## Exact calculation and interpretation

For each speed threshold s, C(s) is the highest requests/allocated-decode-chip
density among saved qualifying layouts and batches. P is the saved optimal
prefill tokens/chip-second and r is new input/output tokens.

`B(s) = C(s) / (1 + r*s*C(s)/P)`.

At r=0, B=C. With r>0 and no prefill throughput, B=0. Integrate B using every
saved speed breakpoint. On a segment [a,b] with constant C=c and k=r*c/P:

- Linear area: `c/k * log((1+k*b)/(1+k*a))`, with limit `c*(b-a)` at k=0.
- Log area: `c * log(b*(1+k*a)/(a*(1+k*b)))`.

Implementation uses stable `log1p` forms. Capacity is zero when no saved point
qualifies, with no interpolation or extension past maximum feasible speed.
Exactly at a saved speed, a point with that speed still qualifies; point
comparisons respect this equality. Endpoints have zero measure for AUC.

Normalize each model to **GB300 under the identical scenario**, then take the
stated weighted arithmetic mean. Single-point scores normalize capacity rather
than area. A positive-weight model with zero GB300 reference makes the whole
scenario score undefined: no epsilon and no silent reweighting.

The combined capacity is monotonic in decode capacity for fixed s, P and r,
so reselecting the maximum saved decode density at each speed remains optimal
under this provisioning equation. This is optimization over saved candidates,
not a fresh layout search. Separate fully utilized prefill pools and fractional
fleet-scale provisioning remain assumptions.

[Robustness ranges](robustness.csv) are min/max across these deliberately selected
scenarios, **not confidence intervals or probabilities**. Single-model endpoints
are architecture diagnostics, not equally plausible replacement baskets.
Use family-specific ranges for narrower conclusions. Scores are displayed to
three decimals; CSVs retain full precision. Ranks use unrounded scores with a
1e-12 relative tie tolerance; positive rank change means improvement.

## Reproduce and audit

Run `python3 reproduce.py sensitivity` from the repository root. Standard library
only; no AISim installation, network access, private traces or GPU required.
Outputs go to `results/methodology-sensitivity/`. Use
`--output docs/research/methodology-sensitivity` to refresh this checked-in snapshot.

Source: `studies/methodology_sensitivity.py`. [Provenance](provenance.json) records
input/code hashes and headline parity. All 23 headline scores are
checked against the frozen release; maximum relative error:
`2.22e-16`.

Changed context length, speculative step size, precision, memory guards, topology
or architecture dimensions generally require new timing/search runs and are
outside this re-integration study. Basket weights are model weights, not traffic
shares. No update to the public headline, Google Sheet or hosted explainer is
implied by these alternative scenarios.
