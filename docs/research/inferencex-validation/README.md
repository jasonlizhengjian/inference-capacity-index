# InferenceX versus modeled SOL — 19 October 2026

[Public interactive comparison](https://inferencex-sol-validation.epoch-3293.chatgpt.site) ·
[Offline HTML](index.html) · [Observed rows](observed_points.csv) ·
[Curve ratios](curve_summary.csv) · [GLM strata](matched_glm_strata.csv)

Kimi K3 is primary, covering NVIDIA, AMD and Google TPU. GLM-5.2 is secondary:
its TensorRT-LLM logs provide the stronger execution-time check. This study
**does not change the index, frozen timings, hardware specifications or scores**
and fits no achieved-performance multiplier.

## Three-curve explorer: the current default

The public explorer now defaults to the **full observed InferenceX speed range**,
a smoothed measured frontier, and **50% of SOL speed** for both modeled phases.
Real TPS = 0.5 × SOL TPS means real TPOT = 2 × SOL TPOT, not half the latency.
The prefill efficiency can instead be left at 100%, or both can be shown at SOL.
This is a scenario, not a fitted multiplier or a change to index scores.

The three lines are:

1. **InferenceX measured:** all saved observations, with a Pareto frontier.
2. **Model restricted to InferenceX layouts and pool sizes:** observed TP/DP/EP/
   DCP, physical pool allocations, and all memory-admitted integer batches.
3. **Best-known model configurations:** frozen search plus newly evaluated
   submitted layouts, with freely provisioned pools and the best prefill rate.

Curve 2 is **layout-matched, not exact iso-config replay**. Both modeled curves
hold index context (256Ki), precision, graph, nextn3/AL3 and memory rules fixed.
Client concurrency is not used as live batch. Actual per-step batch, context,
draft graph and speculative width remain unmatched. Prefill is PCP1/DCP1.
There are 45 admitted unique deployment shapes; unsupported mappings or
admission failures are recorded, not imputed. In particular, TPU's TP32 across
16 physical chips has no audited native logical-device mapping, so its green
curve is unavailable. Some other platforms have no admitted observed-layout
recipe at the fixed index workload/precision; this is not zero real performance.

The blue curve includes the new observed layouts in addition to the original
search so it cannot lose merely because the frozen search omitted, for example,
TP16/DCP16. It is not the unchanged published index curve or a globally optimal
search. Its candidate-set extension affects this validation only.

For each candidate, decode speed shifts by ηD while admitted batch and decode
requests/chip stay fixed. Prefill rate shifts by ηP. At achieved speed floor s:

```text
Freely provisioned or ideal shared pools:
  C(s) = 1 / [1/Cdecode(s/ηD) + r*s/(ηP*P)]
Fixed separate pools (N = ND + NP):
  C(s) = min[Cdecode(s/ηD)*ND/N, ηP*P*NP/(r*s*N)]
```

The shared-pool equation is an ideal accelerator-time bound, not a mixed-batch
scheduler. For output throughput, evaluate each candidate at its own discounted
speed and maximize its resulting Q among qualifying candidates. Both modeled
curves receive the same efficiency assumptions. The measured data never shift.

Under the default half-speed/both-phases scenario over each chip's **own full
observed band**, Kimi GB300 measured/layout-restricted capacity AUC is 68.1%,
measured/best-known is 38.6%, and restricted/best-known is 56.6%. MI355X gives
98.0%, 91.8%, and 93.7%. These describe conditional curve gaps, not matched-step
efficiency or a cross-chip ranking. Workload/precision differences still apply.
Some measured fast points exceed even the scenario's modeled maximum speed;
they remain plotted and explicitly warned about rather than being clipped.

Smoothing uses shape-preserving cubic Hermite interpolation through measured
Pareto points in **log speed**, not a regression or altered measurements. Raw
points remain visible. The smoothed guide need not be an achievable frontier
between observations and does not extrapolate above the fastest observation.
Every AUC uses the **unsmoothed** data. Full-band statistics cover min–max
observed speed for each chip; the 50–400 option supplies a common band instead.

Measured areas use exact step integration. New modeled three-curve areas use
midpoint quadrature in log speed, split at all candidate speed breakpoints and
refined until successive results agree relatively within 1e-6. This handles
crossings between fixed-pool limits. Numerical convergence is not physical
accuracy. `three_curve_summary.csv` preserves bounds, assumptions and convergence
checks. `layout_restricted.json` retains native candidates, pool sizes and
exclusions. The original undiscounted frozen-curve results below remain in
`curve_summary.csv`; they are not the new default explorer statistics.

## Findings

- GB300 GLM sampled decode steps achieve **73.5%** of modeled SOL speed for
  TP1/DP8/EP8 and **53.6%** for TP1/DP16/EP16 using live batches. Treating padded
  verification positions as full requests gives **87.2%** and **66.4%** instead.
  These are work-accounting conventions, not uncertainty bounds.
- Kimi exact matched-step validation remains unavailable. Actual DCP can differ
  from API metadata; some recipes exceed the pinned engine's speculative-width
  support. Interval running-request samples do not identify kernel batches.
- Whole-service streaming-equivalent capacity AUC ratios are **13.3% GB300,
  31.6% MI355X and 6.4% TPU7** for Kimi. These are **not matched-workload execution
  efficiency**: workload, precision, implementation, utilization and search
  coverage differ. Do not rank hardware implementation quality from them.
- Approximate Kimi scenarios can exceed 100%. This signals mismatched graphs
  or workload assumptions, not exceeding a physical roofline for identical work.

## Reproduction and provenance

The pre-study repository commit was `a7f3e7406fae4c2da0d2597177d413c8e7460b84`.
Runtime: unchanged AISim 0.13.0, upstream commit
`2f0f4cd8606d580567b253d02f3d699fe36db865`; native binary SHA-256
`2e2a1772e067c45253bbe6b6fe9f7fbdedbb8f90d7a9d2bbaf2c4a2ffbe3ccb4`.

`studies/inferencex_validation/evidence/manifest.json` records source URLs,
retrieval times and API hashes (117 Kimi rows, 116 GLM rows before filtering).
The theoretical reference is `reference/2026-10-15/curves_and_inputs.json`;
its hash is also recorded in `curves.json`. Log audit JSON includes exact file
names, source URLs, hashes, bytes and completeness. Large logs are bounded to
100 chunks of 262,144 characters, about 26 MB. Later source availability/content
can change; reproduction from re-fetched logs must compare hashes.

```bash
# Offline, standard-library curve recomputation and HTML regeneration:
python3 studies/inferencex_validation/analyze.py
python3 studies/inferencex_validation/compare.py
python3 studies/inferencex_validation/render.py

# Optional public-log acquisition and native replay in the pinned environment:
python3 studies/inferencex_validation/fetch_logs.py \
  --ids 444604 444605 444607 444602 445218 445220 445227 445368 445370
.venv-current/bin/python studies/inferencex_validation/analyze.py --native
.venv-current/bin/python studies/inferencex_validation/layouts.py
python3 studies/inferencex_validation/compare.py
python3 studies/inferencex_validation/render.py
```

`fetch.py` refreshes API evidence, changing the snapshot; it is not part of
offline reproduction. Raw logs stay in ignored `results/`. No private trace,
GPU, credentials, model-weight download or SGLang scheduler is needed. The
public report contains technical summaries, not raw serving logs.

## Service-curve equations

Select AgentX rows with positive TPOT and output throughput. For disaggregated
recipes, allocated chips `N = Nprefill + Ndecode`; for aggregated recipes,
`N = max(Nprefill, Ndecode)` because both fields describe the same shared pool.
Check `output_tput_tps/N` against `output_tput_per_gpu`. TPU uses API physical
chip counts, not logical TP devices.

Define `Q = output_tput_tps/N`, `s = 1/mean_tpot` (seconds), and `C = Q/s`.
Reciprocal mean TPOT is not mean request speed. **C is throughput-equivalent
continuous-stream density, not measured concurrency.** Client concurrency is
never substituted for live kernel batch or measured capacity.

At threshold `t`, measured capacity is `max C_i` among saved rows with `s_i ≥ t`.
The output-throughput view independently takes `max Q_i` among those rows.
Dates, precision and frameworks can differ. There is no interpolation or
extrapolation: empty regions mean zero *observed* coverage, not physical
infeasibility. H200 Kimi has no observed point reaching the 50-TPS band.

The SOL capacity comparison is the published index calculation:

```text
C_SOL(t) = 1 / [1/Cdecode(t) + r*t/P]
r = 1.513342523752522
```

`P` is the saved maximum prefill throughput for 1,024 new tokens over 262,144
resident tokens. The direct output-throughput view evaluates each saved decode
point at its native speed: `Q_SOL,i = C_i*s_i/(1+r*C_i*s_i/P)`, then maximizes
qualifying Q. Using `t*max C_i` instead would not maximize output throughput.

Integrate over log speed, 50–400. Measured steps contribute `C*ln(b/a)`; SOL
capacity contributes `C*ln[b(1+rCa/P)/(a(1+rCb/P))]` per constant decode-density
segment. Every saved speed is a breakpoint; equality qualifies. The analytic
integral is independently checked by midpoint quadrature to relative tolerance
1e-5. Display sampling and the optional 800-TPS view do not alter the score.

This is **not a matched-workload comparison**. AgentX includes variable contexts,
new input, tool pauses, prefix handling and actual pool allocation. The model
has fixed resident-cache decode and idealized separate prefill provisioning,
without queueing or KV handoff. Different precision/draft/acceptance assumptions
can shift the comparison either way. Many observed contexts average less than
256Ki. The gap cannot be attributed solely to execution inefficiency.

## Kimi audit and timing probes

| Result | Hardware | Actual TP / DCP | Draft tokens | Recorded AL | Replay |
|---|---|---|---:|---:|---|
| 445218 | GB300 | 16 / 16 | 5 | 3.620 | SGLang substitution only |
| 445220 | GB300 | 8 / 8 | 7 | 3.841 | unsupported width |
| 445227 | GB300 | 8 / 8 | 4 | 3.352 | SGLang substitution only |
| 445368 | MI355X | 8 / 1 | 7 | 3.841 | unsupported width |
| 445370 | MI355X | 8 / 8 | 4 | 3.361 | SGLang substitution only |

The first two GB300 rows report DCP1 in the API but DCP16/DCP8 at startup.
DCP is within TP, not an additional multiplicative chip count. The first has
client concurrency 1 yet sampled positive running counts 1–3. Neither client
concurrency nor sampled running count is an exact scheduled kernel batch.

Pinned Kimi/vLLM rejects DCP. SGLang supports it but uses a different draft
graph. Native nextn accepts 0–5, so seven-token records are **not clamped**.
Supported-width probes use running counts {1, median positive, min positive,
max positive}, input contexts {mean,p50,p90}, and precision variants. These
are not jointly observed states and have **no memory admission check**.
Raw times sum native operation latencies. The scenarios are not exact replay.

For fixed-width speculative counters, `AL = 1 + gamma*Σaccepted/Σdrafted`, using
the log's bonus-token convention. This is recorded execution AL, not natural
model acceptance: recipes can impose synthetic acceptance distributions.
Precision probes use index NVFP4/BF16, MXFP4/FP8 MoE and (AMD) blanket FP8
linear layers. The last does not match selective FP8 runtime overrides.

TPU Kimi uses published FP4 storage versus the frozen index's FP8 expert
storage. A private serving branch and missing equivalent step logs prevent
precision/configuration-matched validation. AMD's actual node/fabric policy
also need not match the conservative index policy. Rubin's theoretical profile
has not been verified against the measured system. These limitations remain
visible rather than being fitted away.

## GLM matched-step method

Pair an iteration's state with the **following consecutive record's
`prev_device_step_time`**. Keep decode-only pairs within the profiling window,
nonempty batch and mean cached context ≥1,024. Verification padding is
`num_generation_tokens/4` (nextn3 verifies four positions). Exclude nonintegral
or smaller-than-live shapes.

Stratify by `(live batch, padded batch, floor(mean context/32768))`; retain
strata with ≥20 observations. Use each stratum's median context for AISim and
median observed device time. Summary times are count-weighted stratum times;
the fraction is their ratio, not the average of per-step percentages.

| Result | Layout | Included steps | Observed ms | SOL live ms | SOL padded ms |
|---|---|---:|---:|---:|---:|
| 444604 | TP1 DP8 EP8 DCP1 | 39,740 | 27.8827 | 20.4887 | 24.3095 |
| 444605 | TP1 DP16 EP16 DCP1 | 37,263 | 22.1563 | 11.8756 | 14.7012 |

`T_SOL/T_observed` is the achieved fraction of modeled SOL speed for the same
step; no accepted-length discount is applied. Native nextn includes modeled
draft and verification work. Logged TRT version:
`1.3.0rc28.dev202609210001`. Bundled checkpoint-derived quantization and explicit
index quantization produced identical live timings here; this does not prove
all runtime precision exclusions or fusions match.

Rank-zero state is treated as representative of other ranks. Means lose
context-distribution information. Padded full-request modeling can overstate
KV work. Windows: 12:07:00–12:26:02 of 12:07:00–13:07:00 (444604), and
14:03:11–14:17:34 of 14:03:11–15:03:11 (444605). These are **early, partial-run
samples, not random or complete samples**. Result444607's bounded log did not
reach profiling; 444602 had incompatible verification shapes; neither enters
the timing summaries. Exclusion counts and representative source line numbers
are preserved. All-rank and whole-run representativeness remain unresolved.

## Implementation map

| Concern | Repository file / function |
|---|---|
| Snapshot and hashes | `studies/inferencex_validation/fetch.py`, `evidence/manifest.json` |
| Bounded public logs | `studies/inferencex_validation/fetch_logs.py:fetch` |
| Pool counts, envelopes, exact AUC | `studies/inferencex_validation/analyze.py:curves` |
| GLM pairing/strata; Kimi audit | `studies/inferencex_validation/analyze.py:native` |
| Observed-layout batch sweep and exclusions | `studies/inferencex_validation/layouts.py` |
| Three curves, efficiency scenarios and AUC | `studies/inferencex_validation/compare.py` |
| Existing envelope implementation | `src/agent_index/index.py:envelope`, `capacity`, `area` |
| Existing prefill-adjusted integral | `studies/log400_release.py:combined_area` |
| Native configuration and runtime pin | `src/agent_index/simulation.py:make_config`; `current_native.py:doctor` |
| Offline/public HTML | `studies/inferencex_validation/render.py`, `template.html` |

Primary sources: [InferenceX](https://inferencex.semianalysis.com/) and its
[benchmark code](https://github.com/SemiAnalysisAI/InferenceX). Exact API and
workflow links are retained per observation. This analysis is independent and
does not imply endorsement by SemiAnalysis.
