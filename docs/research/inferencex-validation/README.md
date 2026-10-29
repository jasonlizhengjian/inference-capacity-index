# InferenceX versus modeled SOL — 19 October 2026

**Benchmark observations: 19 October 2026. Modeled hardware: 15 October base plus the 20 October Rubin revision, refreshed 29 October.** Rubin uses 19.2 TB/s HBM and 1.5 TB/s outbound NVLink. All five admitted Rubin submitted-layout recipes were retimed with the pinned engine; non-Rubin recipes and benchmark observations are unchanged. This is separate from the launch report’s October 27 chart. [Evidence levels](../../METHODOLOGY.md#what-the-evidence-supports) | [Snapshot guide](../../SNAPSHOTS.md) | [Pre-refresh archive](../archives/2026-10-19-pre-rubin-refresh/README.md).

[Historical hosted comparison](https://inferencex-sol-validation.epoch-3293.chatgpt.site) ·
[Offline HTML](index.html) · [Observed rows](observed_points.csv) ·
[Current curve ratios](three_curve_summary.csv) ·
[Pointwise agreement](point_agreement.csv) · [GLM strata](matched_glm_strata.csv)

Kimi K3 is primary, covering NVIDIA, AMD and Google TPU. GLM-5.2 is secondary:
its TensorRT-LLM logs provide the stronger execution-time check. This study
**does not change the report’s index definition or scores**
and fits no achieved-performance multiplier.

## What the comparison establishes

**Kimi-K3 on GB300 provides encouraging agreement at the measured frontier
points.** With decode and prefill performance set to 50% of modeled SOL,
predicted output throughput differs from measured throughput by **11.6%
on average (about 12%) at five frontier points in the 50–400 TPS/user band**.
The prediction searches only the admitted deployment layouts and pool sizes
found in the benchmark submissions. The 50% assumption was specified before
this pointwise calculation; it was not fitted to minimize these errors.

This is an **envelope comparison**, not five iso-config replays: the modeled
winner at a speed floor can use a different submitted layout from the measured
winner. Both sides optimize throughput at that speed requirement. The measured
points are selected by Pareto dominance, not by their agreement with the model.
The model holds the index workload, precision and speculation fixed.

The result is informative about this setup, not a universal 12% error bar on
the index. The wider measured speed range gives 31.1% average frontier-point
error, and other hardware/model combinations differ considerably. The full
point table, alternative cohorts and scenarios are saved—not just the five
points supporting the short explainer.

| Question | Comparison | What the number means |
|---|---|---|
| Does a half-SOL model describe observed operating points? | Modeled submitted-layout envelope versus measured frontier points | Average absolute throughput difference; 11.6% for Kimi/GB300, 50–400 |
| How much integrated capacity or throughput does the model predict? | Areas under the unsmoothed curves | An area ratio, not average prediction error; can hide cancellation |
| What does the broader configuration search add? | Best-known modeled envelope versus submitted-layout envelope | Modeled benefit of layout search and flexible pool allocation, not demonstrated engineering headroom |
| How close is actual device execution to SOL? | GLM log-derived decode steps versus modeled steps | Achieved-speed fraction; a different quantity from service-curve error |

## Accuracy: upstream predictions versus this SOL study

AISim’s predecessor, [AIConfigurator](https://arxiv.org/abs/2601.06288),
reported **7.8% mean absolute percentage error in decode-token latency** in
its aggregated-serving evaluation (§5.1). That evaluation used measured
kernel-performance data for Qwen3-32B and Qwen3-235B on an eight-H100 SXM
node, across TensorRT-LLM and (for the smaller model) vLLM configurations.
It is not an accuracy result for our specification-based SOL mode, 256Ki
context workload, other hardware, or final capacity-index scores.

Our Kimi result above tests a separate proposition: whether a simple half-SOL
scenario describes the service frontier reasonably well. GLM's step-level
check later in this report provides additional execution evidence, rather
than being pooled into the Kimi prediction-error statistic. Neither comparison
establishes an error bound on cross-hardware index ratios, especially for
unmeasured architectures or future chips.

## The five-point result, worked through

At each measured **output-throughput frontier** point with speed `s_i` between
50 and 400, look up the best modeled throughput among admitted submitted
layouts and batches reaching at least `s_i`. Apply `ηD = ηP = 0.5` first.
Both throughputs count all allocated prefill and decode chips:

```text
signed difference_i = predicted_Q(s_i) / measured_Q_i − 1
average absolute difference = (1/n) × Σ abs(signed difference_i)
```

Every retained point receives equal weight. The denominator is measured
throughput. The absolute value prevents over- and underprediction from
cancelling. There is no smoothing, efficiency fitting or interpolation in this
calculation. It is not time-, token-, traffic- or log-speed-weighted.

| InferenceX result | TPS/user | Measured output TPS/chip | Predicted output TPS/chip | Signed difference |
|---|---:|---:|---:|---:|
| 445228 | 60.50 | 108.95 | 123.92 | +13.7% |
| 445230 | 79.81 | 95.12 | 72.80 | −23.5% |
| 445229 | 108.81 | 70.93 | 63.07 | −11.1% |
| 445232 | 143.88 | 58.30 | 58.91 | +1.0% |
| 445223 | 387.60 | 27.76 | 25.41 | −8.5% |

The mean absolute difference is **11.5552%**, median 11.1%, largest 23.5%.
Four of five points are within 20%. These five points are from one benchmark
snapshot and are not independent replications supporting a statistical
confidence interval.

**Which configuration actually wins?** All five modeled predictions above
use the submitted TP16/DCP16 aggregated layout represented by result 445218,
with native per-replica batches 39, 16, 10, 7 and 1 respectively. The measured
winners use TP8 layouts: the first four use separate prefill/decode pools,
and the last uses eight shared chips. Thus “using the benchmark's layouts”
means choosing from the **union** of submitted layouts, not holding each
point's own layout fixed. `winning_recipe_id`, `winning_batch` and
`winning_native_speed` in [the point CSV](point_agreement.csv) expose this.
Shared-pool modeled throughput uses ideal time-sharing without interference.

For example, at the first observed speed floor of 60.496 TPS/user, the modeled
TP16 candidate has native SOL speed 121.927 and half-SOL speed 60.964, so it
qualifies. After including the modeled prefill resource requirement its
throughput is 123.916 output TPS per total chip, versus 108.953 observed:
`abs(123.916 / 108.953 − 1) = 13.7%`. The model need not run at exactly the
threshold; it must meet or exceed it.

### How sensitive is this description of agreement?

| Kimi/GB300 comparison | Average absolute difference | Interpretation |
|---|---:|---|
| Five frontier points, 50–400; both phases at 50% | **11.6%** | The short-explainer result |
| Same points; decode 50%, prefill 100% | 12.6% | Small change from prefill-efficiency assumption here |
| Same points; both phases at undiscounted SOL | 256.3% | SOL is not an achieved-performance prediction |
| All 11 frontier points, full observed 31–680 range; both phases 50% | 31.1% | Includes two fast points with zero qualifying modeled throughput |
| All nine observations in 50–400, including dominated runs; both phases 50% | 130.3% | An optimized envelope should not be expected to predict inefficient individual runs |
| Entire unsmoothed step curves, log-uniform thresholds over 50–400; both phases 50% | 34.9% | Weights speed intervals, including sparsely measured regions, instead of measured points |

There is no output-frontier observation between 143.9 and 387.6 TPS/user.
The conservative measured step envelope drops immediately above 143.9;
the smoothed display draws a gradual decline. Neither establishes actual
performance at intermediate configurations. This explains part of the
difference between pointwise agreement and full-curve agreement.

The measured/model **throughput AUC ratio is 84.1%** over 50–400, while the
capacity AUC ratio is 77.5%. Neither is the 11.6% error statistic. Throughput
and streaming-equivalent density have different Pareto frontiers. Even though
`C_i = Q_i/s_i` for an individual measurement, independently maximizing `Q`
and `C` over all qualifying measurements need not select the same run.

### Other hardware and model checks

For the same 50–400 band and half-SOL/both-phases scenario, the saved
output-frontier point comparisons include:

| Model / hardware | Points in band | Average absolute throughput difference |
|---|---:|---:|
| Kimi / GB300 | 5 | 11.6% |
| Kimi / GB200 | 5 | 202.5% |
| Kimi / B300 | 5 | 82.2% |
| Kimi / MI355X | 11 | 40.0% |
| GLM / GB300 | 11 | 165.5% |
| GLM / MI355X | 10 | 344.0% |

These are descriptive **service-envelope gaps**, not a ranking of simulator
or vendor kernel accuracy: workloads, measured contexts, serving policies,
precision and graph matching differ. A gap over 100% is possible when modeled
throughput is more than twice measured throughput. Missing layout mappings
(including TPU) have unavailable errors, not fabricated zero-efficiency results.
The complete table is in [point_agreement_summary.csv](point_agreement_summary.csv).
This contrast is why we name Kimi/GB300 in the short explanation rather than
generalizing its 12% agreement to all platforms or GLM service curves.

**Next evidence that would be most useful:** replay the measured winning
deployment at its observed batch, context distribution, precision and draft
graph; obtain comparable all-rank logs for AMD and TPU; and extend empirical
checks to Qwen and the MAI proxy. Those are more direct tests of generalization
than selecting an efficiency multiplier separately for every hardware. This
refresh retimes only the five Rubin recipes and makes no new calibration fit.

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

The Y-axis can be linear or logarithmic. Log Y makes multiplicative gaps and
low-capacity fast tails easier to compare; zero-capacity segments are omitted,
not replaced with an arbitrary positive floor. This is display-only: the AUC
still integrates capacity against log speed, not log capacity. Geometric area
on a log–log plot therefore is not the reported AUC.

Measured areas use exact step integration. New modeled three-curve areas use
midpoint quadrature in log speed, split at all candidate speed breakpoints and
refined until successive results agree relatively within 1e-6. This handles
crossings between fixed-pool limits. Numerical convergence is not physical
accuracy. `three_curve_summary.csv` preserves bounds, assumptions and convergence
checks. `layout_restricted.json` retains native candidates, pool sizes and
exclusions. The original undiscounted frozen-curve results below remain in
`curve_summary.csv`; they are not the new default explorer statistics.

## Additional execution evidence and legacy comparison

- GB300 GLM sampled decode steps achieve **73.5%** of modeled SOL speed for
  TP1/DP8/EP8 and **53.6%** for TP1/DP16/EP16 using live batches. Treating padded
  verification positions as full requests gives **87.2%** and **66.4%** instead.
  These are work-accounting conventions, not uncertainty bounds.
- Kimi exact matched-step validation remains unavailable. Actual DCP can differ
  from API metadata; some recipes exceed the pinned engine's speculative-width
  support. Interval running-request samples do not identify kernel batches.
- In the **original frozen-envelope, undiscounted-SOL** comparison over
  50–400, whole-service streaming-equivalent capacity AUC ratios are **13.3% GB300,
  31.6% MI355X and 6.4% TPU7** for Kimi. These are **not matched-workload execution
  efficiency**: workload, precision, implementation, utilization and search
  coverage differ. These are legacy `curve_summary.csv` values, not the
  current half-SOL, submitted-layout comparison. Do not rank hardware
  implementation quality from them.
- Approximate Kimi scenarios can exceed 100%. This signals mismatched graphs
  or workload assumptions, not exceeding a physical roofline for identical work.

## Reproduction and provenance

The pre-study repository commit was `a7f3e7406fae4c2da0d2597177d413c8e7460b84`.
Runtime: unchanged AISim 0.13.0, upstream commit
`2f0f4cd8606d580567b253d02f3d699fe36db865`; native binary SHA-256
`2e2a1772e067c45253bbe6b6fe9f7fbdedbb8f90d7a9d2bbaf2c4a2ffbe3ccb4`.

`studies/inferencex_validation/evidence/manifest.json` records source URLs,
retrieval times and API hashes (117 Kimi rows, 116 GLM rows before filtering).
The theoretical reference combines `reference/2026-10-15/` with
`reference/2026-10-20-rubin/`, using `rubin_release.report_inputs`.
The merged-input and revision-manifest hashes are recorded in `curves.json`. Log audit JSON includes exact file
names, source URLs, hashes, bytes and completeness. Large logs are bounded to
100 chunks of 262,144 characters, about 26 MB. Later source availability/content
can change; reproduction from re-fetched logs must compare hashes.

```bash
# To repeat only the affected native timing refresh (optional; results are saved):
.venv-current/bin/python studies/inferencex_validation/layouts.py --rubin-only

# Offline, standard-library curve recomputation and HTML regeneration:
python3 studies/inferencex_validation/analyze.py
python3 studies/inferencex_validation/compare.py
python3 studies/inferencex_validation/agreement.py
python3 studies/inferencex_validation/render.py
python3 studies/verify_analysis_refresh.py

# Optional public-log acquisition and native replay in the pinned environment:
python3 studies/inferencex_validation/fetch_logs.py \
  --ids 444604 444605 444607 444602 445218 445220 445227 445368 445370
.venv-current/bin/python studies/inferencex_validation/analyze.py --native
.venv-current/bin/python studies/inferencex_validation/layouts.py
python3 studies/inferencex_validation/compare.py
python3 studies/inferencex_validation/agreement.py
python3 studies/inferencex_validation/render.py
```

`fetch.py` refreshes API evidence, changing the snapshot; it is not part of
offline reproduction. Raw logs stay in ignored `results/`. No private trace,
GPU, credentials, model-weight download or SGLang scheduler is needed. The
public report contains technical summaries, not raw serving logs.

`agreement.py` reads only the existing curve and layout artifacts and records
their SHA-256 hashes in `point_agreement.json`. The detailed CSV retains all
point/scenario/metric combinations. The JSON and summary CSV store equal-point
errors for both Pareto and all-observation cohorts, separately for both
comparators. Error fields are fractions (`0.115552` means `11.5552%`);
`restricted_signed_error` is positive for overprediction. `winning_native_speed`
is the winning candidate's undiscounted SOL TPS/user, including the fixed AL3
assumption. Winner IDs refer to representative submitted deployment shapes,
not the IDs of the measured rows being predicted. A zero modeled prediction
at an observed point counts as 100%
error; an entirely unavailable layout comparison remains null. Output step-curve
error is integrated exactly over the union of measured and modeled breakpoints,
excluding intervals without observed coverage and reporting that coverage.
The 50–400 report includes only measured points in that band; it does not invent
measurements at the band boundaries.

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
| Pointwise errors, cohorts, winners, exact output step-curve gaps | `studies/inferencex_validation/agreement.py` |
| Existing envelope implementation | `src/agent_index/index.py:envelope`, `capacity`, `area` |
| Existing prefill-adjusted integral | `studies/log400_release.py:combined_area` |
| Native configuration and runtime pin | `src/agent_index/simulation.py:make_config`; `current_native.py:doctor` |
| Offline/public HTML | `studies/inferencex_validation/render.py`, `template.html` |

Primary sources: [InferenceX](https://inferencex.semianalysis.com/) and its
[benchmark code](https://github.com/SemiAnalysisAI/InferenceX). Exact API and
workflow links are retained per observation. This analysis is independent and
does not imply endorsement by SemiAnalysis.

## 29 October hardware-only refresh

The saved 19 October API observations and non-Rubin timing probes are unchanged.
Five admitted GLM/Rubin recipes were retimed, preserving their layouts, pool
sizes, graph, precision, context and nextn3/AL3 assumptions. The saved benchmark
snapshot has no admitted Kimi/Rubin observations, so no Kimi/Rubin agreement
claim is added. The historical hosted site is not the refreshed output; use
this directory’s HTML and CSVs for the report revision.

For GLM/Rubin, measured / index-optimal SOL capacity area changes from
0.253789 to 0.256107, and output-throughput area from 0.087845 to 0.099725
over 50–400 tokens/s/request. These are area ratios, not pointwise prediction
errors. The Kimi/GB300 five-point mean absolute throughput difference remains
11.5552%. [Refresh checks and hashes](refresh_verification.json) document
what changed and verify the retained observations and unaffected results.
