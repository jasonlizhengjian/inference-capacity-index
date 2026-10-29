# How the Inference Capacity Index is calculated

**Report snapshot: ICI v1 methodology, 20 October 2026 Rubin profile revision.**
Four equally weighted models; 50–400 tokens/s/request on a logarithmic speed
axis. GB300 = 1; VR200 = 2.177 GB300e. This is the snapshot behind report
Figures 1, 3 and 4. [Reproduction and figure map](report/README.md) |
[Snapshot guide](SNAPSHOTS.md) | [Interactive explorer](visualizations/REPORT.html).

The ICI compares the number of concurrent generating requests supported per
allocated chip across a fixed basket of workloads. It combines decode and
separately provisioned prefill resources. It is not an arithmetic average of
a decode score and a prefill score.

### Terminology in saved files

Public text uses **output speed** (tokens/s/request), **concurrency**
(requests/chip), and **output throughput** (tokens/s/chip). Legacy `TPS/user`
and `speed` mean output speed; `agents_chip` and `capacity` mean concurrent
continuously generating requests per allocated chip, with the phase denominator
specified by the table. `agent_index_GB300e` is the combined ICI score.
`agent_index` Python imports and legacy column names are retained so old files
remain readable. “SOL” identifies an analytical estimate, not another unit.

## 1. Concurrency and output speed

A decode configuration specifies how the model and requests are divided across
chips. For a batch of `b` requests per attention replica:

```text
requests = b × DP
allocated chips = TP × DP × PP
c = requests / allocated chips                 [requests/chip]
s = 1000 × accepted_length / cycle_time_ms      [tokens/s/request]
q = c × s                                     [output tokens/s/chip]
```

Only chips allocated to that model instance enter the denominator, not unused
chips in the maximum deployment budget. DCP partitions work within TP; EP and
expert TP partition the same chips used by attention. They are not additional
multipliers. For example, TP8 DP8 DCP8 EP64 ETP1 PP1 uses 64 chips. A batch
of 16 per replica is 128 requests, or 2 requests/chip.

Larger batches share weight reads across more requests, but add cache reads
and other request-dependent work. Consequently, total throughput can rise
while each request slows down. This does not require the workload to become
compute-bound. [Figure 2 and its data](report/README.md#figure-reproduction-map)
show the fixed-layout example; [bottleneck explorer](visualizations/PERFORMANCE_KNEES.html)
provides additional examples with its own dated inputs.

### Execution-time assumptions

The native analytical engine is **Nvidia AISimulate 0.13.0**, upstream commit
`2f0f4cd8606d580567b253d02f3d699fe36db865`. The engine is unmodified; this
repository supplies hardware profiles, model adapters, memory guards and search
rules. The [runtime manifest](../reference/2026-10-15/runtime.json) pins the
wheel, native binary and bundled model configurations. No SGLang scheduling
trace produces the headline score; SGLang/vLLM labels select native operation
recipes, not actual serving-engine execution.

SOL means analytical performance under the modeled hardware resource limits.
For operations with both compute and memory costs, the roofline estimate uses
`max(FLOPs / FLOP_per_second, bytes / bytes_per_second)`. Other operations have
specialized rules, including bandwidth-only paths and collective costs.
`Native.evaluate` sums the native operation latencies and any topology overlay.
Do not interpret the generic roofline equation as a claim that every operator
models all physical bottlenecks, or that every operation can overlap freely.

Decode uses native `nextn=3`: verification covers four target positions, with
model-specific draft work included. The cycle is divided by an **assumed three
accepted output tokens**, not by four. This is not a measured acceptance rate
for every model, nor three independent samples of the same kernel. The generic
GQA verification path can repeat cache-read costs that a fused implementation
could share. The index applies no universal 50% achieved-performance discount.

Implementation: [`Native.graph`, `memory`, `evaluate`, `batches`](../src/agent_index/current_native.py),
[`make_config`, `enumerate_layouts`, `_memory`](../src/agent_index/simulation.py),
[model-specific rules](../src/agent_index/current_models.py),
[topology overlay](../src/agent_index/topology.py).

## 2. Provision prefill alongside decode

Let `P` be the maximum evaluated prefill throughput in **new input tokens/s/chip**.
It is optimized independently for 1,024 new tokens over a resident
262,144-token prefix, without a prefill latency target or speculative decoding.
Let `r = 1.513342523752522` new input tokens per generated output token.
For `N` continuously generating requests at required speed `s`:

```text
decode chips = N / c
prefill chips = r × N × s / P
combined capacity B(s) = N / (decode chips + prefill chips)
                      = c / (1 + r × s × c / P)
```

This provisions just enough prefill capacity to keep up with decode. The prefill
share changes with hardware, model and required speed; it is not a fixed 15%
weight. It assumes fully utilized separate pools and fractional allocation at
fleet scale. It excludes queueing, TTFT, KV transfer between pools, tool pauses
and production scheduling interference. A single small deployment may not be
able to realize the fractional allocation exactly.

The ratio is **aggregate new-input tokens divided by aggregate output tokens**
in a selected Claude warm-cache coding-trace cohort, not the mean of individual
request ratios and not an all-provider average. The saved
[workload record](../reference/2026-10-15/workload.json) contains the cohort
query, count (110,565), mean new input (1,422.85), mean output (940.21) and source
hashes. Filters require at least 90% cached input, 131,072–393,216 cached tokens,
and 1–10,240 new tokens. The underlying trace database is not redistributed.
The same ratio is transferred to all four models; their differing tokenizers,
verbosity and quality are not separately modeled. The 1,024-token prefill case
is a deliberate throughput proxy, not replay of the full input distribution.
See [prefill-ratio sensitivity](research/methodology-sensitivity/README.md#what-changes).

## 3. Combine configurations into an optimal curve

At each required speed, select the greatest decode request density among saved
memory-admitted configurations that meet the speed:

```text
C(s) = max c_i for candidates with s_i >= s
```

For fixed `s`, `P` and `r`, combined capacity is increasing in `c`, so selecting
this decode maximum also maximizes `B(s)`. Different layouts or batches can win
at different speeds. This is the best of a finite search, not proof of a global
optimum across every possible kernel or parallelism scheme.

Search uses every integer batch up to the memory admission limit within its
allowed layout family. TP/DP and EP/ETP cover the same per-stage chips;
head/expert divisibility, DCP restrictions and maximum chip budgets apply.
Node-restricted GPU profiles keep collectives within eight-chip stages and
allow the saved conservative **serial** PP schedule between stages. Other
profiles use PP1. Prefill uses CP1/DCP1. Wider exploratory layouts in the
validation study are not silently added to the report's index.

Memory admission uses 90% of total HBM, native weight/workspace reservations,
resident cache, and model-specific guards for metadata, pipeline boundaries
and uneven stage weights. The default profile's fixed reservation is 4 GiB.
See the actual guards in `simulation._memory` and `current_models`; capacity
cannot be obtained simply by dividing free HBM by KV size. Hardware capacity
and bandwidth use decimal bytes; displayed GiB use 2^30 bytes. The 10% unused
headroom is not an extra cache allocation.

Independent instances cannot improve the maximum per-allocated-chip density:
their combined density is a chip-weighted average of their densities. This
claim assumes the same workload and speed floor, independent resources and
fractional provisioning. It does not exclude benefits from heterogeneous pools,
packing under integer chip constraints, or a different traffic mixture.

## 4. Integrate over output speeds

For each hardware/model pair, integrate combined capacity:

```text
A = integral from 50 to 400 of B(s) ds/s
```

Log weighting gives equal weight to each doubling: 50–100, 100–200 and
200–400 tokens/s/request. It represents a deliberate coverage choice, not an
estimated probability distribution of demand. The band covers slower bulk work
and faster interactive use; SOL speeds are not promised achieved speeds.
[Axis, band and single-speed alternatives](research/methodology-sensitivity/README.md#linear-versus-logarithmic-weighting)
show the effect of this choice, using the same hardware revision as the report.

Integration splits at **every saved speed breakpoint** inside the band. Between
breakpoints decode capacity is constant, `c`, but prefill-adjusted capacity
varies smoothly as `c/(1+k*s)`, where `k=r*c/P`. The exact contribution on `[a,b]` is:

```text
c × ln[b × (1+k*a) / (a × (1+k*b))]
```

Equality qualifies at a saved speed. If no candidate reaches a speed, capacity
is zero there. There is no extrapolation, smoothed interpolation or positive
floor. Endpoints have zero measure for the integral. Display sampling does not
change scores. A log Y-axis is only a display option: we integrate capacity,
not log capacity. Implementation: [`segments`, `combined_area`](../studies/log400_release.py)
and [`capacity`](../src/agent_index/index.py).

## 5. Normalize and average models

For each model `m`, divide hardware `h`'s area by GB300's area for that same
model, then average the four ratios:

```text
ICI(h) = (1/4) × sum_m [ A(h,m) / A(GB300,m) ]
```

This keeps a model with more requests/chip from dominating merely because its
raw capacity is higher. Equal model weights express architectural coverage;
they are not observed traffic shares or a claim of equal model quality.
[Basket and proxy-weight sensitivity](research/methodology-sensitivity/README.md#model-basket-sensitivity)
shows why the ICI is not an architecture-independent hardware multiplier.

For the report's Rubin revision the four ratios are 1.224658, 2.476490,
2.354606 and 2.653713. Their mean is **2.177367**, displayed as **2.18 GB300e**.
The **2.48** in Figure 3 is Kimi's area ratio, not the final ICI or a ratio
holding at every speed. GB300 is 1 by construction.

## Workload and model assumptions

| Input | Value used | Basis |
| --- | --- | --- |
| Resident context | 262,144 tokens per request | Deliberate long-context representative case; not the trace's exact mean |
| New input per prefill | 1,024 tokens | Fixed throughput proxy |
| New input/output | 1.513342523752522 | Aggregate ratio from the filtered Claude coding-trace cohort above |
| Speculation | nextn3, accepted length 3 | Native model graph plus assumed acceptance; not universally measured |
| Attention/non-expert arithmetic | BF16 | Shared study precision policy |
| KV cache / communication | FP8 / half precision | Shared study policy; recurrent-state accounting is model-specific |
| Expert arithmetic/storage | Profile-dependent, table below | Hardware adaptation and precision-equivalence assumptions |
| Memory admission | 90% total HBM plus explicit guards | Conservative study policy, not observed production occupancy |
| Basket weights | 25% each | Architectural diversity, not market shares |
| Compute and links | Profile peak FLOP/s, HBM bytes/s, one-way outbound link bytes/s and group sizes | Dated specifications and disclosed topology assumptions |

The frozen [catalog](../reference/2026-10-15/catalog.json) requests `fp8` experts
for H100, H200 and TPU v7. Its other 20 profiles request `nvfp4`, sometimes as
an **FP4-equivalent modeling proxy**, not a claim that each vendor implements
Nvidia's format. [Model rules](../src/agent_index/current_models.py) and
[hardware YAMLs](../reference/2026-10-15/hardware/) define the actual modeled
costs. The revised [Rubin YAML](../reference/2026-10-20-rubin/hardware/vr200_deployment.yaml)
supersedes only Rubin's bandwidth inputs. Equal model quality across these
precision choices is not established. Later hardware extensions have their own
precision policies and must not be mixed silently with this catalog.

| Model | Attention representation in the study | Status and main memory consequence |
| --- | --- | --- |
| GLM-5.2 | Compressed latent cache with sparse attention selecting up to 2,048 tokens; indexer work also counted | Bundled native model. Retains full-context cache although the main attention reads a subset |
| Kimi-K3 | 69 recurrent linear-attention layers plus 24 full-context MLA layers | Bundled native model. Recurrent state stays bounded; full-attention cache grows with context |
| Qwen3-Coder 480B | 62 full-context GQA layers, 8 KV heads | Native GQA graph. Cache and reads grow with context throughout |
| MAI-Base-2 proxy | 85 local-window and 17 global GQA layers; latent MoE | Study adapter. Window 512 and head dimension 128 inherited from Base-1; shared FFN width 4,352 assumed |

The MAI reconstruction has about 2.84T total and 85.9B active parameters versus
Microsoft's rounded 2.8T/80B description. It is not an exact Microsoft serving
implementation. [Saved model records](../reference/2026-10-15/models.json) and
[adapter](../src/agent_index/current_models.py) preserve these assumptions.

## What the evidence supports

Reproduction checks establish computational consistency: identical inputs and
runtime reproduce saved results. They do not measure physical accuracy.
The launch report's October 27 InferenceX chart is evidence about **broad
ordering** under benchmark workloads. It is separate from the
[validation study](research/inferencex-validation/README.md), which retains
19 October observations and uses the report’s hardware revision after its
29 October refresh.
That study's approximately 12% average throughput difference applies to five
Kimi/GB300 frontier points under a 50%-SOL speed scenario, using a union of
submitted layouts. It is not exact point-by-point iso-config replay or an
index-wide error estimate. Read its other hardware results and workload/layout
mismatches alongside that number. Upstream AIConfigurator's empirical-kernel
accuracy is also distinct from this specification-based SOL calculation.

Compare scores from the same [snapshot](SNAPSHOTS.md). Architecture, workload,
precision, topology and implementation assumptions can change relative scores;
the index does not currently assign a statistical confidence interval to them.
