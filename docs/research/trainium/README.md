# Trainium inference-capacity scenarios

**Hardware-profile revision: 22 October 2026 Trainium scenarios.** The GB300 reference is unchanged; no revised Rubin curve enters these Trainium/GB300 ratios. This remains a separate extension. [Snapshot guide](../../SNAPSHOTS.md).

22 October 2026. **Separate research extension, not promoted to the headline
index or the inference-compute-growth chart.** Scores are modeled SOL capacity,
not Neuron benchmarks. [Charts and tables](index.html), [all scenario scores](overall.csv),
[per-model scores](per_model.csv), [winning layouts](targets.csv),
[growth sensitivity](growth_impact.csv), [validation](validation.json).

## Main result

The H100-FLOPs fallback materially understates Trainium capacity in these
matched-workload projections. The numerical replacement is still conditional
on network interpretation and deployment configuration.

| Scenario | Combined GB300e/chip | Relative to existing fallback |
| --- | ---: | ---: |
| Trainium2: existing fallback | 0.026 | 1.0× |
| Trainium2: 16-chip, link-limited | 0.073 | 2.8× |
| Trainium2: 64-chip, link-limited | 0.128 | 4.9× |
| Trainium2: same, plus illustrative 4 μs collective startup | 0.118 | 4.5× |
| Trainium2: 64-chip, link-limited with one-way table interpretation | 0.152 | 5.8× |
| Trainium2: 64-chip injection ceiling | 0.171 | 6.5× |
| Trainium3: existing fallback | 0.050 | 1.0× |
| Trainium3: 64-chip, detailed link tiers | 0.349 | 6.9× |
| Trainium3: 64-chip, half aggregate table bandwidth | 0.464 | 9.2× |
| Trainium3: 144-chip, half aggregate table bandwidth | 0.478 | 9.5× |
| Trainium3: 144-chip injection ceiling | 0.506 | 10.0× |

The complete table also includes a pessimistic Trainium3 slow-tier placement
case (0.275, or 0.257 with startup) and eight-chip controls. These are scenario
results, **not confidence bounds**. No probability weights or central average
have been assigned to conflicting specifications.

## Does this look too high for the hardware?

The main results pass a resource-based sanity check. The injection-ceiling
cases should still be treated as optimistic rather than preferred estimates.

| Modeled per-chip resource | Trainium2 | Trainium3 | GB300 reference |
| --- | ---: | ---: | ---: |
| HBM capacity, GiB | 96 | 144 | 268.22 |
| HBM bandwidth, TB/s | 2.9 | 4.9 | 8.0 |
| Dense BF16 tensor throughput, PF/s | 0.632 | 0.632 | 2.5 |
| Expert arithmetic throughput, PF/s | 1.264 FP8 | 2.517 MXFP8/MXFP4 | 15 NVFP4 |
| Expert weight bytes/parameter | 1 | 0.53125 | 0.5625 |
| Domain budgets tested | 16 / 64 | 64 / 144 | 72 |

Trainium3 has 61% of GB300's HBM bandwidth, 54% of its capacity and 25% of
its BF16 throughput. Accounting for the different FP4 scale formats, its
expert-weight streaming ceiling is about 65% of GB300's. A combined score
around 0.35–0.48 is therefore not an implausible resource advantage: it remains
below both the bandwidth ratio and the weight-streaming ratio. Its standalone
prefill score is only about 0.21–0.28, consistent with its smaller compute budget.
These ratios are sanity checks, not bounds on every optimized workload.

The apparently strong comparison to HGX or Hopper is partly a **domain-size
effect**. With Trainium forcibly limited to at most eight chips and PP1, scores
fall to 0.043 for Trainium2 and 0.160 for Trainium3. The 64-chip cases rise to
0.128 and 0.349 respectively under the more cautious bandwidth assumptions.
For orientation, the existing H200 score is 0.123 and B200 is 0.392. Those
GPU profiles allow node-local collectives with serial cross-node PP, so these
are not perfectly iso-topology comparisons. The control establishes that
wide placement materially contributes; it does not attribute every difference
to topology alone.

Remaining potential optimism includes collective routing/placement, chip-local
SRAM and tensor tiling, unmodeled conversion/fusion costs, balanced expert work,
and the common assumed speculative accepted length. There is no empirical
Trainium calibration here. An AWS hardware profile does not turn the inherited
vLLM/SGLang operation recipes into validated Neuron serving implementations.

## Workload and scoring

- Same equally weighted basket: GLM-5.2, Kimi-K3, Qwen3-Coder 480B and the
  MAI-Base-2 2.8T architecture proxy. No model weights are downloaded.
- Decode: 262,144 resident tokens per request, native nextn3, assumed accepted
  length 3. TPS/user = 3000 / modeled verification-and-draft step milliseconds.
- Prefill: 1,024 new tokens over a 262,144-token cached prefix, no speculation;
  optimize input tokens per allocated chip-second, without a latency target.
- Exact capacity envelope across every admitted integer batch and the existing
  finite layout family. TP/DP/DCP/EP/ETP divisibility rules are unchanged; PP1
  within these Trainium domains. No unbounded EFA scale-out is granted.
- 144 is a deployment budget, not automatically 144 charged chips. Existing
  layouts use up to 128; 144 does not permit integral expert placement in these
  four models with the searched ETP values. Chips charged = TP × DP × PP;
  requests = batch × DP. Unallocated chips are not charged.
- Memory: same 90% total-HBM admission, native workspace and 4 GiB fixed
  reservation, plus the existing generic metadata guard and explicit MX scales.
- Let C(s) be decode requests per allocated chip at speed s, P be the maximum
  prefill input tokens/chip-second, and r = 1.513342523752522. Combined capacity
  is C(s)/(1+r s C(s)/P). Integrate over d ln(s), 50–400 TPS/user, normalize to
  GB300 separately per model, then average the four ratios. Infeasible ranges
  contribute zero. This is not an 85/15 arithmetic score.

## Resource choices and precision accounting

The AWS chip pages report 96/144 **GiB**, not decimal GB, and 2.9/4.9 decimal
TB/s. Whole-chip headline BF16 rates are 667/671 TF/s. This study uses the
eight Tensor Engines' approximately 632 TF/s BF16 rather than crediting all
aggregate engine throughput to GEMMs. Trainium2 FP8 uses 1264 TF/s; Trainium3
uses the published 2517 TF/s MX rate. Structured-sparsity peaks are excluded.
These rounded tensor-only rates are explicit choices, not measured utilization.

Trainium2 experts use FP8; no native FP4 rate is assigned. Trainium3 uses
MXFP4 expert weights and MXFP8 input/intermediates. AWS documents a scale byte
per 32 values and equal MXFP4/MXFP8 arithmetic rates, not a doubled FP4 peak.
BF16 remains the non-expert GEMM/attention precision and KV stays FP8.
No equal-accuracy claim is made between different FP4 formats.

The pinned native MoE mode represents weights at 0.5 byte/parameter without
explicit MX scales. The isolated Trainium wrapper therefore:

1. Adds 1/16 of native routed-weight bytes to residency, giving 0.53125 byte
   per parameter. The existing generic reserve remains separate and unchanged.
2. Re-evaluates only MoE leaf memory traffic: 0.53125 byte/weight,
   1.03125 byte/input or intermediate value, and BF16 final output. It retains
   the native token counts, expert saturation/floor convention, GEMM work,
   actual FP8 compute peak, and max(compute time, memory time) aggregation.
3. Reconstructs native overlap branch sums and maxima, applying the corrected
   leaf inside its branch. All other compute/attention timing stays native.

This is a small study-layer mapping correction, **not an AISim source patch**.
Baseline native MoE and overlap timings are checked before the correction.
The active native MoE path already uses the dtype-specific FP8 rate; it does
not need a fabricated BF16-to-FP8 multiplier adjustment.

## Network scenarios and their limitations

Trainium2 has a 4×4 torus per 16-chip instance; four instances are joined by
rings connecting corresponding chip positions. The detailed table gives
1024 GiB/s local and 256 GiB/s cross-instance aggregate bandwidth per chip,
without resolving directionality. The main link-limited scenario divides by
two for send/receive, then by four local neighbors or two ring neighbors:
128 GiB/s local and 64 GiB/s wider effective collective rates. This is an
explicit conservative ring-bandwidth abstraction, **not a proven optimal
torus roofline**. The one-way interpretation doubles those rates. The injection
ceiling uses the full aggregates and ignores contention.

Trainium3 has switched domains of 64 or 144 chips. AWS's detailed connectivity
section gives 256/320/128 GB/s bidirectional local/inter-server/inter-rack
links, while its table says 2048 GiB/s per device and its chip page says
2.56 TB/s. These do not reconcile just by converting units. Accordingly:

- `64_detailed_tiers`: 128 GB/s within four chips, 160 GB/s beyond four.
- `64/144_aggregate_half`: 1024 GiB/s one-way, half the aggregate table entry.
- `144_detailed_slow_tier`: 128 GB/s within four, 64 GB/s for every wider group.
  This deliberately pessimistic placement scenario does **not** assert that
  every actual group crosses racks or that 144-chip systems are inherently
  slower than 64-chip systems. Rack placement is not known sufficiently well.
- `144_injection_ceiling`: uniform 2.56 TB/s one-way, deliberately optimistic.

Tiered rates are supplied to separate native engines, including any transfer
costs internal to attention modules. The optional 4 μs is charged per exposed
collective invocation, within overlap branches; it is not a measured AWS
startup or a per-hop model. Hidden module-internal startups are not augmented.
No physical routing, bisection-load solver or per-link contention simulation
is claimed. Physical NeuronCore placement, eight-core tiling, compiler padding,
and SRAM-managed data reuse remain outside the model.

## Growth impact and interpretation

The published fallback gives 4.677× growth from Q2 2025 to Q2 2026, with
6.432 million GB300e at the endpoint. Replacing all Trainium2 shipments by the
64-chip link-limited score and Trainium3 by the 64-chip detailed-tier score
gives about **4.583× and 6.665 million GB300e**. Alternative tested deployment
scenarios give approximately 4.55–4.64× growth and a 1.9–5.2% higher endpoint.
These are conditional scenario replacements, not fleet deployment estimates.
The source does not reveal the proportion of Trainium2 deployed as 16 versus
64 chips, nor the Trainium3 generation/domain mix. Trainium1 is not rescored.

The `growth_inputs.json` file contains only the required public aggregate
shipment counts, baseline capacities, fallback scores and source hashes.
No private workload traces are included. No live chart, Sheet or headline
index has been modified by this study.

## Reproduction and verification

From the repository root, using the existing pinned environment:

```bash
.venv-current/bin/python studies/trainium/run.py --jobs 2
python3 studies/trainium/report.py
```

The first command writes new scenario results to
`results/trainium-2026-10-22-audit/`; it does not overwrite published inputs.
The second rebuilds the report and growth calculation from the checked-in
research results, without AISim, GPUs or network. `--prepare` is an authoring
step that imports local completed results and a sibling public sales snapshot;
it is not required to reproduce the saved report.

`runs/` contains all scenario frontier winners, prefill winners and layout
admission audits. Dominated candidate timings are not all retained, but the
source reruns every admitted integer batch. `targets.csv` identifies winning
parallelisms, batch, chips, requests, timing and memory at each target speed.
`hardware/` contains native profile inputs, including flat profiles used only
for parity checks. Native source is pinned to AISim 0.13.0 commit
`2f0f4cd8606d580567b253d02f3d699fe36db865`; runtime hashes and wrapper/input
hashes are in `provenance.json`. Validation checks computational consistency,
not accuracy against real Trainium hardware.

## Primary sources

- [Trainium2 chip resources](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html)
- [Trainium2 topology and bandwidth table](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html)
- [Trainium3 chip resources](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html)
- [Trainium3 domain configurations and conflicting bandwidth descriptions](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html)
- [Trainium3 MX formats and Tensor Engine](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium3_arch.html)
- [NeuronCore-v3 Tensor Engine](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html)

Sources rechecked on 22 October 2026. Numerical assumptions are distinguished
from published rates above; no vendor-provided Trainium Inference Capacity
Index score exists in these sources.
