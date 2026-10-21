# Inference Capacity Index v1: full explainer

These three interactive chapters are the detailed companion to the short
[methodology overview](https://docs.google.com/document/d/1w0XFlIYy276_eYZnJI-CLXsC8hPfNz0SBx8jaPrbra4).
The current score uses four equally weighted models, log 50–400 SOL TPS/user,
and workload-based provisioning of separate prefill and decode pools.

**20 October update:** the main explainer, index CSV and accelerator bars now
use the revised Rubin bandwidths and **2.177 GB300e** score. See the
[Rubin revision](../research/rubin-spec-update/README.md). Run
`python3 reproduce.py rubin-update` to regenerate these results. The existing
`v1.0.0` release download and no-argument rebuild remain the original snapshot
(Rubin 2.389); fixed-layout knee examples also retain their labeled old profile.

Download the repository or the release's `visualizations.zip`, keep its files
together, and open `REPORT.html` in a browser. GitHub shows HTML source rather
than executing it. No server or Python setup is needed to read the pages.
Expand the side notes and detail panels when you want the underlying assumptions.
HTML is the full explainer here: a static PDF would lose the configuration
explorer, model switching and interactive curve examples.

For empirical grounding, see the separate [InferenceX validation study](../research/inferencex-validation/README.md)
and [public interactive comparison](https://inferencex-sol-validation.epoch-3293.chatgpt.site).
It centers Kimi K3 and uses GLM's detailed logs for a secondary step-timing check.
Whole-service gaps are distinguished from matched execution efficiency.

For sensitivity to the choice of models, speed range, linear/log weighting and
prefill provisioning, see the [consolidated methodology sensitivity study](../research/methodology-sensitivity/README.md)
and its [interactive tables](../research/methodology-sensitivity/index.html).
Those comparisons reuse the same frozen simulations and leave the headline unchanged.

## 1. Understand the calculation

Start with [the methodology and configuration explorer](REPORT.html). Its six
steps follow one calculation all the way to the index:

1. **Define the workload.** Fix the existing context at 262,144 tokens, with
   architecture-appropriate cache already resident. Compare GLM-5.2, Kimi-K3,
   Qwen3-Coder 480B and the MAI-2 2.8T proxy. Side notes distinguish disclosed
   architecture details from study assumptions.
2. **Estimate execution time.** Follow the layer diagram through attention,
   expert routing, computation and communication. SOL estimates use hardware
   resource limits for the modeled work, not measured serving speed. The
   native speculative step includes verification and model-specific draft work;
   divide its time by the assumed three accepted output tokens.
3. **Find decode capacity.** At each speed threshold, choose the highest request
   density among the memory-admitted batches and layouts. Only allocated chips
   count. No qualifying configuration means zero capacity, not extrapolation.
4. **Provision input work.** Independently maximize prefill throughput for
   1,024 new tokens over the cached prefix. The selected coding-trace ratio is
   1.5133425 new input tokens per output token. Add prefill and decode chip
   requirements for that same workload, instead of imposing a fixed phase weight.
5. **Compare capacity curves.** Inspect the GB300/VR200 area comparison and
   move the speed control to see winning configurations, parallelism labels and
   memory breakdowns. Integrate the combined curve over log speed, 50–400;
   each speed doubling has equal horizontal weight.
6. **Combine architectures.** Normalize each model's area to its own GB300 area,
   then take the arithmetic mean of the four ratios. Read the component table:
   the basket score is not a universal hardware multiplier.

### The central resource equation

Let `C(s)` be decode requests per decode chip at speed floor `s`, `P` be the best
new-input tokens per prefill chip-second, and `r = 1.513342523752522`.

```text
Decode chips per request  = 1 / C(s)
Prefill chips per request = r × s / P
Combined requests/chip   = 1 / [1/C(s) + r × s/P]
```

For example, 100 decode chips plus 20 supporting prefill chips means dividing
the concurrent decoding requests by 120 total chips. It does not mean assigning
prefill a 20% score weight. The calculation assumes fully utilized, separate
pools of the same chip type, with fractional allocation at fleet scale. It is
not a simulation of queueing, mixed-batch contention or cache transfer.

The score is the area under this combined curve over `d ln(s)`, normalized per
model before averaging. Display sampling does not determine the score: integration
uses every saved speed breakpoint and the analytic expression between them.

## 2. Understand why the curves bend

Continue to [performance knees and bottlenecks](PERFORMANCE_KNEES.html). This
separates fixed-layout behavior from the optimized envelope. It explains why
straight-ish time-per-token curves become curved speed/capacity plots, how
expert-weight reads saturate, and how bandwidth, compute and memory capacity
produce different limits. Worked examples distinguish illustrative reasoning
from saved native results; a visual knee alone does not diagnose its cause.

## 3. Read the hardware comparison

Finish with [the accelerator comparison charts](ACCELERATOR_COMPARISON.html).
Read profile notes alongside the bars: roadmap chips, precision proxies and
network assumptions carry different uncertainty. Rated-power comparisons are
separate annotations, not part of the headline score. SVG and CSV downloads are
included beside the HTML pages.

## Audit or reproduce the result

| Question | Implementation / frozen evidence |
| --- | --- |
| What are the workload and hardware inputs? | [Frozen base](../../reference/2026-10-15/) plus [Rubin revision](../../reference/2026-10-20-rubin/) |
| How are native timing calls made? | [current_native.py](../../src/agent_index/current_native.py) |
| Where are the MAI adapter and model memory rules? | [current_models.py](../../src/agent_index/current_models.py) |
| Where are shared layouts and topology rules? | [simulation.py](../../src/agent_index/simulation.py), [topology.py](../../src/agent_index/topology.py) |
| How is the capacity envelope constructed? | [index.py](../../src/agent_index/index.py), functions `envelope` and `capacity` |
| How are prefill provisioning and AUC combined? | [log400_release.py](../../studies/log400_release.py), functions `combined_area` and `main` |
| How do I rebuild, replay or rerun the search? | [Repository instructions](../../README.md), [reproduce.py](../../reproduce.py) |

The current index uses analytical AISim, not an SGLang scheduling trace.
Replaying 9,200 saved timing cases and searching 718,289 candidates reproduces
the scores to floating-point precision. That establishes computational
consistency, not measured accuracy across every vendor. Actual efficiency,
expert balance, data reuse and communication can differ from the model.
The exact MAI implementation and equal model quality across precisions are
not established. The HTML expanders discuss these limitations, model-mixture
sensitivity and why versions must remain fixed for comparisons over time.

The older `docs/METHODOLOGY.*` files describe the historical two-model release;
this guide and the linked HTMLs describe the current four-model index.
