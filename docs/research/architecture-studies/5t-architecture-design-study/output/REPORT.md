> Historical exploratory study. See [scope and snapshot notes](../README.md) before using these projections.

# Controlled 5T architecture design-space study

Fixed-SLO follow-ups: [dense no-spec 50/100 TPS-user concurrency analysis](SLO_CONCURRENCY_REPORT.md)
and [spec-aware 50/100 TPS-user concurrency analysis](SLO_SPEC_CONCURRENCY_REPORT.md).
The [calibrated achievable-concurrency sweep](ACHIEVABLE_CONCURRENCY_REPORT.md)
maps 0.4x/0.5x/0.7x SOL efficiency to achieved 50/100 TPS-user targets.
The [latency-versus-concurrency curves](LATENCY_CONCURRENCY_CURVES.md) visualize
six key configurations across hardware and calibration assumptions.

## Scope and interpretation

This is an all-else-equal AIC SOL study for resident-cache decode. The primary point is 256K; 32K and 1M are context sensitivities. There is no prefill, scheduler, queueing, or framework overhead. No-spec results are the baseline; a separate speculative sensitivity applies the same measured Kimi-K3 DSpark acceptance curve to every design. Synthetic attention mixtures are operation-level composites under AIC's additive SOL model; they are not runnable checkpoints and AIC does not predict model quality.

All bodies are approximately 5T, hidden-size 7168, FP4 routed experts, FP8 non-expert weights, FP8 KV, TP1/PP1, and attention-DP=MoE-EP. The hardware cases are GB300; GB300 compute/fabric with 22-TB/s VR200 HBM; and the same with directional NVLink doubled to 1.8 TB/s.

## Executive findings

- The best balanced point is a 61-layer direct-MoE body with DeepSeek-V4 compressed attention: 120.9 TPS/user on GB300 and 324.0 on the VR200-HBM substitution at DEP64, c/r=1, 256K.
- A more conservative 78-layer/high-active design reaches 108.8 TPS/user. The 216.8-TPS/user DSA result is an optimistic quality and implementation frontier, not the central recommendation.
- Under controlled size and active-work budgets, depth is the strongest architectural penalty. Native Kimi-trend scaling is slow primarily because it grows to 116 sequential layers; sparse attention alone does not erase that cost.
- V4 compressed attention gives the strongest combined latency/capacity result. DSA gives the lowest modeled attention latency, while KDA gives the flattest context scaling; both involve larger quality or implementation uncertainty when used for most layers.
- The 22/8 HBM-bandwidth substitution yields about 2.68x on the balanced design. Doubling directional fabric adds only 1.02x at c/r=1 but 1.08x at c/r=4.
- With the common DSpark AL curve, the spec-aware balanced frontier switches to 100% DSA top-1024 at c/r=1 (405.8 TPS/user, nextn=5); V4 compressed attention remains best at c/r=4 (284.7, nextn=5).

## Recommended architectures

| Category | Tier | L | MoE | Active tier | Attention | Active | GB c1 | GB c4 | VR-HBM c1 | 1M retention | GiB | 90% max c/r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggressive latency floor | aggressive | 48 | direct | aggressive | 100% DSA top-512 | 64B | 216.8 | 149.7 | 575.1 | 95% | 63.9 | 21 |
| Balanced recommendation | balanced | 61 | direct | balanced | 100% DeepSeek-V4 compressed | 131B | 120.9 | 114.6 | 324.0 | 99% | 71.5 | 167 |
| Conservative recommendation | conservative | 78 | direct | conservative | 100% DeepSeek-V4 compressed | 197B | 108.8 | 102.3 | 290.5 | 99% | 79.4 | 122 |
| KDA+sparse hybrid | balanced | 61 | direct | balanced | 50% KDA + 50% V4 compressed | 132B | 117.4 | 110.0 | 314.8 | 100% | 73.5 | 121 |
| Latent-MoE option | balanced | 61 | latent | balanced | 100% DeepSeek-V4 compressed | 134B | 98.6 | 96.0 | 268.4 | 99% | 81.7 | 158 |
| Full-attention fallback | conservative | 78 | direct | conservative | 75% KDA + 25% MLA | 198B | 103.7 | 86.9 | 271.5 | 87% | 82.9 | 38 |

TPS columns are TPS/user at DEP64 and 256K. The aggressive row is a hardware floor, not a quality recommendation. The 90% maximum is an extrapolated capacity boundary, not a latency optimum.

## Native scaled-architecture anchors

| Anchor | Family | L | Total | Active | GB c1 | GB c4 | VR c1 | 1M retention | GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek ratio-preserved 5T | DeepSeek V4 | 61 | 4.953T | 150B | 120.8 | 114.5 | 323.9 | 99% | 71.5 |
| GLM-5.2 ratio-preserved 5T | GLM-5.2 | 78 | 5.000T | 160B | 113.3 | 97.2 | 303.5 | 87% | 90.4 |
| Kimi ratio-preserved 5T | Kimi K3 | 93 | 4.919T | 188B | 67.5 | 59.4 | 180.4 | 89% | 117.3 |
| Kimi-trend 5T | Kimi K3 | 116 | 4.955T | 184B | 47.6 | 42.4 | 126.0 | 91% | 164.1 |

These are direct AIC executions of structurally scaled native families under the same quantization and workload assumptions. They are comparison anchors, not members of the all-else-equal synthetic search; family-specific hidden sizes, MoE layouts, and attention stacks remain intact.

## Speculative decoding with the common DSpark AL curve

| nextn | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| InferenceX golden AL | 1.00 | 1.85 | 2.51 | 3.00 | 3.36 | 3.62 | 3.75 | 3.84 | 4.00 |

AL includes the guaranteed target token, so AIC receives `nextn_accepted = AL - 1`. For every architecture/workload point, the selected row is the fastest memory-fitting depth among no speculation and nextn 1–8.

### Recommended designs on GB300, DEP64, 256K

| Design | c/r | No spec | Best nextn | AL | Spec TPS/user | Gain | GiB/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggressive latency floor | 1 | 216.8 | 5 | 3.62 | 436.1 | 2.01x | 64.4 |
| Aggressive latency floor | 2 | 157.3 | 4 | 3.36 | 370.9 | 2.36x | 73.4 |
| Aggressive latency floor | 4 | 149.7 | 4 | 3.36 | 299.6 | 2.00x | 91.5 |
| Aggressive latency floor | 8 | 136.6 | 3 | 3.00 | 229.5 | 1.68x | 127.5 |
| Balanced recommendation | 1 | 120.9 | 8 | 4.00 | 372.9 | 3.08x | 72.3 |
| Balanced recommendation | 2 | 118.7 | 5 | 3.62 | 336.7 | 2.84x | 73.0 |
| Balanced recommendation | 4 | 114.6 | 5 | 3.62 | 284.7 | 2.48x | 75.2 |
| Balanced recommendation | 8 | 107.2 | 4 | 3.36 | 222.5 | 2.08x | 79.4 |
| Conservative recommendation | 1 | 108.8 | 8 | 4.00 | 337.5 | 3.10x | 80.2 |
| Conservative recommendation | 2 | 106.5 | 5 | 3.62 | 300.1 | 2.82x | 81.3 |
| Conservative recommendation | 4 | 102.3 | 5 | 3.62 | 248.9 | 2.43x | 84.1 |
| Conservative recommendation | 8 | 94.7 | 4 | 3.36 | 190.4 | 2.01x | 89.6 |
| KDA+sparse hybrid | 1 | 117.4 | 8 | 4.00 | 367.7 | 3.13x | 74.3 |
| KDA+sparse hybrid | 2 | 114.8 | 5 | 3.62 | 327.6 | 2.85x | 75.5 |
| KDA+sparse hybrid | 4 | 110.0 | 5 | 3.62 | 273.7 | 2.49x | 78.4 |
| KDA+sparse hybrid | 8 | 101.5 | 4 | 3.36 | 210.5 | 2.07x | 84.2 |
| Latent-MoE option | 1 | 98.6 | 5 | 3.62 | 283.4 | 2.87x | 84.7 |
| Latent-MoE option | 2 | 97.8 | 5 | 3.62 | 241.0 | 2.47x | 88.3 |
| Latent-MoE option | 4 | 96.0 | 4 | 3.36 | 189.5 | 1.97x | 95.3 |
| Latent-MoE option | 8 | 92.7 | 3 | 3.00 | 139.6 | 1.51x | 109.5 |
| Full-attention fallback | 1 | 103.7 | 5 | 3.62 | 280.5 | 2.70x | 83.4 |
| Full-attention fallback | 2 | 97.5 | 4 | 3.36 | 221.0 | 2.27x | 87.8 |
| Full-attention fallback | 4 | 86.9 | 3 | 3.00 | 160.5 | 1.85x | 96.6 |
| Full-attention fallback | 8 | 71.5 | 2 | 2.51 | 107.2 | 1.50x | 114.3 |

### Speculative performance frontiers

| Tier | c/r | L | MoE | Attention | Active | nextn | AL | Spec TPS/user | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 1 | 48 | direct | 100% DSA top-512 | 64B | 5 | 3.62 | 436.1 | 2.01x |
| Balanced | 1 | 61 | direct | 100% DSA top-1024 | 117B | 5 | 3.62 | 405.8 | 2.65x |
| Conservative | 1 | 78 | direct | 100% DSA top-2048 | 180B | 5 | 3.62 | 369.4 | 2.57x |
| Overall | 2 | 48 | direct | 100% DSA top-512 | 64B | 4 | 3.36 | 370.9 | 2.36x |
| Balanced | 2 | 61 | direct | 100% DSA top-1024 | 117B | 4 | 3.36 | 338.3 | 2.27x |
| Conservative | 2 | 78 | direct | 100% DSA top-2048 | 180B | 4 | 3.36 | 301.6 | 2.18x |
| Overall | 4 | 48 | direct | 100% DeepSeek-V4 compressed | 75B | 5 | 3.62 | 318.2 | 2.52x |
| Balanced | 4 | 61 | direct | 100% DeepSeek-V4 compressed | 131B | 5 | 3.62 | 284.7 | 2.48x |
| Conservative | 4 | 78 | direct | 100% DeepSeek-V4 compressed | 197B | 5 | 3.62 | 248.9 | 2.43x |
| Overall | 8 | 48 | direct | 100% DeepSeek-V4 compressed | 75B | 4 | 3.36 | 254.2 | 2.14x |
| Balanced | 8 | 61 | direct | 100% DeepSeek-V4 compressed | 131B | 4 | 3.36 | 222.5 | 2.08x |
| Conservative | 8 | 78 | direct | 100% DeepSeek-V4 compressed | 197B | 4 | 3.36 | 190.4 | 2.01x |

Speculation changes the architecture ordering: shared-index DSA is the modeled low-c/r frontier, while V4 compressed attention wins at c/r=4–8 because wider verification penalizes DSA's indexer and sparse-attention work more strongly. The DSA frontier retains the implementation and quality caveats of the non-spec study.

### Hardware sensitivity for the balanced and conservative designs

| Design | System | c/r | No spec | nextn | AL | Spec TPS/user | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced | GB300 | 1 | 120.9 | 8 | 4.00 | 372.9 | 3.08x |
| Balanced | GB300 | 4 | 114.6 | 5 | 3.62 | 284.7 | 2.48x |
| Balanced | VR200 HBM | 1 | 324.0 | 5 | 3.62 | 892.0 | 2.75x |
| Balanced | VR200 HBM | 4 | 286.9 | 3 | 3.00 | 562.8 | 1.96x |
| Balanced | VR200 HBM+NVLink | 1 | 330.6 | 5 | 3.62 | 988.8 | 2.99x |
| Balanced | VR200 HBM+NVLink | 4 | 308.6 | 4 | 3.36 | 718.6 | 2.33x |
| Conservative | GB300 | 1 | 108.8 | 8 | 4.00 | 337.5 | 3.10x |
| Conservative | GB300 | 4 | 102.3 | 5 | 3.62 | 248.9 | 2.43x |
| Conservative | VR200 HBM | 1 | 290.5 | 5 | 3.62 | 792.1 | 2.73x |
| Conservative | VR200 HBM | 4 | 252.9 | 3 | 3.00 | 481.6 | 1.90x |
| Conservative | VR200 HBM+NVLink | 1 | 297.2 | 5 | 3.62 | 889.4 | 2.99x |
| Conservative | VR200 HBM+NVLink | 4 | 274.7 | 4 | 3.36 | 624.9 | 2.28x |

### Native-family anchors with the same AL assumption

| Anchor | c/r | No spec | nextn | AL | Spec TPS/user | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek ratio-preserved 5T | 1 | 120.8 | 8 | 4.00 | 372.4 | 3.08x |
| DeepSeek ratio-preserved 5T | 4 | 114.5 | 5 | 3.62 | 284.0 | 2.48x |
| GLM-5.2 ratio-preserved 5T | 1 | 113.3 | 5 | 3.62 | 301.9 | 2.66x |
| GLM-5.2 ratio-preserved 5T | 4 | 97.2 | 3 | 3.00 | 178.8 | 1.84x |
| Kimi ratio-preserved 5T | 1 | 67.5 | 5 | 3.62 | 182.6 | 2.70x |
| Kimi ratio-preserved 5T | 4 | 59.4 | 3 | 3.00 | 105.2 | 1.77x |
| Kimi-trend 5T | 1 | 47.6 | 5 | 3.62 | 135.0 | 2.84x |
| Kimi-trend 5T | 4 | 42.4 | 4 | 3.36 | 80.6 | 1.90x |

The AL curve is measured Kimi-K3 SPEED-Bench coding data from InferenceX and is deliberately held constant here. It is not evidence that another architecture or a scaled 5T checkpoint would achieve the same acceptance. Native Kimi bodies use AIC's K3/DSpark verification path; direct DeepSeek-style bodies use AIC's generic target-verification/MTP geometry while borrowing the DSpark AL. Therefore relative speculative gains outside Kimi-like bodies are sensitivity bounds, not implementation forecasts.

## What controls performance

| Ablation | Active | GB c1 | GB c4 | VR HBM gain |
| --- | ---: | ---: | ---: | ---: |
| Direct active=aggressive | 89B | 131.9 | 102.4 | 2.65x |
| Direct active=balanced | 136B | 108.8 | 102.3 | 2.67x |
| Direct active=conservative | 197B | 108.8 | 102.3 | 2.67x |
| Direct MoE | 136B | 108.8 | 102.3 | 2.67x |
| Latent MoE | 143B | 86.6 | 84.1 | 2.72x |

Lower routed-active weight budgets reduce single-token weight traffic but increase quality risk. More depth adds sequential attention, routing, normalization, and projection work even when total and active expert weights are held approximately fixed. Latent MoE halves routed activation/dispatch width but adds replicated hidden↔latent projections and a larger router/expert pool.

### Expert granularity and shared experts

| Expert width | Experts | top-k | Active | GB c1 | GB c4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1408 | 29 | 136B | 109.3 | 102.8 |
| 3072 | 960 | 19 | 136B | 108.8 | 102.3 |
| 4096 | 704 | 15 | 142B | 110.5 | 103.9 |

| Latent shared experts | Routed experts | top-k | Active | GB c1 | GB c4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1920 | 39 | 138B | 97.5 | 94.3 |
| 1 | 1920 | 39 | 143B | 86.6 | 84.1 |
| 2 | 1920 | 39 | 148B | 77.9 | 75.9 |

Narrower experts require more expert selections for the same active budget; wider experts reduce routing fan-out but make each selected expert coarser. Shared experts add dense per-token work on every layer, so their quality/stability benefit has a direct decode cost.

## Attention tradeoffs on the common 78-layer balanced direct body

| Attention | Risk | GB c1 | GB c4 | 1M retention | VR gain | GiB | Attn share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% DSA top-512 | aggressive | 144.1 | 129.9 | 95% | 2.65x | 76.8 | 23% |
| 100% DSA top-1024 | balanced | 144.0 | 129.7 | 95% | 2.65x | 76.8 | 23% |
| 100% DSA top-2048 | conservative | 143.9 | 129.3 | 95% | 2.65x | 76.8 | 23% |
| 100% DSA top-4096 | conservative | 143.7 | 128.5 | 95% | 2.65x | 76.8 | 23% |
| 100% DSA top-2048, per-layer index | conservative | 137.7 | 113.8 | 84% | 2.65x | 77.3 | 26% |
| 25% KDA + 75% DSA-2048 | balanced | 130.1 | 117.7 | 97% | 2.66x | 78.8 | 30% |
| 50% KDA + 50% DSA-2048 | balanced | 119.2 | 108.3 | 98% | 2.66x | 80.7 | 36% |
| 100% full MLA | conservative | 109.9 | 71.4 | 58% | 2.44x | 77.8 | 41% |
| 75% KDA + 25% DSA-2048 | balanced | 109.6 | 100.0 | 99% | 2.67x | 82.7 | 41% |
| 100% DeepSeek-V4 compressed | conservative | 108.8 | 102.3 | 99% | 2.67x | 79.4 | 42% |
| 50% KDA + 50% MLA | conservative | 105.8 | 81.0 | 76% | 2.56x | 81.2 | 43% |
| 50% KDA + 50% V4 compressed | balanced | 105.2 | 97.8 | 99% | 2.67x | 82.0 | 44% |
| 75% KDA + 25% MLA | conservative | 103.7 | 87.0 | 87% | 2.62x | 82.9 | 44% |
| 75% KDA + 25% V4 compressed | balanced | 103.5 | 95.6 | 100% | 2.67x | 83.3 | 45% |
| 90% KDA + 10% MLA | balanced | 102.7 | 90.7 | 94% | 2.65x | 83.8 | 45% |
| 100% KDA | aggressive | 101.9 | 93.6 | 100% | 2.68x | 84.5 | 45% |

KDA has context-independent recurrent state but pays dense projection cost every layer. MLA stores a compact latent KV cache but scans the full context. DSA reads only selected KV entries during attention, yet stores full latent KV plus its indexer cache; smaller DSA top-k therefore improves latency more than capacity. The shared-index DSA rows assume GLM-style index reuse every four layers. The per-layer-index row removes that reuse and is the conservative control. DeepSeek-V4 compressed attention reduces both reads and stored cache, but relies on a more novel compressor/indexer stack. Mixed KDA+sparse designs trade a bounded recurrent path against explicit token retrieval.

### Attention architecture decision matrix

| Mechanism | Decode context work | Stored state | Performance upside | Principal tradeoff |
| --- | ---: | ---: | ---: | ---: |
| MLA | Full latent-cache scan | Compact KV/token | Mature full retrieval | TPS falls with context |
| KDA | Fixed recurrent update | Fixed recurrent state | Flat long-context cost | Dense projections; retrieval/quality risk |
| DSA | Indexer scan + top-k KV | Full latent KV + index keys | Fastest modeled sparse read | Indexer/reuse kernels and selection quality |
| V4 compressed | Multi-rate compressed scan | Window + compressed histories | Best latency/capacity balance | Novel compressor/indexer stack |
| KDA+sparse | Bounded path + sparse retrieval | Recurrent + selected cache | Robust hybrid failure modes | Two mechanisms to train and optimize |

The study holds model quality out of scope. In particular, an O(1) or top-k attention path is only a performance win if training preserves the useful long-range information that dense attention would have retrieved.

## Depth scaling at balanced active work and V4 attention

| Layers | Experts | top-k | Total | Active | GB c1 | GB c4 | VR c1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 1536 | 32 | 4.895T | 126B | 131.9 | 126.0 | 355.0 |
| 61 | 1216 | 25 | 4.930T | 131B | 120.9 | 114.6 | 324.0 |
| 78 | 960 | 19 | 4.984T | 136B | 108.8 | 102.3 | 290.5 |
| 93 | 832 | 17 | 4.943T | 144B | 100.8 | 94.3 | 268.5 |
| 116 | 640 | 13 | 4.959T | 154B | 90.1 | 83.6 | 239.0 |

Shallower designs win performance because less non-expert work is sequential. The tradeoff is architectural capacity: increasing experts can preserve total and active parameter budgets, but it does not replace additional transformation depth.

## Context-length sensitivity

| Design | 32K c1 | 32K c4 | 256K c1 | 256K c4 | 1M c1 | 1M c4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggressive latency floor | 219.3 | 154.6 | 216.8 | 149.7 | 208.6 | 135.0 |
| Balanced recommendation | 121.1 | 115.5 | 120.9 | 114.6 | 120.0 | 111.6 |
| Conservative recommendation | 109.0 | 103.2 | 108.8 | 102.3 | 107.9 | 99.2 |
| Full-attention fallback | 107.2 | 97.6 | 103.7 | 86.9 | 93.3 | 63.3 |

MLA's cache scan becomes conspicuous at 1M and higher c/r. DSA still scans index keys but limits expensive KV attention to top-k tokens; V4 compresses older history; KDA avoids token-cache reads entirely. Context robustness and memory capacity are separate: native DSA stores a full latent cache even when it reads only top-k values.

## Parallelism sensitivity

| Design | DEP | c/r | Global conc. | TPS/user | TPS/GPU | GiB/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced | 16 | 1 | 16 | 103.2 | 103.2 | 178.4 |
| Balanced | 16 | 4 | 64 | 44.0 | 176.1 | 181.6 |
| Balanced | 32 | 1 | 32 | 102.9 | 102.9 | 107.1 |
| Balanced | 32 | 4 | 128 | 75.4 | 301.6 | 110.3 |
| Balanced | 64 | 1 | 64 | 120.9 | 120.9 | 71.5 |
| Balanced | 64 | 4 | 256 | 114.6 | 458.3 | 74.7 |
| Conservative | 16 | 1 | 16 | 70.3 | 70.3 | 187.4 |
| Conservative | 16 | 4 | 64 | 42.0 | 168.1 | 191.6 |
| Conservative | 32 | 1 | 32 | 71.7 | 71.7 | 115.4 |
| Conservative | 32 | 4 | 128 | 70.0 | 279.9 | 119.6 |
| Conservative | 64 | 1 | 64 | 108.8 | 108.8 | 79.4 |
| Conservative | 64 | 4 | 256 | 102.3 | 409.1 | 83.6 |

Increasing DEP shards routed experts and their weights over more ranks, but it does not remove sequential attention, projections, routing, normalization, or communication. Raising c/r improves weight reuse and aggregate TPS/GPU until KV/state traffic, activation work, or communication dominates; TPS/user usually falls.

## Rubin sensitivity

| Architecture | GB c1 | VR-HBM c1 | HBM gain | +fabric c1 | Fabric gain | VR-HBM c4 | +fabric c4 | Fabric gain c4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggressive latency floor | 216.8 | 575.1 | 2.65x | 591.5 | 1.03x | 374.0 | 403.0 | 1.08x |
| Balanced recommendation | 120.9 | 324.0 | 2.68x | 330.6 | 1.02x | 286.9 | 308.6 | 1.08x |
| Conservative recommendation | 108.8 | 290.5 | 2.67x | 297.2 | 1.02x | 252.9 | 274.7 | 1.09x |
| KDA+sparse hybrid | 117.4 | 314.8 | 2.68x | 321.0 | 1.02x | 276.4 | 296.5 | 1.07x |
| Latent-MoE option | 98.6 | 268.4 | 2.72x | 270.6 | 1.01x | 253.6 | 261.7 | 1.03x |
| Full-attention fallback | 103.7 | 271.5 | 2.62x | 277.4 | 1.02x | 204.4 | 218.4 | 1.07x |

HBM-heavy expert and projection work approaches the 22/8 bandwidth ratio. Architectures with larger attention-compute or recurrent-kernel shares gain less. Fabric matters more as c/r raises dispatch volume; these are still isolated specification substitutions, not a complete Rubin implementation model.

## Source provenance

- AIC model implementation: the local `repos/aiconfigurator-kimi-k3-pr1435` checkout; its exact commit is recorded in `metadata.json`.
- SGLang simulator feature context: [sglang PR #33824](https://github.com/sgl-project/sglang/pull/33824).
- GB300 hardware inputs: AIC's checked-in `aiconfigurator_core/systems/gb300.yaml`.
- VR200 hardware inputs: [NVIDIA Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) and [NVIDIA's Rubin architecture article](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/).
- Native-family anchor configs: the existing Kimi/DeepSeek scaling experiments plus AIC's checked-in NVIDIA GLM-5.2-NVFP4 config.
- Speculative acceptance: InferenceX `golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml`; exact repository commit is recorded in `metadata.json`.

## Method and limitations

- All nonzero direct body components must report SOL provenance; zero-time TP1 collectives may retain an empirical label.
- DSA top-k and shared-index variants extend AIC's published DSA SOL equations because the model parser fixes native GLM dimensions/top-2048. These rows are labeled in the raw provenance column.
- Native reconstruction validation passed: maximum no-spec Kimi/V4 latency error 0 ms, maximum speculative Kimi/V4 error 0 ms, DSA no-reuse error 1.29e-06 ms, and maximum memory-component error 0 bytes.
- Composite memory replaces native attention weights and KV using AIC's architecture formulas. Maximum c/r is a linear capacity estimate; use the 90% boundary operationally.
- Speculative Kimi-like composites preserve the additional DSpark verification/cache allocation reported by AIC; native-profile timing and memory reconstruct the direct AIC run exactly.
- VR200 cases retain GB300 tensor compute and kernel efficiencies. They isolate HBM and NVLink specifications and are not full Rubin performance predictions.
- Low active ratios, shallow depth, all-linear attention, and narrow sparse retrieval may reduce quality. No performance model can quantify that loss.
- SOL is a lower-bound hardware model. Achievable serving throughput will be lower and architecture-dependent.

## Artifacts

- `all_results.csv`: every architecture/workload/hardware result.
- `architecture_summary.csv`: one-row primary metrics per architecture.
- `pareto_front.csv`: non-dominated architectures.
- `body_manifest.csv`: exact generated MoE bodies.
- `native_anchors.csv`: direct native-family scaled-model comparisons.
- `spec_all_results.csv`: every depth 1–8 for every controlled architecture/workload point.
- `spec_best.csv`: fastest memory-fitting depth, including no-spec fallback.
- `spec_architecture_summary.csv`: primary GB300 speculative metrics per architecture.
- `native_anchor_spec_results.csv` and `native_anchor_spec_best.csv`: native-family speculative sensitivity.
- `estimate_cache.jsonl`: direct AIC evidence and operation breakdowns.
