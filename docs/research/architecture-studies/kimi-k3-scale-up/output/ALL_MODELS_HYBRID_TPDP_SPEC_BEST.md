> Historical exploratory study. See [scope and snapshot notes](../README.md) before using these projections.

# Best hybrid TP×DP + DSpark configurations across scaled Kimi models

AIC formula-only SGLang SOL at 256K resident context on the hypothetical 288-GPU NVLink domain. Global request concurrency equals GPU count, BF16 KV is resident, and every model is assumed to retain K3's measured InferenceX SPEED-Bench DSpark acceptance curve. Depth-only variants are excluded.

## Model definitions

| Model | Scaling direction | Params | Active/token | Layers | Experts/layer | Top-k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| K3 2.8T baseline | K3 baseline | 2.78T | 105B | 93 | 896 | 16 |
| ~5T expert-only | More experts only | 4.92T | 106B | 93 | 1600 | 16 |
| ~5T Kimi-trend | Deeper + more/larger active experts | 4.96T | 184B | 116 | 1280 | 24 |
| ~10T expert-only | More experts only | 9.78T | 107B | 93 | 3200 | 16 |
| ~10T Kimi-trend | Deeper + more/larger active experts | 9.84T | 314B | 144 | 2048 | 32 |

## Jointly optimized configurations

| Model | GPUs / MoE EP | Global concurrency | Concurrency/rank | Best attention layout | Best nextn | AL | Best no-spec TPS/user (layout) | Spec TPS/user | Gain | TPS/GPU | Aggregate TPS | GiB/rank |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| K3 2.8T baseline | 64 | 64 | 2 | TP2×DP32 | 3 | 3.00 | 90.2 (TP4×DP16) | 146.0 | 1.62× | 146.0 | 9342 | 103.6 |
| K3 2.8T baseline | 128 | 128 | 2 | TP2×DP64 | 3 | 3.00 | 101.5 (TP4×DP32) | 153.1 | 1.51× | 153.1 | 19593 | 93.7 |
| K3 2.8T baseline | 256 | 256 | 2 | TP2×DP128 | 3 | 3.00 | 107.8 (TP4×DP64) | 153.1 | 1.42× | 153.1 | 39204 | 88.7 |
| ~5T expert-only | 64 | 64 | 2 | TP2×DP32 | 4 | 3.36 | 86.3 (TP4×DP16) | 132.2 | 1.53× | 132.2 | 8460 | 120.1 |
| ~5T expert-only | 128 | 128 | 2 | TP2×DP64 | 3 | 3.00 | 91.6 (TP4×DP32) | 145.2 | 1.58× | 145.2 | 18583 | 102.3 |
| ~5T expert-only | 256 | 256 | 2 | TP2×DP128 | 3 | 3.00 | 100.3 (TP4×DP64) | 148.0 | 1.47× | 148.0 | 37881 | 93.5 |
| ~5T Kimi-trend | 64 | 64 | 2 | TP2×DP32 | 4 | 3.36 | 60.0 (TP4×DP16) | 103.8 | 1.73× | 103.8 | 6641 | 159.2 |
| ~5T Kimi-trend | 128 | 128 | 2 | TP2×DP64 | 4 | 3.36 | 69.4 (TP4×DP32) | 109.8 | 1.58× | 109.8 | 14058 | 141.5 |
| ~5T Kimi-trend | 256 | 256 | 2 | TP2×DP128 | 3 | 3.00 | 74.3 (TP4×DP64) | 110.6 | 1.49× | 110.6 | 28322 | 132.6 |
| ~10T expert-only | 64 | 64 | 2 | TP2×DP32 | 4 | 3.36 | 84.4 (TP4×DP16) | 110.4 | 1.31× | 110.4 | 7066 | 157.4 |
| ~10T expert-only | 128 | 128 | 2 | TP2×DP64 | 4 | 3.36 | 83.7 (TP4×DP32) | 128.5 | 1.53× | 128.5 | 16446 | 122.1 |
| ~10T expert-only | 256 | 256 | 2 | TP2×DP128 | 3 | 3.00 | 88.0 (TP4×DP64) | 138.4 | 1.57× | 138.4 | 35428 | 104.3 |
| ~10T Kimi-trend | 64 | 64 | 2 | TP2×DP32 | 5 | 3.62 | 36.4 (TP4×DP16) | 68.0 | 1.87× | 68.0 | 4355 | 268.4 |
| ~10T Kimi-trend | 128 | 128 | 2 | TP2×DP64 | 4 | 3.36 | 43.7 (TP4×DP32) | 73.9 | 1.69× | 73.9 | 9464 | 233.1 |
| ~10T Kimi-trend | 256 | 256 | 2 | TP2×DP128 | 4 | 3.36 | 48.0 (TP4×DP64) | 75.4 | 1.57× | 75.4 | 19312 | 215.5 |

`TPS/GPU` equals `TPS/user` in this comparison because global concurrency equals GPU count. The winning TP/DP split and speculative depth are selected jointly after excluding OOM configurations.

The fixed AL assumption is intentional for this scenario, but every result remains an uncalibrated custom-model SOL ceiling. It excludes prefill, KV transfer, queueing, scheduler overhead, and runtime efficiency loss.

Raw data: [`all_models_hybrid_tpdp_spec_sweep.csv`](../data.zip); metadata: [`all_models_hybrid_tpdp_spec_metadata.json`](all_models_hybrid_tpdp_spec_metadata.json).
