> Historical exploratory study. See [scope and snapshot notes](../README.md) before using these projections.

# Controlled 5T Kimi K3 versus DeepSeek V4 Pro SOL comparison

AIC SOL, SGLang, 256K resident-cache decode-only, FP8 KV, no speculative decoding, TP1/PP1/attention-DP=DEP/MoE-EP=DEP.

## Variant definitions

| Variant | Experts | top-k | k/E | Total | Routed pool | Routed active | All active | Active/total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi ratio-preserved | 1600 | 43 | 2.688% | 4.919T | 4.862T | 130.7B | 187.9B | 3.819% |
| DeepSeek ratio-preserved | 1216 | 31 | 2.549% | 4.953T | 4.900T | 124.9B | 150.1B | 3.031% |
| Kimi MoE-normalized | 1600 | 33 | 2.062% | 4.919T | 4.862T | 100.3B | 157.5B | 3.201% |
| DeepSeek MoE-normalized | 1216 | 25 | 2.056% | 4.953T | 4.900T | 100.7B | 125.9B | 2.543% |

The ratio-preserved pair uses the nearest integer top-k to each baseline active/total ratio: Kimi 3.790% and DeepSeek 3.062%. The MoE-normalized pair preserves native layer counts and expert tensor shapes but matches aggregate routed capacity, selection fraction, and selected routed weights. This is an aggregate MoE control, not literally identical per-layer tensors.

If sparsity instead means preserving each baseline's top-k/expert-count ratio, the nearest 5T choices are Kimi top-29/1600 (145.3B all-active) and DeepSeek top-19/1216 (101.7B all-active). That does not preserve active/total ratios because the fixed non-routed parts of each model were not scaled.

## Constant original active/total ratio: matched concurrency/rank

| System | DEP | c/r | Kimi TPS/user | DeepSeek TPS/user | DS/Kimi | Kimi GiB | DS GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GB300 | 16 | 1 | 43.9 | 89.2 | 2.03x | 257.8 | 178.5 |
| GB300 | 16 | 2 | 31.7 | 52.4 | 1.65x | 263.3 | 179.6 |
| GB300 | 16 | 3 | 28.7 | 44.1 | 1.54x | 268.7 | 180.7 |
| GB300 | 16 | 4 | 28.2 | 44.0 | 1.56x | 274.2 | 181.8 |
| GB300 | 64 | 1 | 51.5 | 120.8 | 2.35x | 151.7 | 71.5 |
| GB300 | 64 | 2 | 49.8 | 118.6 | 2.38x | 157.1 | 72.6 |
| GB300 | 64 | 3 | 48.1 | 116.5 | 2.42x | 162.6 | 73.7 |
| GB300 | 64 | 4 | 46.6 | 114.5 | 2.46x | 168.1 | 74.8 |
| B300 | 16 | 1 | 42.2 | 84.8 | 2.01x | 257.8 | 178.5 |
| B300 | 16 | 2 | 30.4 | 49.6 | 1.63x | 263.3 | 179.6 |
| B300 | 16 | 3 | OOM | 41.5 | — | 268.7 | 180.7 |
| B300 | 16 | 4 | OOM | 41.0 | — | 274.2 | 181.8 |
| B300 | 64 | 1 | 48.1 | 105.1 | 2.18x | 151.7 | 71.5 |
| B300 | 64 | 2 | 45.0 | 93.9 | 2.09x | 157.1 | 72.6 |
| B300 | 64 | 3 | 42.3 | 84.9 | 2.01x | 162.6 | 73.7 |
| B300 | 64 | 4 | 39.9 | 77.4 | 1.94x | 168.1 | 74.8 |

## MoE-normalized: matched concurrency/rank

| System | DEP | c/r | Kimi TPS/user | DeepSeek TPS/user | DS/Kimi | Kimi GiB | DS GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GB300 | 16 | 1 | 47.9 | 103.2 | 2.15x | 257.8 | 178.5 |
| GB300 | 16 | 2 | 36.0 | 62.3 | 1.73x | 263.3 | 179.6 |
| GB300 | 16 | 3 | 28.9 | 44.6 | 1.54x | 268.7 | 180.7 |
| GB300 | 16 | 4 | 28.2 | 44.0 | 1.56x | 274.2 | 181.8 |
| GB300 | 64 | 1 | 51.5 | 120.8 | 2.35x | 151.7 | 71.5 |
| GB300 | 64 | 2 | 49.8 | 118.7 | 2.38x | 157.1 | 72.6 |
| GB300 | 64 | 3 | 48.2 | 116.6 | 2.42x | 162.6 | 73.7 |
| GB300 | 64 | 4 | 46.6 | 114.5 | 2.46x | 168.1 | 74.8 |
| B300 | 16 | 1 | 46.0 | 97.7 | 2.12x | 257.8 | 178.5 |
| B300 | 16 | 2 | 34.5 | 58.7 | 1.70x | 263.3 | 179.6 |
| B300 | 16 | 3 | OOM | 42.0 | — | 268.7 | 180.7 |
| B300 | 16 | 4 | OOM | 41.0 | — | 274.2 | 181.8 |
| B300 | 64 | 1 | 48.1 | 105.1 | 2.18x | 151.7 | 71.5 |
| B300 | 64 | 2 | 45.0 | 93.9 | 2.09x | 157.1 | 72.6 |
| B300 | 64 | 3 | 42.3 | 84.9 | 2.01x | 162.6 | 73.7 |
| B300 | 64 | 4 | 39.9 | 77.4 | 1.94x | 168.1 | 74.8 |

## GB300 DEP16 c/r=1 latency decomposition

| Control | Model | TPOT ms | Routed MoE ms | Everything else ms | Attention bucket ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant original active/total ratio | Kimi | 22.77 | 8.17 | 14.60 | 0.45 |
| Constant original active/total ratio | DeepSeek | 11.21 | 7.97 | 3.23 | 2.96 |
| MoE-normalized | Kimi | 20.87 | 6.27 | 14.60 | 0.45 |
| MoE-normalized | DeepSeek | 9.69 | 6.46 | 3.23 | 2.96 |

For Kimi, AIC reports the KDA/MLA projection GEMMs separately from the attention bucket; therefore the attention-bucket column must not be treated as the whole attention-stack cost. In the MoE-normalized control, routed MoE time is nearly equal (6.27 versus 6.46 ms), while the remaining stack is 14.60 versus 3.23 ms.

## Maximum memory-fitting concurrency

| Control | System | DEP | K c/r | K TPS/u | K TPS/GPU | DS c/r | DS TPS/u | DS TPS/GPU | DS/K throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant original active/total ratio | GB300 | 16 | 4 | 28.2 | 113 | 89 | 36.1 | 3209 | 28.40x |
| Constant original active/total ratio | GB300 | 64 | 24 | 28.7 | 688 | 185 | 24.8 | 4596 | 6.68x |
| Constant original active/total ratio | B300 | 16 | 2 | 30.4 | 61 | 81 | 21.3 | 1725 | 28.39x |
| Constant original active/total ratio | B300 | 64 | 22 | 19.7 | 433 | 177 | 4.7 | 830 | 1.92x |
| MoE-normalized | GB300 | 16 | 4 | 28.2 | 113 | 89 | 36.1 | 3214 | 28.45x |
| MoE-normalized | GB300 | 64 | 24 | 28.7 | 688 | 185 | 26.0 | 4810 | 6.99x |
| MoE-normalized | B300 | 16 | 2 | 34.5 | 69 | 81 | 21.3 | 1727 | 25.03x |
| MoE-normalized | B300 | 64 | 22 | 19.7 | 433 | 177 | 4.7 | 837 | 1.93x |

SOL is a hardware/model lower-bound projection, not achieved serving performance. OOM boundaries use raw physical capacity with no operational headroom.
