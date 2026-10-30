> Historical exploratory study. See [scope and snapshot notes](../README.md) before using these projections.

# Hypothetical 288-GPU NVLink-domain DEP sensitivity for 10T Kimi-trend

This keeps GB300 compute, 8-TB/s HBM, 277.6-GiB usable capacity, and 900-GB/s-per-GPU rack-local fabric unchanged while extending the rack-local domain from 72 to 288 GPUs. It is formula-only AIC SOL, not a proposed or real NVIDIA topology.

| DEP | c/r | Global | TPS/user | TPS/GPU | Aggregate TPS | TPOT | MoE | Current GiB/rank | FP8 non-routed estimate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 1 | 64 | 19.3 | 19.3 | 1232 | 51.93 ms | 9.45 ms | 393.7 (OOM) | 242.2 (fits) |
| 64 | 2 | 128 | 18.5 | 37.0 | 2369 | 54.03 ms | 9.46 ms | 410.4 (OOM) | 258.8 (fits) |
| 64 | 3 | 192 | 17.8 | 53.4 | 3421 | 56.13 ms | 9.46 ms | 427.0 (OOM) | 275.5 (fits) |
| 64 | 4 | 256 | 17.2 | 68.7 | 4396 | 58.23 ms | 9.47 ms | 443.7 (OOM) | 292.2 (OOM) |
| 128 | 1 | 128 | 21.1 | 21.1 | 2703 | 47.35 ms | 4.73 ms | 358.5 (OOM) | 207.0 (fits) |
| 128 | 2 | 256 | 20.2 | 40.3 | 5161 | 49.60 ms | 4.73 ms | 375.2 (OOM) | 223.6 (fits) |
| 128 | 3 | 384 | 19.3 | 57.9 | 7407 | 51.85 ms | 4.74 ms | 391.8 (OOM) | 240.3 (fits) |
| 128 | 4 | 512 | 18.5 | 73.9 | 9465 | 54.09 ms | 4.74 ms | 408.5 (OOM) | 257.0 (fits) |
| 256 | 1 | 256 | 22.1 | 22.1 | 5653 | 45.28 ms | 2.37 ms | 340.9 (OOM) | 189.4 (fits) |
| 256 | 2 | 512 | 20.9 | 41.8 | 10707 | 47.82 ms | 2.37 ms | 357.6 (OOM) | 206.0 (fits) |
| 256 | 3 | 768 | 19.9 | 59.6 | 15251 | 50.36 ms | 2.38 ms | 374.2 (OOM) | 222.7 (fits) |
| 256 | 4 | 1024 | 18.9 | 75.6 | 19359 | 52.90 ms | 2.38 ms | 390.9 (OOM) | 239.4 (fits) |

## Interpretation

At c/r=1, DEP128 improves TPS/user by 9.7% over DEP64; DEP256 improves it by 14.7%. Aggregate throughput grows much more because the cluster admits proportionally more requests.

The non-MoE floor is about 42.5 ms/token, so even free routed-MoE compute would cap this TP1 DEP design near 23.5 TPS/user. Wider EP cannot shard the active KDA/MLA, latent, shared-expert, dense, router, or output work.

All native-precision rows remain OOM because approximately 303 GiB of non-routed BF16 weights are replicated under TP1. The FP8 column is arithmetic only: it halves that component without changing timing. It is not a verified FP8-kernel simulation.

Raw data: [`kimi_k3_10t_nvl288_dep_sensitivity.csv`](../data.zip); metadata: [`kimi_k3_10t_nvl288_dep_metadata.json`](kimi_k3_10t_nvl288_dep_metadata.json).
