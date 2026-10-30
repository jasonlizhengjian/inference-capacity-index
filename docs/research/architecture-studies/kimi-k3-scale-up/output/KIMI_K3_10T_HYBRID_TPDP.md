> Historical exploratory study. See [scope and snapshot notes](../README.md) before using these projections.

# 10T Kimi-trend hybrid TP×DP projection

AIC formula-only SGLang SOL at 256K context on the hypothetical 288-GPU NVLink domain. Attention uses `TP × DP = world`; routed MoE uses `EP = world, MoE-TP = 1`. The TensorRT-LLM AIC backend currently rejects simultaneous attention TP>1 and DP>1.

## Equal global concurrency and GPU count

For each world size, local batch equals TP so global concurrency equals the number of GPUs. This isolates how GPUs are divided between TP and DP.

| GPUs | Attention layout | EP | Global | TPS/user | TPS/GPU | Aggregate TPS | GiB/rank | Fit |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | TP1×DP64 | 64 | 64 | 19.3 | 19.3 | 1232 | 394.4 | no |
| 64 | TP2×DP32 | 64 | 64 | 29.3 | 29.3 | 1873 | 262.9 | yes |
| 64 | TP4×DP16 | 64 | 64 | 36.4 | 36.4 | 2330 | 212.2 | yes |
| 64 | TP8×DP8 | 64 | 64 | 35.4 | 35.4 | 2267 | 217.2 | yes |
| 64 | TP16×DP4 | 64 | 64 | 27.1 | 27.1 | 1736 | 280.4 | no |
| 64 | TP32×DP2 | 64 | 64 | 17.3 | 17.3 | 1110 | 433.6 | no |
| 64 | TP64×DP1 | 64 | 64 | 9.9 | 9.9 | 633 | 753.1 | no |
| 128 | TP1×DP128 | 128 | 128 | 21.1 | 21.1 | 2703 | 359.2 | no |
| 128 | TP2×DP64 | 128 | 128 | 33.8 | 33.8 | 4325 | 227.7 | yes |
| 128 | TP4×DP32 | 128 | 128 | 43.7 | 43.7 | 5592 | 177.0 | yes |
| 128 | TP8×DP16 | 128 | 128 | 42.3 | 42.3 | 5410 | 182.0 | yes |
| 128 | TP16×DP8 | 128 | 128 | 31.0 | 31.0 | 3964 | 245.2 | yes |
| 128 | TP32×DP4 | 128 | 128 | 18.8 | 18.8 | 2411 | 398.4 | no |
| 128 | TP64×DP2 | 128 | 128 | 10.4 | 10.4 | 1327 | 717.9 | no |
| 256 | TP1×DP256 | 256 | 256 | 22.1 | 22.1 | 5653 | 341.6 | no |
| 256 | TP2×DP128 | 256 | 256 | 36.3 | 36.3 | 9299 | 210.1 | yes |
| 256 | TP4×DP64 | 256 | 256 | 48.0 | 48.0 | 12295 | 159.4 | yes |
| 256 | TP8×DP32 | 256 | 256 | 46.3 | 46.3 | 11858 | 164.4 | yes |
| 256 | TP16×DP16 | 256 | 256 | 33.1 | 33.1 | 8472 | 227.7 | yes |
| 256 | TP32×DP8 | 256 | 256 | 19.6 | 19.6 | 5018 | 380.8 | no |
| 256 | TP64×DP4 | 256 | 256 | 10.6 | 10.6 | 2712 | 700.3 | no |

## Interpretation

TP4 is the best swept equal-concurrency split at every cluster size. It shards the dominant BF16 KDA and shared-expert projections enough to overcome collective overhead while preserving enough DP replicas that each rank holds only four requests.

Past TP4, the latent MLA cache becomes increasingly costly at equal global concurrency. MLA cache is replicated across the TP ranks, while reducing DP forces local batch to rise in proportion to TP. Thus TP8 halves more projection work but doubles each rank's resident/request cache scan versus TP4.

These are uncalibrated custom-shape SOL ceilings. Hybrid TP×DP is supported in this AIC branch's SGLang formula path, but not its TensorRT-LLM path, and this does not prove that a production K3 kernel stack supports every topology.

Raw data: [`kimi_k3_10t_hybrid_tpdp_sweep.csv`](../data.zip); metadata: [`kimi_k3_10t_hybrid_tpdp_metadata.json`](kimi_k3_10t_hybrid_tpdp_metadata.json).
