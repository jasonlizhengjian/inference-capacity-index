> Historical exploratory study. See [scope and snapshot notes](../README.md) before using these projections.

# 10T Kimi-trend hybrid TP×DP speculative-decoding sensitivity

AIC formula-only SGLang SOL at 256K on the hypothetical 288-GPU NVLink domain. Global concurrency equals GPU count. DSpark acceptance lengths are borrowed from InferenceX's measured K3 SPEED-Bench thinking-on curve; they are not measurements of a hypothetical 10T target.

| GPUs | Layout | No-spec TPS/user | Best nextn | Borrowed AL | Spec TPS/user | Speedup | TPS/GPU | GiB/rank |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | TP2×DP32, EP64 | 29.3 | 5 | 3.62 | 68.0 | 2.33× | 68.0 | 268.4 |
| 64 | TP4×DP16, EP64 | 36.4 | 3 | 3.00 | 63.7 | 1.75× | 63.7 | 217.5 |
| 64 | TP8×DP8, EP64 | 35.4 | 2 | 2.51 | 48.0 | 1.36× | 48.0 | 222.4 |
| 128 | TP2×DP64, EP128 | 33.8 | 4 | 3.36 | 73.9 | 2.19× | 73.9 | 233.1 |
| 128 | TP4×DP32, EP128 | 43.7 | 3 | 3.00 | 69.8 | 1.60× | 69.8 | 182.3 |
| 128 | TP8×DP16, EP128 | 42.3 | 2 | 2.51 | 52.3 | 1.24× | 52.3 | 187.2 |
| 256 | TP2×DP128, EP256 | 36.3 | 4 | 3.36 | 75.4 | 2.08× | 75.4 | 215.5 |
| 256 | TP4×DP64, EP256 | 48.0 | 2 | 2.51 | 72.1 | 1.50× | 72.1 | 164.6 |
| 256 | TP8×DP32, EP256 | 46.3 | 1 | 1.85 | 54.3 | 1.17× | 54.3 | 169.5 |

Speculation raises effective progress per target verification step, but the target must verify a wider token block and run the five-layer DSpark draft. The optimum is where the borrowed AL curve's flattening no longer pays for deeper verification.

The acceptance curve is the dominant uncertainty. The 10T target preserves K3 hidden size in this study, so the same DSpark geometry can be priced, but its prediction agreement with a future target could be materially different. Treat speedups as conditional sensitivities, not forecasts.

Raw data: [`kimi_k3_10t_hybrid_tpdp_spec_sweep.csv`](../data.zip); metadata: [`kimi_k3_10t_hybrid_tpdp_spec_metadata.json`](kimi_k3_10t_hybrid_tpdp_spec_metadata.json).
