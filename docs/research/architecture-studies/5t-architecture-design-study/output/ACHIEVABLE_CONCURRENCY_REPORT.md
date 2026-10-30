> Historical exploratory study. See [scope and snapshot notes](../README.md) before using these projections.

# Calibrated achievable-concurrency sweep

## Scope

This maps achieved 50/100 TPS-user targets onto the best-spec AIC SOL envelope at DEP64 and 256K resident context. The efficiency factors apply to throughput at fixed concurrency: 0.4× conservative, 0.5× central, and 0.7× optimistic. Equivalently, actual TPOT is modeled as 2.5×, 2.0×, or 1.43× SOL TPOT.

| Achieved target | Calibration | Required SOL target |
| ---: | --- | ---: |
| 50 | conservative (0.4×) | 125.0 |
| 50 | central (0.5×) | 100.0 |
| 50 | optimistic (0.7×) | 71.4 |
| 100 | conservative (0.4×) | 250.0 |
| 100 | central (0.5×) | 200.0 |
| 100 | optimistic (0.7×) | 142.9 |

## Whole 448-architecture grid

| Achieved target | Efficiency | Hardware | Feasible | Median c/r | P10–P90 c/r | Max c/r | Median usable physical c/r | Clears target at physical max | Median lift vs GB300 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.4× | Rubin HBM | 448/448 | 18.0 | 9.0–36.0 | 60 | 49% | 194/448 | 2.00x |
| 50 | 0.4× | Rubin HBM+fabric | 448/448 | 19.0 | 9.0–49.0 | 96 | 69% | 218/448 | 2.72x |
| 50 | 0.5× | Rubin HBM | 448/448 | 19.0 | 10.0–47.0 | 87 | 67% | 213/448 | 1.72x |
| 50 | 0.5× | Rubin HBM+fabric | 448/448 | 22.0 | 11.0–64.0 | 123 | 93% | 224/448 | 2.18x |
| 50 | 0.7× | Rubin HBM | 448/448 | 24.0 | 11.0–69.3 | 125 | 99% | 224/448 | 1.35x |
| 50 | 0.7× | Rubin HBM+fabric | 448/448 | 24.0 | 11.0–93.0 | 173 | 100% | 262/448 | 1.82x |
| 100 | 0.4× | Rubin HBM | 448/448 | 9.0 | 3.0–15.0 | 22 | 19% | 0/448 | 4.50x |
| 100 | 0.4× | Rubin HBM+fabric | 448/448 | 12.0 | 4.0–22.3 | 40 | 27% | 84/448 | 6.42x |
| 100 | 0.5× | Rubin HBM | 448/448 | 13.0 | 5.0–20.0 | 30 | 26% | 91/448 | 3.33x |
| 100 | 0.5× | Rubin HBM+fabric | 448/448 | 15.0 | 5.0–26.0 | 51 | 37% | 161/448 | 4.75x |
| 100 | 0.7× | Rubin HBM | 448/448 | 17.0 | 8.0–30.0 | 46 | 41% | 190/448 | 2.50x |
| 100 | 0.7× | Rubin HBM+fabric | 448/448 | 19.0 | 9.0–42.0 | 80 | 59% | 206/448 | 3.00x |

## Recommended designs on Rubin HBM+fabric

Each cell is c/r (global concurrency).

| Design | Achieved target | 0.4× conservative | 0.5× central | 0.7× optimistic | Physical max c/r |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aggressive latency floor | 50 | 24 (1,536) | 24 (1,536) | 24 (1,536) | 24 |
| Aggressive latency floor | 100 | 24 (1,536) | 24 (1,536) | 24 (1,536) | 24 |
| Balanced recommendation | 50 | 61 (3,904) | 79 (5,056) | 111 (7,104) | 191 |
| Balanced recommendation | 100 | 27 (1,728) | 35 (2,240) | 52 (3,328) | 191 |
| Conservative recommendation | 50 | 42 (2,688) | 54 (3,456) | 80 (5,120) | 141 |
| Conservative recommendation | 100 | 18 (1,152) | 24 (1,536) | 36 (2,304) | 141 |
| Full-attention fallback | 50 | 17 (1,088) | 22 (1,408) | 32 (2,048) | 44 |
| Full-attention fallback | 100 | 7 (448) | 9 (576) | 15 (960) | 44 |
| KDA+sparse hybrid | 50 | 54 (3,456) | 69 (4,416) | 100 (6,400) | 139 |
| KDA+sparse hybrid | 100 | 24 (1,536) | 32 (2,048) | 47 (3,008) | 139 |
| Latent-MoE option | 50 | 75 (4,800) | 97 (6,208) | 138 (8,832) | 184 |
| Latent-MoE option | 100 | 17 (1,088) | 32 (2,048) | 62 (3,968) | 184 |

## Native-family anchors on Rubin HBM+fabric

| Anchor | Achieved target | 0.4× conservative | 0.5× central | 0.7× optimistic | Physical max c/r |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek ratio-preserved 5T | 50 | 56 (3,584) | 72 (4,608) | 103 (6,592) | 185 |
| DeepSeek ratio-preserved 5T | 100 | 25 (1,600) | 32 (2,048) | 47 (3,008) | 185 |
| GLM-5.2 ratio-preserved 5T | 50 | 13 (832) | 13 (832) | 13 (832) | 13 |
| GLM-5.2 ratio-preserved 5T | 100 | 11 (704) | 13 (832) | 13 (832) | 13 |
| Kimi ratio-preserved 5T | 50 | 11 (704) | 16 (1,024) | 24 (1,536) | 30 |
| Kimi ratio-preserved 5T | 100 | 3 (192) | 5 (320) | 9 (576) | 30 |
| Kimi-trend 5T | 50 | 6 (384) | 9 (576) | 12 (768) | 15 |
| Kimi-trend 5T | 100 | 1 (64) | 3 (192) | 5 (320) | 15 |

## Interpretation limits

- Calibration is a global sensitivity multiplier, not an operator-by-operator achievable model. Real efficiency can change with c/r, architecture, kernel maturity, and hardware.
- The 0.7× case is an optimized upside, especially uncertain for synthetic attention mechanisms and future Rubin kernels.
- The common DSpark acceptance curve remains a borrowed sensitivity assumption outside Kimi-like bodies; no-spec is always available as a fallback.
- Results remain decode-only SOL-derived projections with resident 256K cache and no prefill, queueing, or scheduler effects.
- `achievable_concurrency_comparison.csv` contains every architecture/scenario and both c/r and global concurrency.
