> Historical exploratory study. See [scope and snapshot notes](../README.md) before using these projections.

# 50/100 TPS-user concurrency scaling at 256K

## Scope

This is an exact integer sweep of c/r=1..280 for all 448 controlled 5T architectures at DEP64 and 256K resident context. It reports the largest concurrency/rank that still clears 50 or 100 TPS/user. Global concurrency is c/r × 64, so hardware concurrency multipliers are identical in either unit.

The primary results are no-spec AIC SOL. GB300 and both Rubin substitutions use the same 277.5-GiB effective capacity; `VR200 HBM` changes only 8→22 TB/s HBM, while `VR200 HBM+NVLink` also changes directional fabric from 0.9→1.8 TB/s. A value of c/r=0 means the design misses the target even with one request/rank. Such a GB300→Rubin transition is called newly feasible, not an infinite multiplier.

## Whole design-space result

| Target | Upgrade | GB-feasible pairs | Newly feasible | Median lift | P10–P90 | Max lift | Median usable fraction of physical c/r | Architectures clearing target at physical max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 22-TB/s HBM only | 448/448 | 0 | 1.08x | 1.00–2.56x | 3.50x | 100% | 301/448 |
| 50 | HBM + 2× fabric | 448/448 | 0 | 1.31x | 1.00–2.72x | 3.50x | 100% | 415/448 |
| 100 | 22-TB/s HBM only | 299/448 | 149 | 3.67x | 1.12–18.40x | 87.00x | 67% | 207/448 |
| 100 | HBM + 2× fabric | 299/448 | 149 | 4.00x | 1.12–26.10x | 123.00x | 93% | 224/448 |

The paired lift statistics include only architectures already meeting the target on GB300. `Newly feasible` counts designs whose GB300 c/r is zero but whose Rubin c/r is positive. The physical-max column is the direct test of whether HBM can be filled while retaining the target rate.

## Attention-profile medians across all MoE bodies

### 50 TPS/user

| Attention | GB c/r | HBM c/r | HBM lift | HBM+fabric c/r | Full lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% DSA top-1024 | 14.0 | 14.0 | 1.00x | 14.0 | 1.00x |
| 100% DSA top-2048 | 14.0 | 14.0 | 1.00x | 14.0 | 1.00x |
| 100% DSA top-2048, per-layer index | 14.0 | 14.0 | 1.00x | 14.0 | 1.00x |
| 100% DSA top-4096 | 14.0 | 14.0 | 1.00x | 14.0 | 1.00x |
| 100% DSA top-512 | 14.0 | 14.0 | 1.00x | 14.0 | 1.00x |
| 100% KDA | 36.0 | 76.0 | 2.08x | 81.5 | 2.28x |
| 25% KDA + 75% DSA-2048 | 17.5 | 17.5 | 1.00x | 17.5 | 1.00x |
| 50% KDA + 50% DSA-2048 | 23.5 | 23.5 | 1.00x | 23.5 | 1.00x |
| 50% KDA + 50% MLA | 11.0 | 27.0 | 2.56x | 29.5 | 2.73x |
| 50% KDA + 50% V4 compressed | 44.0 | 88.0 | 1.85x | 103.5 | 2.23x |
| 75% KDA + 25% DSA-2048 | 36.5 | 36.5 | 1.00x | 36.5 | 1.00x |
| 75% KDA + 25% MLA | 17.0 | 41.0 | 2.43x | 43.5 | 2.59x |
| 75% KDA + 25% V4 compressed | 39.5 | 82.5 | 1.99x | 90.5 | 2.27x |
| 90% KDA + 10% MLA | 24.0 | 56.0 | 2.35x | 59.5 | 2.47x |
| 100% full MLA | 7.0 | 17.0 | 2.44x | 18.0 | 2.57x |
| 100% DeepSeek-V4 compressed | 59.0 | 107.0 | 1.57x | 133.0 | 2.06x |

### 100 TPS/user

| Attention | GB c/r | HBM c/r | HBM lift | HBM+fabric c/r | Full lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% DSA top-1024 | 12.0 | 14.0 | 1.20x | 14.0 | 1.20x |
| 100% DSA top-2048 | 11.0 | 14.0 | 1.33x | 14.0 | 1.33x |
| 100% DSA top-2048, per-layer index | 4.5 | 14.0 | 3.00x | 14.0 | 3.00x |
| 100% DSA top-4096 | 11.0 | 14.0 | 1.33x | 14.0 | 1.33x |
| 100% DSA top-512 | 12.0 | 14.0 | 1.20x | 14.0 | 1.20x |
| 100% KDA | 0.0 | 36.0 | 16.00x | 50.5 | 24.50x |
| 25% KDA + 75% DSA-2048 | 7.0 | 17.5 | 2.33x | 17.5 | 2.33x |
| 50% KDA + 50% DSA-2048 | 3.5 | 23.5 | 4.38x | 23.5 | 4.38x |
| 50% KDA + 50% MLA | 0.0 | 11.0 | 9.00x | 12.5 | 10.00x |
| 50% KDA + 50% V4 compressed | 0.0 | 40.5 | 11.00x | 59.0 | 16.00x |
| 75% KDA + 25% DSA-2048 | 1.0 | 33.0 | 8.25x | 36.5 | 9.25x |
| 75% KDA + 25% MLA | 0.0 | 16.0 | 13.00x | 20.0 | 16.00x |
| 75% KDA + 25% V4 compressed | 0.0 | 38.0 | 16.00x | 55.0 | 22.50x |
| 90% KDA + 10% MLA | 0.0 | 23.5 | 18.00x | 31.0 | 25.00x |
| 100% full MLA | 1.0 | 7.0 | 5.00x | 7.0 | 6.00x |
| 100% DeepSeek-V4 compressed | 1.0 | 48.0 | 7.80x | 71.0 | 12.80x |

## Recommended controlled designs

| Design | Target | GB c/r | HBM c/r | HBM lift | HBM+fabric c/r | Full lift | Physical max c/r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggressive latency floor | 50 | 24 | 24 | 1.00x | 24 | 1.00x | 24 |
| Aggressive latency floor | 100 | 24 | 24 | 1.00x | 24 | 1.00x | 24 |
| Balanced recommendation | 50 | 78 | 107 | 1.37x | 159 | 2.04x | 191 |
| Balanced recommendation | 100 | 12 | 50 | 4.17x | 79 | 6.58x | 191 |
| Conservative recommendation | 50 | 56 | 79 | 1.41x | 115 | 2.05x | 141 |
| Conservative recommendation | 100 | 5 | 36 | 7.20x | 54 | 10.80x | 141 |
| Full-attention fallback | 50 | 17 | 39 | 2.29x | 44 | 2.59x | 44 |
| Full-attention fallback | 100 | 1 | 16 | 16.00x | 20 | 20.00x | 44 |
| KDA+sparse hybrid | 50 | 61 | 99 | 1.62x | 139 | 2.28x | 139 |
| KDA+sparse hybrid | 100 | 8 | 46 | 5.75x | 69 | 8.62x | 139 |
| Latent-MoE option | 50 | 107 | 151 | 1.41x | 184 | 1.72x | 184 |
| Latent-MoE option | 100 | 0 | 72 | —x | 97 | —x | 184 |

## Native-family scaling anchors

| Anchor | Target | GB c/r | HBM c/r | HBM lift | HBM+fabric c/r | Full lift | Physical max c/r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek ratio-preserved 5T | 50 | 77 | 101 | 1.31x | 148 | 1.92x | 185 |
| DeepSeek ratio-preserved 5T | 100 | 12 | 48 | 4.00x | 72 | 6.00x | 185 |
| GLM-5.2 ratio-preserved 5T | 50 | 13 | 13 | 1.00x | 13 | 1.00x | 13 |
| GLM-5.2 ratio-preserved 5T | 100 | 3 | 13 | 4.33x | 13 | 4.33x | 13 |
| Kimi ratio-preserved 5T | 50 | 8 | 30 | 3.75x | 30 | 3.75x | 30 |
| Kimi ratio-preserved 5T | 100 | 0 | 12 | —x | 13 | —x | 30 |
| Kimi-trend 5T | 50 | 0 | 15 | —x | 15 | —x | 15 |
| Kimi-trend 5T | 100 | 0 | 4 | —x | 4 | —x | 15 |

## Interpretation limits

- These are hardware speed-of-light boundaries, not achieved serving concurrency. Scheduler, runtime, imbalance, queueing, and kernel inefficiency reduce the practical count.
- The VR200 rows are controlled substitutions into a GB300 compute model, not a complete Rubin prediction. HBM-only therefore deliberately exposes the non-HBM ceiling; the full row tests one fabric upgrade but still retains GB300 compute.
- Every request has a resident 256K cache and there is no prefill. Capacity is held equal to isolate bandwidth/fabric effects.
- Medians describe coverage of this constructed 448-design grid, not probabilities that a future architecture will take each form.
- The detailed per-architecture values, including zero/newly-feasible cases and fraction of physical capacity used at each target, are in `slo_concurrency_comparison.csv`.
