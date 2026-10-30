> Historical exploratory study. See [scope and snapshot notes](../README.md) before using these projections.

# Spec-aware 50/100 TPS-user concurrency scaling at 256K

## Scope

This extends the exact no-spec SLO sweep with the common Kimi-K3 DSpark acceptance-length curve. At each adaptively searched c/r, it evaluates nextn 1–8 and selects the fastest memory-fitting result, with no-spec as a fallback. Each reported boundary and the immediately following failing c/r were directly evaluated.

| nextn | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AL | 1.00 | 1.85 | 2.51 | 3.00 | 3.36 | 3.62 | 3.75 | 3.84 | 4.00 |

## Spec-aware hardware concurrency lift across all 448 designs

| Target | Upgrade | GB-feasible | Newly feasible | Median lift | P10–P90 | Max lift | Clears target at physical max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 22-TB/s HBM only | 448/448 | 0 | 1.08x | 1.00–2.20x | 2.45x | 305/448 |
| 50 | HBM + 2× fabric | 448/448 | 0 | 1.24x | 1.00–2.20x | 2.57x | 415/448 |
| 100 | 22-TB/s HBM only | 448/448 | 0 | 1.72x | 1.00–3.75x | 5.80x | 213/448 |
| 100 | HBM + 2× fabric | 448/448 | 0 | 2.18x | 1.00–4.54x | 8.20x | 224/448 |

## How much speculation itself raises the SLO concurrency boundary

| Target | Hardware | No-spec-feasible | Newly feasible with spec | Median spec/no-spec c/r | P10–P90 | Boundary falls back to no-spec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 50 | GB300 | 448/448 | 0 | 1.00x | 1.00–1.41x | 141/448 |
| 50 | VR200 HBM | 448/448 | 0 | 1.00x | 1.00–1.00x | 295/448 |
| 50 | VR200 HBM+fabric | 448/448 | 0 | 1.00x | 1.00–1.00x | 281/448 |
| 100 | GB300 | 299/448 | 149 | 2.43x | 1.09–11.00x | 0/448 |
| 100 | VR200 HBM | 448/448 | 0 | 1.00x | 1.00–1.17x | 171/448 |
| 100 | VR200 HBM+fabric | 448/448 | 0 | 1.00x | 1.00–1.14x | 182/448 |

## Recommended controlled designs: no-spec → best-spec c/r

| Design | Target | GB300 | HBM | HBM+fabric | Spec-aware HBM lift vs GB | Spec-aware full lift vs GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggressive latency floor | 50 | 24→24 | 24→24 | 24→24 | 1.00x | 1.00x |
| Aggressive latency floor | 100 | 24→24 | 24→24 | 24→24 | 1.00x | 1.00x |
| Balanced recommendation | 50 | 78→87 | 107→107 | 159→159 | 1.23x | 1.83x |
| Balanced recommendation | 100 | 12→35 | 50→50 | 79→79 | 1.43x | 2.26x |
| Conservative recommendation | 50 | 56→63 | 79→79 | 115→115 | 1.25x | 1.83x |
| Conservative recommendation | 100 | 5→26 | 36→36 | 54→54 | 1.38x | 2.08x |
| Full-attention fallback | 50 | 17→23 | 39→39 | 44→44 | 1.70x | 1.91x |
| Full-attention fallback | 100 | 1→8 | 16→18 | 20→22 | 2.25x | 2.75x |
| KDA+sparse hybrid | 50 | 61→76 | 99→99 | 139→139 | 1.30x | 1.83x |
| KDA+sparse hybrid | 100 | 8→30 | 46→46 | 69→69 | 1.53x | 2.30x |
| Latent-MoE option | 50 | 107→107 | 151→151 | 184→184 | 1.41x | 1.72x |
| Latent-MoE option | 100 | 0→15 | 72→72 | 97→97 | 4.80x | 6.47x |

## Native-family anchors: best-spec c/r

| Anchor | Target | GB c/r | HBM c/r | HBM lift | HBM+fabric c/r | Full lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek ratio-preserved 5T | 50 | 82 | 101 | 1.23x | 148 | 1.80x |
| DeepSeek ratio-preserved 5T | 100 | 35 | 48 | 1.37x | 72 | 2.06x |
| GLM-5.2 ratio-preserved 5T | 50 | 13 | 13 | 1.00x | 13 | 1.00x |
| GLM-5.2 ratio-preserved 5T | 100 | 11 | 13 | 1.18x | 13 | 1.18x |
| Kimi ratio-preserved 5T | 50 | 13 | 30 | 2.31x | 30 | 2.31x |
| Kimi ratio-preserved 5T | 100 | 4 | 14 | 3.50x | 16 | 4.00x |
| Kimi-trend 5T | 50 | 9 | 15 | 1.67x | 15 | 1.67x |
| Kimi-trend 5T | 100 | 2 | 8 | 4.00x | 9 | 4.50x |

## Interpretation limits

- This is still SOL: no scheduler, framework overhead, queueing, prefill, or imbalance penalty.
- The AL curve is measured Kimi-K3 SPEED-Bench coding behavior and is deliberately held fixed. It is not evidence that a synthetic 5T architecture, DeepSeek-style body, or GLM body would achieve the same acceptance.
- Kimi-like bodies use AIC's K3/DSpark verifier; direct bodies use its generic MTP verifier. Cross-body speculative comparisons are sensitivity bounds.
- The adaptive search assumes the best-depth TPS envelope does not recover above a failed c/r. Every selected edge and edge+1 are directly checked, but this is not a dense evaluation of every speculative c/r.
- See `slo_spec_concurrency_comparison.csv` for every architecture/target and `SLO_CONCURRENCY_REPORT.md` for the dense no-spec control.
