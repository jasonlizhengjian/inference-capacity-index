> Historical exploratory study. See [scope and snapshot notes](../README.md) before using these projections.

# Scaling Kimi-K3 to approximately 5T and 10T: decode-only SOL study

These are hypothetical architecture projections, not announced Moonshot models. Every point is a formula-only AIC TensorRT-LLM speed-of-light decode step with BF16 KV already resident. There is no prefill, KV transfer, speculation, scheduler, queueing, or production-runtime loss.

## Scaling assumptions

Kimi K2.5→K3 was a compound scale-up, not merely more experts: layers 61→93, routed experts 384→896, top-k 8→16, expert intermediate width 2048→3072, shared experts 1→2, and heads 64→96, while hidden size stayed 7168. K3 also replaced all-MLA with 69 KDA + 24 MLA layers and introduced a 3584-dimensional LatentMoE path.

| Dimension | Kimi K2.5 | Kimi K3 | Observed change |
| --- | --- | --- | --- |
| Nominal scale class | ~1T | ~2.8T text / ~3T | ~2.8–3× |
| Hidden size | 7168 | 7168 | unchanged |
| Layers | 61 | 93 | 1.52× |
| Routed experts | 384 | 896 | 2.33× |
| Experts/token | 8 | 16 | 2.00× |
| Expert intermediate | 2048 | 3072 | 1.50× |
| Shared experts | 1 | 2 | 2.00× |
| Attention heads | 64 | 96 | 1.50× |
| Dense-stem intermediate | 18432 | 33792 | 1.83× |
| Attention pattern | 61 MLA | 69 KDA + 24 MLA | hybridized |
| Maximum context | 256K | 1M | 4.00× |

The table is taken from the two AIC model configurations used by this workspace. K3's LatentMoE changes the expert input geometry, so no single K2.5→K3 dimension ratio can be extrapolated safely.

The study therefore brackets three directions:

- **Expert-only:** increase total routed experts to 1600/3200 while retaining K3's 93 layers, top-16, and per-expert geometry. This is the constant-active-compute lower bound.
- **Kimi-trend:** jointly increase depth, experts, heads, top-k, shared experts, and the dense stem, while preserving K3's hidden size, LatentMoE expert geometry, and approximately 3:1 KDA:MLA pattern. This is the central scenario informed by K2.5→K3.
- **Depth-only:** hold every K3 dimension except layer count. This makes active work scale nearly with total parameters and serves as a high-penalty bound, not the expected product design.

Parameter counts below are text-model accounting from the operation geometry. They omit the small vision tower and quantization metadata overhead.

| Variant | Family | Params | Active/token | Layers KDA/MLA | Experts | Top-k | Heads | Packed weights |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K3 2.8T baseline | baseline | 2.78T | 105B | 93 (69/24) | 896 | 16 | 96 | 1374 GiB |
| ~5T expert-only | expert_only | 4.92T | 106B | 93 (69/24) | 1600 | 16 | 96 | 2371 GiB |
| ~10T expert-only | expert_only | 9.78T | 107B | 93 (69/24) | 3200 | 16 | 96 | 4637 GiB |
| ~5T Kimi-trend | kimi_trend | 4.96T | 184B | 116 (87/29) | 1280 | 24 | 128 | 2438 GiB |
| ~10T Kimi-trend | kimi_trend | 9.84T | 314B | 144 (108/36) | 2048 | 32 | 192 | 4808 GiB |
| ~5T depth-only | depth_only | 4.98T | 186B | 166 (124/42) | 896 | 16 | 96 | 2459 GiB |
| ~10T depth-only | depth_only | 9.99T | 371B | 332 (249/83) | 896 | 16 | 96 | 4926 GiB |

## Minimum memory-fit topology in the sweep

`DEP` means TP1 with attention-DP=MoE-EP=world. `TEP` means attention TP=world and MoE-EP=world. Only TP widths that evenly divide the variant's head count were evaluated. DEP128/256 and most TEP widths are theoretical K3 layouts, not documented serving recipes.

| Variant | System | Context | Layout | Min GPUs | TPS/user | GiB/rank | Weights | Cache/state |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| K3 2.8T baseline | GB300 | 256K | DEP | 16 | 55.6 | 197.3 | 184.9 | 8.8 |
| K3 2.8T baseline | GB300 | 256K | TEP | 16 | 299.4 | 106.8 | 96.0 | 6.9 |
| K3 2.8T baseline | GB300 | 1M | DEP | 16 | 48.3 | 217.6 | 184.9 | 29.1 |
| K3 2.8T baseline | GB300 | 1M | TEP | 16 | 165.1 | 127.0 | 96.0 | 27.1 |
| K3 2.8T baseline | GB300 + Rubin HBM | 256K | DEP | 16 | 148.2 | 197.3 | 184.9 | 8.8 |
| K3 2.8T baseline | GB300 + Rubin HBM | 256K | TEP | 16 | 821.0 | 106.8 | 96.0 | 6.9 |
| K3 2.8T baseline | GB300 + Rubin HBM | 1M | DEP | 16 | 120.1 | 217.6 | 184.9 | 29.1 |
| K3 2.8T baseline | GB300 + Rubin HBM | 1M | TEP | 16 | 453.2 | 127.0 | 96.0 | 27.1 |
| ~5T expert-only | GB300 | 256K | DEP | 16 | 55.3 | 260.4 | 248.0 | 8.8 |
| ~5T expert-only | GB300 | 256K | TEP | 16 | 289.4 | 169.9 | 159.1 | 6.9 |
| ~5T expert-only | GB300 | 1M | DEP | 32 | 48.0 | 209.9 | 177.3 | 29.1 |
| ~5T expert-only | GB300 | 1M | TEP | 16 | 162.0 | 190.2 | 159.1 | 27.1 |
| ~5T expert-only | GB300 + Rubin HBM | 256K | DEP | 16 | 147.3 | 260.4 | 248.0 | 8.8 |
| ~5T expert-only | GB300 + Rubin HBM | 256K | TEP | 16 | 793.5 | 169.9 | 159.1 | 6.9 |
| ~5T expert-only | GB300 + Rubin HBM | 1M | DEP | 32 | 119.2 | 209.9 | 177.3 | 29.1 |
| ~5T expert-only | GB300 + Rubin HBM | 1M | TEP | 16 | 444.7 | 190.2 | 159.1 | 27.1 |
| ~10T expert-only | GB300 | 256K | DEP | 32 | 54.4 | 262.4 | 250.0 | 8.8 |
| ~10T expert-only | GB300 | 256K | TEP | 24 | 305.8 | 217.0 | 206.2 | 6.8 |
| ~10T expert-only | GB300 | 1M | DEP | 64 | 47.3 | 211.9 | 179.2 | 29.1 |
| ~10T expert-only | GB300 | 1M | TEP | 24 | 167.0 | 237.3 | 206.2 | 27.1 |
| ~10T expert-only | GB300 + Rubin HBM | 256K | DEP | 32 | 144.7 | 262.4 | 250.0 | 8.8 |
| ~10T expert-only | GB300 + Rubin HBM | 256K | TEP | 24 | 837.8 | 217.0 | 206.2 | 6.8 |
| ~10T expert-only | GB300 + Rubin HBM | 1M | DEP | 64 | 117.2 | 211.9 | 179.2 | 29.1 |
| ~10T expert-only | GB300 + Rubin HBM | 1M | TEP | 24 | 458.3 | 237.3 | 206.2 | 27.1 |
| ~5T Kimi-trend | GB300 | 256K | DEP | 32 | 33.2 | 259.8 | 244.6 | 11.7 |
| ~5T Kimi-trend | GB300 | 256K | TEP | 16 | 218.8 | 177.9 | 165.6 | 8.4 |
| ~5T Kimi-trend | GB300 | 1M | DEP | 64 | 30.8 | 248.9 | 209.2 | 36.1 |
| ~5T Kimi-trend | GB300 | 1M | TEP | 16 | 127.3 | 202.4 | 165.6 | 32.8 |
| ~5T Kimi-trend | GB300 + Rubin HBM | 256K | DEP | 32 | 87.5 | 259.8 | 244.6 | 11.7 |
| ~5T Kimi-trend | GB300 + Rubin HBM | 256K | TEP | 16 | 600.1 | 177.9 | 165.6 | 8.4 |
| ~5T Kimi-trend | GB300 + Rubin HBM | 1M | DEP | 64 | 73.1 | 248.9 | 209.2 | 36.1 |
| ~5T Kimi-trend | GB300 + Rubin HBM | 1M | TEP | 16 | 349.6 | 202.4 | 165.6 | 32.8 |
| ~10T Kimi-trend | GB300 | 256K | DEP | none | — | — | — | — |
| ~10T Kimi-trend | GB300 | 256K | TEP | 24 | 171.8 | 232.9 | 218.6 | 10.4 |
| ~10T Kimi-trend | GB300 | 1M | DEP | none | — | — | — | — |
| ~10T Kimi-trend | GB300 | 1M | TEP | 24 | 101.0 | 263.3 | 218.6 | 40.8 |
| ~10T Kimi-trend | GB300 + Rubin HBM | 256K | DEP | none | — | — | — | — |
| ~10T Kimi-trend | GB300 + Rubin HBM | 256K | TEP | 24 | 471.1 | 232.9 | 218.6 | 10.4 |
| ~10T Kimi-trend | GB300 + Rubin HBM | 1M | DEP | none | — | — | — | — |
| ~10T Kimi-trend | GB300 + Rubin HBM | 1M | TEP | 24 | 277.4 | 263.3 | 218.6 | 40.8 |
| ~5T depth-only | GB300 | 256K | DEP | 32 | 31.4 | 274.9 | 255.8 | 15.6 |
| ~5T depth-only | GB300 | 256K | TEP | 16 | 168.9 | 187.8 | 171.8 | 12.0 |
| ~5T depth-only | GB300 | 1M | DEP | 64 | 27.8 | 274.8 | 220.2 | 51.0 |
| ~5T depth-only | GB300 | 1M | TEP | 16 | 93.7 | 223.2 | 171.8 | 47.5 |
| ~5T depth-only | GB300 + Rubin HBM | 256K | DEP | 32 | 83.5 | 274.9 | 255.8 | 15.6 |
| ~5T depth-only | GB300 + Rubin HBM | 256K | TEP | 16 | 463.1 | 187.8 | 171.8 | 12.0 |
| ~5T depth-only | GB300 + Rubin HBM | 1M | DEP | 64 | 68.6 | 274.8 | 220.2 | 51.0 |
| ~5T depth-only | GB300 + Rubin HBM | 1M | TEP | 16 | 257.1 | 223.2 | 171.8 | 47.5 |
| ~10T depth-only | GB300 | 256K | DEP | none | — | — | — | — |
| ~10T depth-only | GB300 | 256K | TEP | 24 | 98.0 | 270.0 | 242.4 | 23.7 |
| ~10T depth-only | GB300 | 1M | DEP | none | — | — | — | — |
| ~10T depth-only | GB300 | 1M | TEP | 48 | 53.5 | 238.0 | 140.5 | 93.5 |
| ~10T depth-only | GB300 + Rubin HBM | 256K | DEP | none | — | — | — | — |
| ~10T depth-only | GB300 + Rubin HBM | 256K | TEP | 24 | 268.3 | 270.0 | 242.4 | 23.7 |
| ~10T depth-only | GB300 + Rubin HBM | 1M | DEP | none | — | — | — | — |
| ~10T depth-only | GB300 + Rubin HBM | 1M | TEP | 48 | 146.8 | 238.0 | 140.5 | 93.5 |

## Comparable 256K DEP64 directional slice

This holds the topology at DEP64, local batch at 1, and global concurrency at 64. It is the cleanest apples-to-apples measure of the architectural penalty. Rows marked `no` exceed 277.6 GiB/rank; their TPS/user is a timing counterfactual, not a runnable configuration.

| Variant | Active/token | GB300 TPS/user | vs K3 | 22 TB/s TPS/user | vs K3 | GiB/rank | Fits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| K3 2.8T baseline | 105B | 56.6 | 1.00× | 149.7 | 1.00× | 137.9 | yes |
| ~5T expert-only | 106B | 55.0 | 0.97× | 145.8 | 0.97× | 154.3 | yes |
| ~10T expert-only | 107B | 54.3 | 0.96× | 143.7 | 0.96× | 191.7 | yes |
| ~5T Kimi-trend | 184B | 34.2 | 0.61× | 89.7 | 0.60× | 224.5 | yes |
| ~10T Kimi-trend | 314B | 19.3 | 0.34× | 50.1 | 0.33× | 393.7 | no |
| ~5T depth-only | 186B | 32.0 | 0.57× | 84.7 | 0.57× | 239.4 | yes |
| ~10T depth-only | 371B | 16.1 | 0.28× | 42.6 | 0.28× | 470.2 | no |

## Why the 10T Kimi-trend model does not fit DEP—and how quantization or TEP changes that

In this experiment, routed experts use MXFP4 while attention, dense, shared-expert, router, latent, and output weights remain BF16. `DEP64, c/r=1` means one request per rank and therefore 64 requests globally. Reducing the global workload to one request would remove cache/state from idle ranks, but would not solve the capacity problem: weights alone exceed one GB300's 277.6 GiB.

| 10T Kimi-trend layout | Weights | Cache/state | Other | Total | Fits GB300 |
| --- | ---: | ---: | ---: | ---: | --- |
| DEP64, c/r=1 | 373.4 GiB | 16.7 GiB | 3.6 GiB | 393.7 GiB | no |
| DEP256, c/r=1 | 320.7 GiB | 16.7 GiB | 3.6 GiB | 340.9 GiB | no |
| TEP24, local batch=1 | 218.6 GiB | 10.4 GiB | 3.9 GiB | 232.9 GiB | yes |
| DEP64 + estimated FP8 non-routed weights, c/r=1 | ~221.9 GiB | 16.7 GiB | 3.6 GiB | ~242.2 GiB | yes |

DEP already shards the routed expert payload through EP, but leaves roughly 303 GiB of non-routed BF16 weights replicated on every rank. TEP tensor-shards much of that non-routed component while retaining expert parallelism. This helps even with MLA: MLA reduces the token-dependent KV cache, but its projection weights remain shardable. Moreover, K3 is hybrid rather than all-MLA—baseline K3 has 69 KDA + 24 MLA layers, and the hypothetical 10T model has 108 KDA + 36 MLA layers.

The FP8 row is capacity arithmetic, not a rerun with verified FP8 kernels: halving the approximately 303 GiB replicated BF16 component while leaving the roughly 70.4 GiB MXFP4 expert shard unchanged gives about 221.9 GiB of weights. That would fit at 256K through approximately c/r=3 (~275.5 GiB total) but not c/r=4 (~292.2 GiB). Accuracy, kernel availability, and realized performance require separate validation.


## 256K DEP64 comparison

| Variant | System | c/r | Global | TPS/user | TPS/GPU | GiB/rank | OOM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| K3 2.8T baseline | GB300 | 1 | 64 | 56.6 | 56.6 | 137.9 | no |
| K3 2.8T baseline | GB300 | 2 | 128 | 53.2 | 106.4 | 146.7 | no |
| K3 2.8T baseline | GB300 | 3 | 192 | 50.2 | 150.6 | 155.6 | no |
| K3 2.8T baseline | GB300 | 4 | 256 | 47.5 | 190.1 | 164.4 | no |
| K3 2.8T baseline | GB300 + Rubin HBM | 1 | 64 | 149.7 | 149.7 | 137.9 | no |
| K3 2.8T baseline | GB300 + Rubin HBM | 2 | 128 | 136.2 | 272.3 | 146.7 | no |
| K3 2.8T baseline | GB300 + Rubin HBM | 3 | 192 | 124.9 | 374.7 | 155.6 | no |
| K3 2.8T baseline | GB300 + Rubin HBM | 4 | 256 | 115.3 | 461.4 | 164.4 | no |
| ~5T expert-only | GB300 | 1 | 64 | 55.0 | 55.0 | 154.3 | no |
| ~5T expert-only | GB300 | 2 | 128 | 47.6 | 95.2 | 163.2 | no |
| ~5T expert-only | GB300 | 3 | 192 | 45.2 | 135.6 | 172.0 | no |
| ~5T expert-only | GB300 | 4 | 256 | 43.0 | 172.1 | 180.8 | no |
| ~5T expert-only | GB300 + Rubin HBM | 1 | 64 | 145.8 | 145.8 | 154.3 | no |
| ~5T expert-only | GB300 + Rubin HBM | 2 | 128 | 122.8 | 245.5 | 163.2 | no |
| ~5T expert-only | GB300 + Rubin HBM | 3 | 192 | 113.5 | 340.6 | 172.0 | no |
| ~5T expert-only | GB300 + Rubin HBM | 4 | 256 | 105.6 | 422.3 | 180.8 | no |
| ~10T expert-only | GB300 | 1 | 64 | 54.3 | 54.3 | 191.7 | no |
| ~10T expert-only | GB300 | 2 | 128 | 44.3 | 88.5 | 200.5 | no |
| ~10T expert-only | GB300 | 3 | 192 | 37.4 | 112.1 | 209.3 | no |
| ~10T expert-only | GB300 | 4 | 256 | 35.4 | 141.6 | 218.2 | no |
| ~10T expert-only | GB300 + Rubin HBM | 1 | 64 | 143.7 | 143.7 | 191.7 | no |
| ~10T expert-only | GB300 + Rubin HBM | 2 | 128 | 114.6 | 229.2 | 200.5 | no |
| ~10T expert-only | GB300 + Rubin HBM | 3 | 192 | 95.3 | 285.9 | 209.3 | no |
| ~10T expert-only | GB300 + Rubin HBM | 4 | 256 | 88.5 | 354.2 | 218.2 | no |
| ~5T Kimi-trend | GB300 | 1 | 64 | 34.2 | 34.2 | 224.5 | no |
| ~5T Kimi-trend | GB300 | 2 | 128 | 32.7 | 65.3 | 236.1 | no |
| ~5T Kimi-trend | GB300 | 3 | 192 | 31.2 | 93.6 | 247.8 | no |
| ~5T Kimi-trend | GB300 | 4 | 256 | 29.9 | 119.5 | 259.5 | no |
| ~5T Kimi-trend | GB300 + Rubin HBM | 1 | 64 | 89.7 | 89.7 | 224.5 | no |
| ~5T Kimi-trend | GB300 + Rubin HBM | 2 | 128 | 82.1 | 164.2 | 236.1 | no |
| ~5T Kimi-trend | GB300 + Rubin HBM | 3 | 192 | 75.7 | 227.0 | 247.8 | no |
| ~5T Kimi-trend | GB300 + Rubin HBM | 4 | 256 | 70.2 | 280.6 | 259.5 | no |
| ~10T Kimi-trend | GB300 | 1 | 64 | 19.3 | 19.3 | 393.7 | yes |
| ~10T Kimi-trend | GB300 | 2 | 128 | 18.5 | 37.0 | 410.4 | yes |
| ~10T Kimi-trend | GB300 | 3 | 192 | 17.8 | 53.4 | 427.0 | yes |
| ~10T Kimi-trend | GB300 | 4 | 256 | 17.2 | 68.7 | 443.7 | yes |
| ~10T Kimi-trend | GB300 + Rubin HBM | 1 | 64 | 50.1 | 50.1 | 393.7 | yes |
| ~10T Kimi-trend | GB300 + Rubin HBM | 2 | 128 | 45.8 | 91.6 | 410.4 | yes |
| ~10T Kimi-trend | GB300 + Rubin HBM | 3 | 192 | 42.2 | 126.6 | 427.0 | yes |
| ~10T Kimi-trend | GB300 + Rubin HBM | 4 | 256 | 39.1 | 156.5 | 443.7 | yes |
| ~5T depth-only | GB300 | 1 | 64 | 32.0 | 32.0 | 239.4 | no |
| ~5T depth-only | GB300 | 2 | 128 | 30.1 | 60.2 | 255.0 | no |
| ~5T depth-only | GB300 | 3 | 192 | 28.4 | 85.2 | 270.5 | no |
| ~5T depth-only | GB300 | 4 | 256 | 26.9 | 107.6 | 286.1 | yes |
| ~5T depth-only | GB300 + Rubin HBM | 1 | 64 | 84.7 | 84.7 | 239.4 | no |
| ~5T depth-only | GB300 + Rubin HBM | 2 | 128 | 77.1 | 154.1 | 255.0 | no |
| ~5T depth-only | GB300 + Rubin HBM | 3 | 192 | 70.7 | 212.1 | 270.5 | no |
| ~5T depth-only | GB300 + Rubin HBM | 4 | 256 | 65.3 | 261.3 | 286.1 | yes |
| ~10T depth-only | GB300 | 1 | 64 | 16.1 | 16.1 | 470.2 | yes |
| ~10T depth-only | GB300 | 2 | 128 | 15.1 | 30.3 | 501.1 | yes |
| ~10T depth-only | GB300 | 3 | 192 | 14.3 | 42.9 | 532.0 | yes |
| ~10T depth-only | GB300 | 4 | 256 | 13.5 | 54.2 | 562.9 | yes |
| ~10T depth-only | GB300 + Rubin HBM | 1 | 64 | 42.6 | 42.6 | 470.2 | yes |
| ~10T depth-only | GB300 + Rubin HBM | 2 | 128 | 38.8 | 77.6 | 501.1 | yes |
| ~10T depth-only | GB300 + Rubin HBM | 3 | 192 | 35.6 | 106.8 | 532.0 | yes |
| ~10T depth-only | GB300 + Rubin HBM | 4 | 256 | 32.9 | 131.5 | 562.9 | yes |

## Theoretical best memory-fit 256K latency at each load

This section minimizes per-user latency across the whole sweep. It can assign 96–192 GPUs to one request and has very poor TPS/GPU, so it is a mathematical latency frontier—not a deployment recommendation.

| Variant | System | Local batch | Best layout | Global | TPS/user | TPS/GPU | GiB/rank |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| K3 2.8T baseline | GB300 | 1 | TEP96 | 1 | 393.2 | 4.1 | 35.7 |
| K3 2.8T baseline | GB300 | 2 | TEP96 | 2 | 285.0 | 5.9 | 42.5 |
| K3 2.8T baseline | GB300 | 3 | TEP96 | 3 | 223.5 | 7.0 | 49.3 |
| K3 2.8T baseline | GB300 | 4 | TEP96 | 4 | 183.8 | 7.7 | 56.0 |
| K3 2.8T baseline | GB300 + Rubin HBM | 1 | TEP96 | 1 | 1042.8 | 10.9 | 35.7 |
| K3 2.8T baseline | GB300 + Rubin HBM | 2 | TEP48 | 2 | 770.1 | 32.1 | 56.7 |
| K3 2.8T baseline | GB300 + Rubin HBM | 3 | TEP48 | 3 | 586.8 | 36.7 | 63.5 |
| K3 2.8T baseline | GB300 + Rubin HBM | 4 | TEP48 | 4 | 489.7 | 40.8 | 70.3 |
| ~5T expert-only | GB300 | 1 | TEP96 | 1 | 376.0 | 3.9 | 46.9 |
| ~5T expert-only | GB300 | 2 | TEP96 | 2 | 275.8 | 5.7 | 53.7 |
| ~5T expert-only | GB300 | 3 | TEP96 | 3 | 217.8 | 6.8 | 60.5 |
| ~5T expert-only | GB300 | 4 | TEP96 | 4 | 179.9 | 7.5 | 67.3 |
| ~5T expert-only | GB300 + Rubin HBM | 1 | TEP96 | 1 | 998.9 | 10.4 | 46.9 |
| ~5T expert-only | GB300 + Rubin HBM | 2 | TEP48 | 2 | 745.8 | 31.1 | 78.3 |
| ~5T expert-only | GB300 + Rubin HBM | 3 | TEP48 | 3 | 572.6 | 35.8 | 85.1 |
| ~5T expert-only | GB300 + Rubin HBM | 4 | TEP48 | 4 | 479.8 | 40.0 | 91.9 |
| ~10T expert-only | GB300 | 1 | TEP96 | 1 | 342.1 | 3.6 | 72.5 |
| ~10T expert-only | GB300 | 2 | TEP96 | 2 | 257.1 | 5.4 | 79.3 |
| ~10T expert-only | GB300 | 3 | TEP96 | 3 | 206.0 | 6.4 | 86.0 |
| ~10T expert-only | GB300 | 4 | TEP96 | 4 | 171.8 | 7.2 | 92.8 |
| ~10T expert-only | GB300 + Rubin HBM | 1 | TEP96 | 1 | 911.5 | 9.5 | 72.5 |
| ~10T expert-only | GB300 + Rubin HBM | 2 | TEP48 | 2 | 696.0 | 29.0 | 127.5 |
| ~10T expert-only | GB300 + Rubin HBM | 3 | TEP48 | 3 | 542.8 | 33.9 | 134.3 |
| ~10T expert-only | GB300 + Rubin HBM | 4 | TEP48 | 4 | 458.6 | 38.2 | 141.1 |
| ~5T Kimi-trend | GB300 | 1 | TEP128 | 1 | 310.0 | 2.4 | 45.2 |
| ~5T Kimi-trend | GB300 | 2 | TEP128 | 2 | 227.5 | 3.6 | 53.3 |
| ~5T Kimi-trend | GB300 | 3 | TEP128 | 3 | 179.7 | 4.2 | 61.5 |
| ~5T Kimi-trend | GB300 | 4 | TEP128 | 4 | 148.5 | 4.6 | 69.7 |
| ~5T Kimi-trend | GB300 + Rubin HBM | 1 | TEP128 | 1 | 822.7 | 6.4 | 45.2 |
| ~5T Kimi-trend | GB300 + Rubin HBM | 2 | TEP64 | 2 | 614.4 | 19.2 | 72.3 |
| ~5T Kimi-trend | GB300 + Rubin HBM | 3 | TEP64 | 3 | 471.5 | 22.1 | 80.5 |
| ~5T Kimi-trend | GB300 + Rubin HBM | 4 | TEP64 | 4 | 395.3 | 24.7 | 88.8 |
| ~10T Kimi-trend | GB300 | 1 | TEP192 | 1 | 237.6 | 1.2 | 58.1 |
| ~10T Kimi-trend | GB300 | 2 | TEP192 | 2 | 176.6 | 1.8 | 68.3 |
| ~10T Kimi-trend | GB300 | 3 | TEP192 | 3 | 140.6 | 2.2 | 78.4 |
| ~10T Kimi-trend | GB300 | 4 | TEP192 | 4 | 116.8 | 2.4 | 88.6 |
| ~10T Kimi-trend | GB300 + Rubin HBM | 1 | TEP192 | 1 | 631.6 | 3.3 | 58.1 |
| ~10T Kimi-trend | GB300 + Rubin HBM | 2 | TEP192 | 2 | 462.1 | 4.8 | 68.3 |
| ~10T Kimi-trend | GB300 + Rubin HBM | 3 | TEP192 | 3 | 364.3 | 5.7 | 78.4 |
| ~10T Kimi-trend | GB300 + Rubin HBM | 4 | TEP192 | 4 | 300.7 | 6.3 | 88.6 |
| ~5T depth-only | GB300 | 1 | TEP96 | 1 | 221.6 | 2.3 | 60.5 |
| ~5T depth-only | GB300 | 2 | TEP96 | 2 | 161.1 | 3.4 | 72.4 |
| ~5T depth-only | GB300 | 3 | TEP96 | 3 | 126.6 | 4.0 | 84.2 |
| ~5T depth-only | GB300 | 4 | TEP96 | 4 | 104.3 | 4.3 | 96.1 |
| ~5T depth-only | GB300 + Rubin HBM | 1 | TEP96 | 1 | 587.6 | 6.1 | 60.5 |
| ~5T depth-only | GB300 + Rubin HBM | 2 | TEP48 | 2 | 435.6 | 18.2 | 97.9 |
| ~5T depth-only | GB300 + Rubin HBM | 3 | TEP48 | 3 | 332.4 | 20.8 | 109.8 |
| ~5T depth-only | GB300 + Rubin HBM | 4 | TEP48 | 4 | 277.7 | 23.1 | 121.7 |
| ~10T depth-only | GB300 | 1 | TEP96 | 1 | 111.2 | 1.2 | 117.0 |
| ~10T depth-only | GB300 | 2 | TEP96 | 2 | 81.0 | 1.7 | 140.4 |
| ~10T depth-only | GB300 | 3 | TEP96 | 3 | 63.7 | 2.0 | 163.9 |
| ~10T depth-only | GB300 | 4 | TEP96 | 4 | 52.5 | 2.2 | 187.3 |
| ~10T depth-only | GB300 + Rubin HBM | 1 | TEP48 | 1 | 294.8 | 6.1 | 168.0 |
| ~10T depth-only | GB300 + Rubin HBM | 2 | TEP48 | 2 | 219.1 | 9.1 | 191.5 |
| ~10T depth-only | GB300 + Rubin HBM | 3 | TEP48 | 3 | 167.3 | 10.5 | 215.0 |
| ~10T depth-only | GB300 + Rubin HBM | 4 | TEP48 | 4 | 139.9 | 11.7 | 238.5 |

## Interpretation

- Total parameters are not enough to predict decode speed for an MoE. Expert-only 5T/10T keeps routed active work fixed; the penalties are larger weight capacity, a wider replicated router, and more difficult expert-parallel communication/load balance. With enough EP ranks, TPS/user can remain surprisingly close to K3.
- The Kimi-trend variants are much slower because K2.5→K3 also increased active structure. More layers multiply KDA, MLA, shared-expert, and communication work; larger top-k raises routed MoE work directly; more heads enlarge KDA and MLA projections.
- Depth-only shows why a dense-like interpretation of 5T/10T is pessimistic for MoE but useful as a bound: both active weights and per-request KDA/MLA state grow nearly linearly with model size.
- 1M context can become a memory constraint even when weights are well sharded. More MLA layers grow token-linear KV, while more KDA layers grow fixed recurrent state per admitted request. TEP shards KDA state and dense projections but does not eliminate MLA KV reads.
- The 22-TB/s substitution remains a sensitivity, not a Rubin benchmark. As the active model grows, HBM bandwidth remains valuable, but attention compute and wider/more frequent communication take a larger non-scaling share.
- On the normalized 256K DEP64 slice, expert-only scaling retains 97%/96% of baseline single-user GB300 speed at 5T/10T. The Kimi-trend path retains 60%/34%; depth-only retains 57%/29%. The 10T Kimi-trend and depth-only DEP64 rows are over capacity.
- The 10T Kimi-trend design cannot fit any swept DEP topology because DEP leaves its roughly 321 GiB of non-routed TP1 weights replicated per rank even at DEP256. It first fits as TEP24 (233 GiB/rank at 256K), which illustrates why architecture and parallelism have to be designed together.
- At 256K on GB300 DEP64, the Kimi-trend slowdown is mostly active model work, not cache reading. The explicit MLA cache-attention term is 5.1% of baseline TPOT, 3.8% at 5T, and 3.0% at 10T; KDA projections, routed MoE, and shared experts dominate instead.
- At fixed GB300 DEP64, moving from 256K to 1M lowers TPS/user by 13% for baseline K3, 10% for the 5T Kimi-trend model, and 8% for the over-capacity 10T timing counterfactual. At the minimum-fit TEP topologies, the same context increase costs about 41–45%, because a single sharded request exposes cache-attention latency rather than amortizing model work across 64 DEP requests.

## Confidence and caveats

- The baseline K3 formula path has limited calibration; custom 5T/10T shapes have none. Treat absolute TPS/user as low-confidence ceilings and model-family comparisons as directional.
- AIC formula mode can price arbitrary dimensions, but it does not prove that kernels, load balancing, routing quality, training stability, or these TP/EP layouts would exist in a real implementation.
- DEP64+ and TEP widths beyond documented K3 recipes idealize multi-node collectives. The 10T models would likely require new parallelism, DCP, pipeline parallelism, weight offload, or more HBM capacity.
- No speculative decoding is modeled. A future 5T/10T draft model would need its own measured acceptance curve and verification-cost geometry; reusing K3's DSpark values would not be justified.

Raw sweep: [`scaled_kimi_k3_256k_1m_sol_sweep.csv`](../data.zip); operation breakdown: [`scaled_kimi_k3_256k_operation_breakdown.csv`](../data.zip); metadata: [`scaled_kimi_k3_metadata.json`](scaled_kimi_k3_metadata.json).
