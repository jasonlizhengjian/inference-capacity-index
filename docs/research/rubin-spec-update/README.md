# Rubin bandwidth revision — 20 October 2026

The revised VR200 combined Inference Capacity Index is **2.177 GB300e**, versus
**2.389** in the original v1.0.0 snapshot (−8.86%). This is a native AISim
re-search of both phases, not a rescaling of the final score. Only Rubin changes.

## What changed

| Per-GPU input | Frozen release | Revised profile |
| --- | ---: | ---: |
| HBM bandwidth | 22 TB/s | 19.2 TB/s |
| Outbound NVLink bandwidth | 1.8 TB/s | 1.5 TB/s |
| Corresponding bidirectional NVLink | 3.6 TB/s | 3.0 TB/s |

Decimal bytes/s are used. All three communication-bandwidth fields in the
profile change together; the searched deployments remain within the same
72-chip scale-up budget. HBM capacity (288 decimal GB), dense compute
(4 PF BF16, 17.5 PF FP8, 35 PF FP4), precision, memory guards, native nextn3/AL3,
model basket and workload are unchanged. This does not newly validate those
other assumptions or introduce a Rubin-specific kernel model.

Sources retained from the October 20 run: NVIDIA's
[Vera Rubin product page](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)
and [NVLink page](https://www.nvidia.com/en-us/data-center/nvlink/).
The contemporaneous [October datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/56p68o47y6f1yucpubl0ump1c0aif422.pdf)
(`5697927. Oct26`, pp. 8–9) reports rounded system/superchip values, including
38.5 TB/s HBM per two-GPU superchip and 216 TB/s bidirectional NVLink per rack.
We preserve the actual run input **19.2**, rather than silently replacing it
with 19.25 or dividing the rounded rack HBM total by 72. Older NVIDIA material
gave 22 TB/s and 3.6 TB/s; this is a dated specification revision, not a change
to the scoring methodology. Live vendor pages can change after the snapshot.

## Results

| Model | Old combined | Revised combined | Change |
| --- | ---: | ---: | ---: |
| GLM-5.2 | 1.236 | 1.225 | −0.91% |
| Kimi-K3 | 2.743 | 2.476 | −9.70% |
| Qwen3-Coder 480B | 2.619 | 2.355 | −10.09% |
| MAI-Base-2 proxy | 2.959 | 2.654 | −10.30% |
| Equal-model average | **2.389** | **2.177** | **−8.86%** |

Decode-only falls from **2.646 to 2.354** (−11.04%); prefill-only falls from
**1.760 to 1.714** (−2.61%). The combined score is still calculated by adding
the prefill and decode chip requirements at each speed, not averaging the
two phase scores. GB300 normalization, log-speed 50–400 integration and the
1.513342523752522 input/output ratio are unchanged.

Different model effects reflect the same resource change applied to different
workloads and newly selected optimal layouts/batches. These percentages are
modeled effects, not measured changes in hardware throughput.

## Reproduce

From the repo root, with standard-library Python only:

```bash
python3 reproduce.py rubin-update
```

This verifies the base and revision manifests, substitutes only the four Rubin
decode frontiers and optimized prefill rates, and regenerates the entire
23-profile table and interactive explainers in `results/rubin-update/`.
Checks cover all 161 headline/phase/component score cells: the 22 other
hardware rows must remain unchanged, and Rubin must match the saved sweep.
No simulations or network are needed for this mode.

To repeat the actual native comparison, after the pinned environment setup in
the [main instructions](../../../README.md):

```bash
.venv-current/bin/python studies/rubin_oct20.py
```

This reruns both old and revised profiles across the saved finite layout
family and all admitted integer batches. It writes to
`results/rubin-spec-update-2026-10-20/`, never to frozen references. On a changed
code/runtime signature, use a fresh output directory by setting the script's
`OUT` constant; do not overwrite a prior run's provenance. The runner imports
the shared `studies/gb300_sensitivity/run.py` and `curvature.py` evaluators;
including them here does not publish their separate studies or run them.

Per profile, the completed search evaluated **20,231 decode** and **13,263
prefill** candidates. Counts are identical before/after because admission and
layout constraints did not change. Baseline scores reproduce the original
release to floating-point precision (maximum relative discrepancy 2.22e−16).
The same pinned AISim 0.13.0 binary/source commit is used:
`2f0f4cd8606d580567b253d02f3d699fe36db865`.

**Portable recheck, 21 October:** a clean checkout of the staged publication
files (without the unrelated local interconnect module) successfully rebuilt
the offline table and reran both native profiles: **66,988 candidate evaluations**
in total. Its comparison CSV and validation JSON are byte-identical to the
saved October 20 artifacts. Inline JavaScript in the three generated HTMLs
was syntax-checked; this is not a browser-rendering check.

## Evidence and boundaries

- [Full-precision comparison](../../../reference/2026-10-20-rubin/comparison.csv).
- [Validation and candidate counts](../../../reference/2026-10-20-rubin/validation.json).
- [Original run provenance](../../../reference/2026-10-20-rubin/provenance.json): input/code hashes, literal specs and runtime.
- [Revised hardware YAML](../../../reference/2026-10-20-rubin/hardware/vr200_deployment.yaml).
- [Old native results](../../../reference/2026-10-20-rubin/frozen_baseline/) and
  [revised native results](../../../reference/2026-10-20-rubin/october_specs/):
  frontiers, prefill winners, layout audits and metrics.
- [Offline integration implementation](../../../studies/rubin_release.py).

The original provenance deliberately retains its source hashes. It also hashes
`src/agent_index/interconnect.py`, an unrelated local audit module not imported
by this run and not required for reproduction. The portable rerun records the
files actually present in its checkout, so its provenance need not be byte-for-byte
identical to that historical signature; timings and scores are the checks.

The original tag and October 15 reference files are immutable. Historical
methodology-sensitivity tables, InferenceX results and fixed-layout knee-point
teaching curves are not silently relabeled as new runs. Current main-explainer
curves, prefill winners and accelerator score charts use the revision.
This update is limited to Rubin: it does not incorporate later hardware-roster,
A100, topology or power studies into the original 23-profile table. None of the
native AISim source, empirical efficiency assumptions or workload assumptions
were changed. Sites and spreadsheets are separate publication destinations and
are not updated by this repository commit.
