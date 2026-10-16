# Agent capacity index

Portable reproduction of the current **four-model, 23-accelerator** study:
GLM-5.2, Kimi-K3, Qwen3-Coder 480B and MAI-2 2.8T, equally weighted.
The score integrates capacity over **50–400 SOL TPS/user with log-speed weighting**,
including separately provisioned prefill resources at the saved workload ratio.

This is a study layer on **unmodified AISim**, not an AISim fork. Architecture,
memory-admission, topology and search policies live in our Python wrappers.
The MAI adapter uses native operators with explicitly assumed dimensions; it
does not claim to reproduce Microsoft's exact implementation.

## Ready-made visualizations

All three current interactive HTML explainers are included, with their companion
data and downloadable charts:

Start with the [full explainer reading guide](docs/visualizations/README.md), which
connects the narrative, equations, charts and implementation references.

- [Methodology and configuration explorer](docs/visualizations/REPORT.html)
- [Performance knees and bottlenecks](docs/visualizations/PERFORMANCE_KNEES.html)
- [Accelerator comparison charts](docs/visualizations/ACCELERATOR_COMPARISON.html)

GitHub displays HTML source rather than running these pages. Clone/download the
repository and open the files in a browser, or download `visualizations.zip` from
the [reproduction release](https://github.com/jasonlizhengjian/agent-capacity-index/releases/tag/current-reproduction-2026-10-16)
and extract it. Keep the files together so navigation and downloads work.
These are generated snapshots; `python3 reproduce.py` regenerates them in
`results/current/` without modifying the checked-in copies.

## Quick reproduction: no installation needed

Use Python 3.12 on this checkout or the unpacked reproduction bundle:

```bash
python3 reproduce.py
```

This verifies frozen input hashes, recomputes scores, checks all 161 score
fields against the release, and renders local explainers. It needs only the
standard library: **no AISim, GPUs, weights, API keys, private traces, sibling
repositories or network access**.

Outputs in `results/current/`:

- `index.csv`: combined, decode-only, prefill-only and per-model scores.
- `per_model_scores.csv`, `capacity_at_targets.csv`, `ratio_sensitivity.csv`.
- `REPORT.html`, `PERFORMANCE_KNEES.html`, `ACCELERATOR_COMPARISON.html`.
- `provenance.json`, `validation.json`, `reproduction.json`.

Expected combined score: **GB300 = 1; VR200 = 2.3889176225372135**.
This re-integrates saved results; it does **not** rerun timings. The knee-point
teaching curves and power annotations are frozen supporting inputs.

## Rerun AISim on CPU

Supported frozen environment: **Linux x86-64, CPython 3.12**. Download the exact
AISim wheel from this repository's `current-reproduction-2026-10-16` release
into `vendor/`, or unpack the complete release archive, which includes it:

```bash
mkdir -p vendor
gh release download current-reproduction-2026-10-16 \
  --repo jasonlizhengjian/agent-capacity-index --pattern '*.whl' --dir vendor

python3 environment/setup_current.py
.venv-current/bin/python reproduce.py doctor

# Replay every saved decode frontier point and prefill winner.
.venv-current/bin/python reproduce.py replay

# Fresh search: every admitted integer batch in the finite layout family.
.venv-current/bin/python reproduce.py sweep
```

Installation downloads hash-pinned Python dependencies into a separate
`.venv-current`. No serving stack, CUDA or GPU is required. Replay checks
timing and memory against saved values. A full replay or search also rebuilds
and verifies the complete score table. Outputs go to `results/current-replay/`
or `results/current-sweep/`; frozen inputs are never overwritten.

For a smaller check:

```bash
.venv-current/bin/python reproduce.py replay --hardware 'GB300 NVL72' \
  --model GLM-5.2 --limit 5 --output results/quick-replay
.venv-current/bin/python reproduce.py sweep --hardware 'GB300 NVL72' \
  --model 'MAI-2 2.8T' --output results/mai-search
```

Hardware keys are in `reference/2026-10-15/catalog.json`. Model keys are
`GLM-5.2`, `Kimi-K3`, `Qwen-480B` and `MAI-2 2.8T`. Repeat filters
for multiple cases. Partial runs are not published as a complete index.
Completed hardware/model cases resume when code, input and runtime signatures
match; otherwise choose a new output directory. An interrupted case is recomputed.
Unsupported prefill layouts are explicitly recorded in each part's audit.

## Provenance and packaging

Verified on 16 October 2026: fresh isolated environment installation; offline
rebuild from an unpacked archive outside the workspace; native replay of 9,108
decode frontier points and 92 prefill winners (identical timings); and a fresh
search of 431,946 decode plus 286,343 prefill candidates across all 92 cases.
Both native paths reproduce all 161 checked score fields, with maximum relative
error below 2e-16. The sweep checks resulting scores; its per-row timing-error
field is not an independent timing comparison (that check belongs to replay).

`reference/2026-10-15/manifest.json` hashes the frozen profiles, model metadata,
frontiers, prefill winners, workload ratio, expected scores and visual inputs.
Expected scores are checks, not inputs to the calculation.

AISim source: `ai-dynamo/aisimulate`, commit
`2f0f4cd8606d580567b253d02f3d699fe36db865`, version `0.13.0`.
`reference/2026-10-15/runtime.json` pins the wheel, native binary and bundled
model configuration hashes. `doctor` rejects a different binary or configuration.
`environment/requirements-current.lock` preserves the resolved upstream dependency
environment, not a claim that every transitive package is intrinsically needed.

The wheel is Git-ignored and distributed as a release asset / inside the archive.
A Git-only copy supports saved-result rebuilding without it. Building upstream
at the pinned commit is an alternative, but an arbitrary toolchain is not
guaranteed to produce a byte-identical runtime.

```bash
python3 reproduce.py bundle
```

Writes `dist/agent-capacity-index-current.tar.gz` and prints its SHA-256.
The explicit allowlist excludes environments, private references, credentials,
logs and unrelated research scripts. Native setup still needs package-index
access for dependencies; the quick rebuild is fully offline.

## Code map and scope

| Path | Role |
| --- | --- |
| `reproduce.py` | Rebuild, replay, search, validation and bundling |
| `src/agent_index/current_native.py` | Native calls and portable search driver |
| `src/agent_index/current_models.py` | Extracted model/memory rules and MAI adapter |
| `src/agent_index/simulation.py`, `topology.py` | Shared layouts, memory and communication overlay |
| `studies/log400_release.py` | Exact combined capacity AUC and normalization |
| `studies/*visual.py` | HTML renderers used by the entry point |
| `reference/2026-10-15/` | Current frozen inputs and provenance |
| `environment/setup_current.py` | Isolated native environment |

Decode: 262,144 resident tokens, native nextn3 / assumed AL3.
Prefill: 1,024 new tokens over that prefix, without speculation.
Saved new-input/output ratio: 1.513342523752522. Private traces are not
redistributed: the aggregate is an input, not independently re-derived here.

The search preserves the study's memory guards, precision, conservative serial
pipeline schedule, deployment limits and topology assumptions, including MAI's
screen against batches with fewer routed assignments than expert-parallel ranks.
It searches a
finite layout family, not every possible implementation. Reproduction checks
establish computational consistency, not empirical calibration or achievable
production throughput. Queueing, TTFT, tool pauses and temporal duty cycles are
not added. The local explainer details the method and assumptions.

`run.py`, `configs/`, `reference/2026-10-09/` and earlier tests retain the
historical two-model release; **use `reproduce.py` for the current index**.
Historical instructions are preserved in `docs/REPRODUCTION_HISTORY.md`, not
current setup guidance. See `THIRD_PARTY_NOTICES.md` for licensing scope.
