# Agent capacity index

## Current research release — 15 October 2026

The published basket is GLM-5.2, Kimi-K3, Qwen3-Coder 480B and
**MAI-2 2.8T**, equally weighted. It uses log-weighted 50–400
SOL TPS/user and combines supporting prefill/decode resources at the saved
1.5133425 new-input/output token ratio. The intervals 50–100, 100–200 and
200–400 receive equal speed-axis weight. Current saved results are in
`results/workload-capacity-oct15-log400/`. Integration is exact over the saved
decode-speed breakpoints, with no interpolation or simulation rerun.

The upstream AISim engine is unchanged. `studies/mai2_swa.py` adds a study-local
MAI architecture adapter using native operators; some dimensions and speculation
are assumed, so this is not an exact Microsoft implementation. This newer research
extension uses the existing October 10 environment and saved study inputs in the
shared workspace; it is not the self-contained October 9 package described below.

Rebuild the current release and explainers from completed saved runs:

```bash
python3 studies/log400_release.py
../sglang-simulator-cpu/.venv-aisim-oct10/bin/python studies/methodology_visual.py
../sglang-simulator-cpu/.venv-aisim-oct10/bin/python studies/performance_knees_visual.py
../sglang-simulator-cpu/.venv-aisim-oct10/bin/python studies/accelerator_comparison_visual.py
```

Open `results/methodology-current/REPORT.html`.

## Original standalone package — 9 October 2026

A standalone reproduction of the **9 October 2026** headline: equal GLM-5.2 and
Kimi-K3 weights, 256 Ki cached tokens, native speculative nextn3 / assumed AL3,
log-capacity AUC over **50–400 SOL TPS/user**, normalized per model to GB300.

The repository is a study layer on **unmodified AISim**, not an AISim fork.
It compares conditional analytical hardware potential, not achieved production
agent throughput. No GPUs, model weights, API credentials, SGLang server,
Dynamo service, or old experiment folders are required.

## Quick reproduction: Python only

From this checkout, with Python 3.12 or newer:

```bash
python3 run.py validate
python3 run.py rebuild
```

Open `results/frozen/REPORT.html`. The output also includes the complete score
and hardware CSV, frontier JSON, target configurations, sensitivity scores, and
validation hashes. **No AISim or third-party Python packages are required.**
This re-integrates frozen frontiers; it does not rerun their timings.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Native timing reproduction: isolated CPU environment

Tested on Linux x86-64, Python 3.12.3. The exact nightly wheel is available from
the public package indexes as of packaging; future availability is not promised.
Use a separate virtual environment, not an existing serving stack:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://pypi.nvidia.com \
  --require-hashes -r environment/requirements-aisim.lock
.venv/bin/python run.py doctor
```

The equivalent faster installation with `uv` is:

```bash
uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com \
  --require-hashes -r environment/requirements-aisim.lock
```

The lock records all 78 packages in the original simulation environment.
It deliberately preserves upstream transitive dependencies rather than
claiming they are all intrinsically necessary for SOL. It includes no Dynamo
or serving-engine installation. `doctor` checks the native binary, both model
configuration hashes, and all hashed AISim distribution files against RECORD.

```bash
# Re-evaluate every saved headline frontier candidate and compare timing/memory.
.venv/bin/python run.py rerun --output results/native-replay

# Small check: one hardware/model and five saved candidates.
.venv/bin/python run.py rerun --hardware 'GB300 NVL72' --model GLM-5.2 \
  --limit 5 --output results/quick-check

# Fresh exhaustive search: every admitted integer batch in the finite layout family.
.venv/bin/python run.py sweep --hardware 'GB300 NVL72' --model GLM-5.2 \
  --output results/gb300-search

# All 17 accelerators and both models. CPU only; more work than frontier replay.
.venv/bin/python run.py sweep --output results/all-search

AGENT_INDEX_TEST_AISIM=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Omit hardware/model filters for the full portfolio; repeat either flag for
multiple selections. Hardware strings are exact keys in `configs/catalog.json`.
Each run requires a fresh output directory if `native_candidates.csv` exists.
Full-portfolio runs also rebuild the index and require agreement with the
frozen scores. A partial run is not labeled a complete index. Interrupted runs
retain their CSV and non-complete status; automatic resume is not implemented.

`rerun` proves saved timing parity, not search completeness. `sweep` searches
the published finite candidate family independently, including losing layouts
and batches; neither claims a globally optimal serving implementation.

Optional editable installation provides `agent-index` instead of `python run.py`:

```bash
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
agent-index --root /path/to/this/checkout validate
```

This is a **checkout-distributed research repository**. Configs and reference
data are not embedded in a standalone package wheel; keep the checkout or
supply `--root`. The checkout-local `run.py` is the simplest supported interface.

## Repository map

| Path | Role |
| --- | --- |
| `configs/catalog.json` | 17 hardware selections, precision and deployment policies |
| `configs/hardware/*.yaml` | Byte-identical frozen hardware resource profiles |
| `configs/workload.json` | Human/machine-readable release definition |
| `src/agent_index/simulation.py` | Pinned native API adapter, memory admission, finite search |
| `src/agent_index/topology.py` | Explicit uncalibrated TPU8i hop overlay |
| `src/agent_index/index.py` | Exact capacity envelope and AUC, standard library only |
| `reference/2026-10-09/` | Frozen frontier inputs, expected scores, immutable input hashes |
| `environment/` | Version/hash lock and native/model provenance |
| `docs/METHODOLOGY.*` | Historical detailed methodology snapshots |
| `tests/` | Analytic integral, score, export, policy and native regression tests |
| `results/` | Generated outputs, deliberately ignored by Git |

## Method and boundaries

For each model and hardware, capacity at speed floor `s` is the maximum
requests / allocated chips among admitted configurations with TPS/user >= `s`.
All PP stages count. No interpolation, extrapolation, or idle deployment-budget
chips are added. Capacity is zero above the fastest qualifying candidate.

We integrate that step function exactly over `d log(s)` from 50 to 400,
divide by the same model's GB300 area, then average the two normalized scores.
Reference fixture scores are **checks**, not inputs to this calculation. The
publication layer reuses the frozen descriptive hardware columns from the
expected CSV; its three score columns are independently recomputed.

The simulator adapter preserves the original study's custom assumptions:
hardware-adapted expert precision, 90% memory admission, metadata guards,
TP/EP confined to eight-chip nodes for HGX/MI355X/Hopper, conservative serial PP,
and the TPU8i overlay. Native Kimi PP layer-rounding warnings and packaged
attention-default fallback are expected and not silently corrected. See the
methodology snapshot for limitations of index reuse, recurrence, SRAM, vendor
topology, speculative cache reuse, and interpretation of native source tags.

The workload JSON documents the pinned implementation; it is not yet a generic
runtime override API. Changes to context, AL, model portfolio, search or resource
assumptions require a new release and regression review. Frozen input hashes
intentionally reject casual in-place edits. Preserve prior releases when evolving
the index; do not relabel changed projections as historical reproduction.

The source checkout commit and installed wheel are identified separately. We
do not know the exact source commit used to build the wheel. Source-building
from the inspected commit is therefore not an asserted equivalent fallback.

## Status and distribution

Packaging validation is recorded in `docs/REPRODUCTION.txt`: all 51 headline
score cells reproduced, all 2,543 saved frontier candidates re-evaluated, and
a fresh 91,863-candidate search recovered the published index to floating-point
precision. A fresh isolated installation also passed runtime identity checks,
native replay and the test suite. These are computational reproducibility
checks, not additional empirical validation of hardware performance.

This local repository is not uploaded or published. Private reference PDFs,
Slack images, raw agent logs and credentials are not included. Numerical
hardware assumptions and attribution are included. See `THIRD_PARTY_NOTICES.md`.
No blanket license has been assigned to the user's original study material.
Choose its publication license before a public release.
