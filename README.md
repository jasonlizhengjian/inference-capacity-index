# Inference Capacity Index v1

Portable reproduction of the **four-model, 23-accelerator** study:
GLM-5.2, Kimi-K3, Qwen3-Coder 480B and the MAI-Base-2 2.8T proxy, equally weighted.
The score integrates concurrency over **50–400 tokens/s/request with log-speed weighting**,
including separately provisioned prefill resources at the saved workload ratio.

This is a study layer on **unmodified AISim**, not an AISim fork. Architecture,
memory-admission, topology and search policies live in our Python wrappers.
The MAI adapter uses native operators with explicitly assumed dimensions; it
does not claim to reproduce Microsoft's exact implementation.

## Start here

- **Intuition:** the short launch report introduces the ICI. Its public URL will
  be added when published; the repository does not require access to a draft.
- **Official scores:** [Epoch AI ML Hardware Data Explorer](https://epoch.ai/data/machine-learning-hardware).
- **Method and assumptions:** [readable Markdown methodology](docs/METHODOLOGY.md),
  including the workload, precision, model basket, equations and code references.
- **Report reproduction:** [Figures 1–4, their CSVs and commands](docs/report/README.md).
- **Versions and experiments:** [snapshot guide](docs/SNAPSHOTS.md), then
  [sensitivity](docs/research/methodology-sensitivity/README.md) and
  [empirical validation](docs/research/inferencex-validation/README.md).

The report uses the **20 October Rubin revision**, not the no-argument historical
rebuild. Start with `python3 reproduce.py rubin-update`. The native `replay` and
`sweep` commands below still target the original 15 October snapshot; the
Rubin study supplies its separate revised run.

## Latest score revision: Rubin, 20 October 2026

The current repo tables and main explainer include Rubin's revised **19.2 TB/s
HBM** and **1.5 TB/s outbound NVLink** profile. Its combined score changes from
**2.389 to 2.177 GB300e** (−8.86%); other hardware scores are unchanged.
This was a fresh decode and prefill search, not a bandwidth-ratio adjustment.
See [the complete revision, component scores and provenance](docs/research/rubin-spec-update/README.md).

Rebuild the updated table and HTMLs without installing AISim:

```bash
python3 reproduce.py rubin-update
```

Outputs go to `results/rubin-update/`. The `v1.0.0` tag includes the 29 October documentation and analysis refresh.
The original release is preserved at `v1.0.0-original-2026-10-16`; October 15
reference files remain unchanged. For backward compatibility, the no-argument
command below still reproduces that original release. The checked-in
`docs/visualizations/` table and main explainer show the Rubin revision; the
knee-point teaching curves retain their historical inputs. Sensitivity and
validation were refreshed on 29 October to use the report’s hardware revision. This revision retains the original 23-profile
roster, not the later live-sheet additions/exclusions.

## Ready-made visualizations

### Reading order

1. **Method:** [Markdown methodology](docs/METHODOLOGY.md), followed by the
   [interactive explorer guide](docs/visualizations/README.md).
2. **Robustness:** [methodology sensitivity](docs/research/methodology-sensitivity/README.md),
   report-revision profiles, including the linear/log × speed-band table, model-basket choices,
   prefill provisioning and single-point alternatives
   ([interactive tables](docs/research/methodology-sensitivity/index.html),
   [full CSV](docs/research/methodology-sensitivity/score_matrix.csv)).
3. **Empirical grounding:** [InferenceX validation](docs/research/inferencex-validation/README.md).

The sensitivity study uses the **15 October base plus 20 October Rubin revision**; it does
not change the headline or silently include later standalone hardware studies.
The [documentation reading guide](docs/README.md) identifies historical
two-model documents as archived, not current setup or methodology instructions.
Their hash-listed original contents are preserved.

All three interactive HTML explainers are included, with their companion
data and downloadable charts:

Start with the [full explainer reading guide](docs/visualizations/README.md), which
connects the narrative, equations, charts and implementation references.

- [Methodology and configuration explorer](docs/visualizations/REPORT.html)
- [Performance knees and bottlenecks](docs/visualizations/PERFORMANCE_KNEES.html)
- [Accelerator comparison charts](docs/visualizations/ACCELERATOR_COMPARISON.html)
- [InferenceX validation: Kimi primarily, GLM step-timing check](docs/research/inferencex-validation/README.md)
  ([public interactive site](https://inferencex-sol-validation.epoch-3293.chatgpt.site)).
  The detailed report reproduces the approximately 12% average throughput
  difference at five Kimi/GB300 frontier points under the half-SOL scenario,
  including winning layouts, wider-band results and cross-hardware checks.

GitHub displays HTML source rather than running these pages. Clone/download the
repository and open the files in a browser, or download `visualizations.zip` from
the [v1 release](https://github.com/jasonlizhengjian/inference-capacity-index/releases/tag/v1.0.0)
and extract it. Keep the files together so navigation and downloads work.
These are generated snapshots: `python3 reproduce.py rubin-update` regenerates
the revised version in `results/rubin-update/`. The no-argument command retains the original snapshot;
the release ZIP contains the revised explainers. Neither command modifies checked-in copies.

## Original 15 October reproduction: no installation needed

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

Original frozen combined score: **GB300 = 1; VR200 = 2.3889176225372135**.
For the updated **VR200 = 2.177366570368923**, use `rubin-update` above.
This re-integrates saved results; it does **not** rerun timings. The knee-point
teaching curves and power annotations are frozen supporting inputs.

### Reproduce methodology sensitivity

```bash
python3 reproduce.py sensitivity
```

This writes 28 named scenarios to `results/methodology-sensitivity/`, including
interactive tables, full-precision score and rank-change CSVs, per-model
contributions, family-specific robustness ranges and provenance hashes.
It uses the standard library only and checks the headline against the revised
report snapshot. The weighting-by-band matrix integrates exactly over saved breakpoints;
these are re-integrations, not new AISim runs. Min/max ranges are **not confidence
intervals**. See the [study](docs/research/methodology-sensitivity/README.md)
for assumptions and definitions.

## Rerun AISim on CPU

Supported frozen environment: **Linux x86-64, CPython 3.12**. Download the exact
AISim wheel from this repository's `v1.0.0` release
into `vendor/`, or unpack the complete release archive, which includes it:

```bash
mkdir -p vendor
gh release download v1.0.0 \
  --repo jasonlizhengjian/inference-capacity-index --pattern '*.whl' --dir vendor

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

Writes `dist/inference-capacity-index-v1.0.0.tar.gz` and prints its SHA-256.
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
| `studies/rubin_release.py`, `reference/2026-10-20-rubin/` | Offline Rubin revision and saved native sweeps |
| `studies/rubin_oct20.py` | Native old/new Rubin bandwidth re-search |
| `studies/methodology_sensitivity.py` | Frozen-run sensitivity to axis, band, basket and prefill provision |
| `studies/*visual.py` | HTML renderers used by the entry point |
| `reference/2026-10-15/` | Original four-model frozen inputs and provenance |
| `studies/report_figures.py`, `docs/report/` | Portable report figures, CSVs and input hashes |
| `docs/METHODOLOGY.md`, `docs/SNAPSHOTS.md` | Readable method and version boundaries |
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
The release tag is `v1.0.0`. Internal `agent_index` imports and saved CSV column
names are retained for compatibility. The legacy `agent-index` command still
targets the historical release; use `reproduce.py` for v1. Archived documents
retain their original names and assumptions.

Historical instructions are preserved in `docs/REPRODUCTION_HISTORY.md`, not
current setup guidance. See `THIRD_PARTY_NOTICES.md` for licensing scope.

Public-repository safety rules are recorded in [AGENTS.md](AGENTS.md). Publication
must exclude secrets, private traces, credentials and unrelated workspace files.
