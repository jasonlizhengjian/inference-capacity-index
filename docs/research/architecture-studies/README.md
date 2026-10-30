# Exploratory model architectures

These studies ask how inference performance changes when model design changes:
more total or active parameters, greater depth or width, different expert sizes,
and different attention/cache structures. They preserve our earlier 5T/10T
experiments separately from the **four-model Inference Capacity Index**.

Start with the controlled 5T study for attention/body tradeoffs, the GLM body
study for MoE and depth/width tradeoffs, or the Kimi study for concrete scaling
directions. Download the repository to open the HTML reports locally; GitHub
displays their source. Each study has a `data.zip` with tables and model inputs.

| Study | What it varies | Saved report and data |
| --- | --- | --- |
| Controlled 5T design | 448 body/attention combinations; dense, sparse, compressed and recurrent hybrids; context, concurrency and speculation | [Guide](5t-architecture-design-study/README.md) |
| Kimi 5T/10T scaling | Expert-only, compound Kimi-trend and depth-only scaling; hybrid TP/DP and speculation | [Guide](kimi-k3-scale-up/README.md) |
| 5T sliding-window hybrids | 43 SWA/global and mixed-attention profiles with a common MoE body; Inkling-inspired options | [Guide](5t-swa-architecture-study/README.md) |
| Cached-prefix prefill | New-input size and concurrent-request saturation for the 5T SWA/hybrid models | [Guide](5t-swa-prefill-study/README.md) |
| 10T extension | Fixed-active versus preserved-sparsity scaling; matched 5T/10T comparisons, native geometries, prefill and resource sensitivities | [Guide](10t-frontier-study/README.md) |
| GLM-style body design | 87 definitions varying active/total parameters, expert granularity, shared experts, depth and width | [Guide](glm-body-design-study/README.md) |
| GLM attention dimensions | 58 definitions varying internal dimensions while retaining the GLM-style attention family | [Guide](glm-attention-dimension-study/README.md) |
| Controlled Kimi/DeepSeek 5T | Preserved sparsity versus matched MoE layouts, to separate attention and body effects | [Guide](kimi-vs-deepseek-5t-controlled/README.md) |
| Qwen-style full GQA | Baseline through 5T; joint/deep/wide/many-expert scaling, full-cache traffic, speculation and expert-rounding sensitivity | [Guide](qwen-gqa/README.md) |

## How to interpret these comparisons

Parameter count alone is not a performance specification. Adding unused experts
primarily changes resident weights; increasing top-k, expert width or depth also
changes work on each token. Attention determines cache capacity and read traffic.
The matched comparisons isolate particular choices; other sweeps deliberately
change several dimensions together. Their model definitions identify which.

Compare rows within the same study, hardware profile, precision, context and
speculation setting. An architecture that admits more requests can have a
different limiting resource and best parallelism. A large-model design here is
not a trained checkpoint, an equal-quality substitute, or evidence about the
undisclosed architecture of a particular frontier model.

## Scope and reproduction

**This is a historical archive, not a new score release.** It was packaged from
saved work on 29 October 2026 without rerunning simulations or changing the ICI.
It is on the development branch after `v1.0.0`; that tag and its assets are
unchanged by this addition.

| Aspect | Historical study assumptions |
| --- | --- |
| Legacy engine | Kimi, controlled 5T and 10T provenance records AIC commit `4197820c5e69f2e9c7553f98c98520d602e03845`; the Kimi metadata reports version 0.11.0. The SWA and GLM studies use the included legacy adapters; their individual manifests do not all independently pin an engine commit. |
| Qwen engine | Its saved provenance reports `0.13.0.dev202610050000000066`; inspected source commit `5c466b567d0ef62d983958ddfcbccbad0e201a29`. The wheel's build commit is explicitly unknown. This is not the current headline's pinned release wheel. |
| Rubin | Older studies use 22 TB/s HBM and, in relevant cases, 1.8 TB/s outbound fabric. Some substitute only memory/fabric into GB300; others add a separate full-hardware proxy. These are not the report's revised 19.2/1.5 TB/s profile. Read each saved hardware definition. |
| Speculation | Some runs disable it; others borrow the DSpark acceptance curve, optimize draft length or use a fixed acceptance assumption. GLM-body speculative runs use nextn4/AL3.36. Do not treat all rows as the headline's nextn3/AL3. |
| Parallelism | Search families differ. Kimi includes explicitly hypothetical deployments beyond NVL72. Qwen includes a conservative serial-PP policy. Those rows do not establish performance within one rack or under the current index policy. |
| Analytical extensions | Mixed-attention composites, DSA dimension/index reuse treatment, MoE/routing overlays and shared-expert corrections live in study wrappers. They are not uniformly native AIC functionality. See `source/` and [shared legacy adapters](legacy-support/README.md). |
| Efficiency and validation | SOL and discounted-throughput sensitivities are labeled separately. Native consistency checks are not empirical validation of a hypothetical trained model. |

### What is reproducible from this package?

**Inspect and check saved results:** open the reports, unzip a study's `data.zip`
in that study directory, and read its CSVs, model configurations and hardware
inputs. No simulator installation is required. Most HTML figures embed their
data; linked CSV downloads point to the ZIP. The large 10T report may load slowly.

Verify the packaged files, ZIP members and recorded table row counts with:

```bash
python3 studies/verify_architecture_archive.py
```

**Rerun native timings:** not offered as a turnkey command in this archive.
The archived scripts retain dependencies on the historical research layout,
sibling studies and runtime. They are provided to inspect the calculation, not
as replacements for `reproduce.py`. Some references are outside the curated
archive, including runtime checkouts and large estimate caches. Restoring those
dependencies and verifying hashes is necessary before claiming a native rerun.
Current index commands remain documented in the [main reproduction guide](../../../README.md).

Local workspace prefixes were replaced by `LEGACY_WORKSPACE`, `INDEX_WORKSPACE`,
`INFERENCEX_CHECKOUT` and `RESEARCH_WORKSPACE`. These are descriptive placeholders,
not configured environment variables. Historical report links were repaired and
archive banners added. Numerical CSV values were not changed. ZIPs avoid adding
hundreds of megabytes of repetitive uncompressed tables to the checkout.

[The publication manifest](manifest.json) records original source hashes,
published hashes, ZIP-member hashes and row counts. Original study manifests
remain with their reports; their hashes refer to the pre-normalization bytes.
[The historical hash audit](historical_hash_audit.json) found 305 matching entries
and no mismatches among the comparable entries in those manifests; 57 entries
reference sources not included in this archive. This is a bounded integrity
check, not a claim that all historical dependencies have been recovered.

## Relationship to methodology sensitivity

The [ICI methodology sensitivity study](../methodology-sensitivity/README.md)
reweights the current saved model basket and speed bands. This collection instead
changes model architectures under several historical modeling setups. Its
projections should not be pooled into that table without first rerunning them
under common hardware, runtime, precision, speculation and search policies.
