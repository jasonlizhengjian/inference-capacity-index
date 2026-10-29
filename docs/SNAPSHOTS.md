# Results snapshots and reproduction

Documentation audited 29 October 2026. Dates below identify calculation inputs,
not a guarantee that every vendor specification remains current. All core runs
use the pinned AISimulate 0.13.0 runtime; standalone extensions document their
own assumptions. An index methodology version and a hardware-profile revision
are separate identifiers.

| Result / entry point | Input snapshot and hardware revision | Reproduce / interpret |
| --- | --- | --- |
| Report Figures 1, 3, 4 and main score table | 15 October four-model, 23-profile base **plus 20 October Rubin revision**: HBM 19.2 TB/s, outbound NVLink 1.5 TB/s | `python3 reproduce.py rubin-update`; GB300 1, VR200 2.177367 |
| Report Figure 2 | Fixed Kimi/GB300 layout from the saved search; GB300 unchanged by Rubin revision | `python3 studies/report_figures.py`; pure decode, no prefill denominator |
| `docs/visualizations/REPORT.html`, accelerator chart and `index.csv` | Same report revision | `rubin-update`; checked-in copies are readable outputs, not inputs |
| `v1.0.0-original-2026-10-16`, `reference/2026-10-15` | Original four-model, 23-profile release; Rubin HBM 22 TB/s, outbound NVLink 1.8 TB/s | `python3 reproduce.py rebuild` (also the no-argument default); VR200 2.388918 |
| Methodology sensitivity, 28 scenarios | 15 October base plus 20 October Rubin revision, refreshed 29 October | `python3 reproduce.py sensitivity`; all 644 scores rebuilt, 616 non-Rubin scores unchanged |
| InferenceX validation, 19 October observations | Report hardware revision, refreshed 29 October; five Rubin recipes retimed, other timing probes retained | Commands in its README; not the launch report's October 27 screenshot or an index-wide calibration |
| Performance-knee teaching curves | Original fixed-layout supporting inputs | Generated HTML retains these examples even when the main score table is revised |
| Trainium study, 22 October | Separate hardware/topology scenarios with their own profiles, four-model workloads and GB300 baseline | Separate research, not added to the 23-profile report table |
| Two-model archive, 9 October | Decode-only GLM/Kimi; historical profiles and tooling | `run.py`, `configs/`, old `agent-index` CLI and `docs/METHODOLOGY.txt` / `.html` are archival |

## Reproduce the report, not the historical default

```bash
python3 reproduce.py rubin-update
# Optional figures: matplotlib is needed only for rendering.
.venv-current/bin/python studies/report_figures.py
```

`results/rubin-update/reproduction.json` verifies all 161 score cells, including
unchanged non-Rubin rows. `reference/2026-10-15/manifest.json` and
`reference/2026-10-20-rubin/manifest.json` retain hashes of the exact saved inputs.
Report figure provenance additionally hashes its script, fixed-layout CSV and
both manifests. No timing simulation is required for this path.

For native reproduction, follow the README's pinned environment setup. The
`replay` and `sweep` commands target the **original** 15 October release.
`studies/rubin_oct20.py` separately reproduces the old/new Rubin searches.
The report revision is the original non-Rubin results plus the revised Rubin
results; it is not a new search over all hardware. Historical archives and
hash-listed files are deliberately not rewritten.

## Publication boundaries

The [ML Hardware Data Explorer](https://epoch.ai/data/machine-learning-hardware)
is the official score destination. Live-sheet additions, exclusions and research
extensions can differ from this repository's frozen 23-profile roster. The
report selects ten named profiles from that roster; it is not a top-ten ranking
or an assertion that all 23 profiles are shipping products.

When linking a sensitivity or validation result, retain its date and profile
revision. Sensitivity to band/basket/prefill choices can be re-integrated from
saved curves, but changing context, precision, model dimensions or topology
generally needs a new timing/search run. Historical sensitivity ranges should
not be relabeled as confidence intervals for a revised score.

The [pre-refresh sensitivity and validation archive](research/archives/2026-10-19-pre-rubin-refresh/README.md)
preserves the former outputs, source scripts and hashes.

## v1 release refresh

On 29 October, `v1.0.0` was updated to include the revised Rubin inputs,
refreshed supporting analyses and report documentation. The methodology remains
ICI v1. The original release commit `a7f3e7406fae4c2da0d2597177d413c8e7460b84`
is preserved as `v1.0.0-original-2026-10-16`. The release bundle and visualization
ZIP are refreshed with the tag; the pinned AISim wheel is unchanged.
Use `python3 reproduce.py rubin-update` for the report scores. The no-argument
command still reconstructs the original 15 October inputs for compatibility.
