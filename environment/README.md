# Runtime provenance

## Current four-model release

Use `python3 environment/setup_current.py` from the repository root. This creates
`.venv-current` using `requirements-current.lock` and the verified wheel in
`vendor/`. Current provenance is `reference/2026-10-15/runtime.json`: AISim 0.13.0,
source commit `2f0f4cd8606d580567b253d02f3d699fe36db865`. Unlike the historical
package below, the complete current reproduction archive includes that wheel.
Supported native installation: Linux x86-64, CPython 3.12.

## Historical October 9 environment

`requirements-aisim-frozen.txt` records the 78 distributions, excluding pip,
installed in the original CPU analytical runtime on 2026-10-09. This is an
observed version snapshot, not a hash-locked supply-chain manifest and not a
claim that every package is required by the extracted study code. It records
AISimulate's substantial transitive environment rather than guessing a
minimal subset. `requirements-aisim.lock` additionally pins accepted artifact
hashes. The exact development wheel and all 78 packages were successfully
downloaded into a fresh isolated environment during packaging on 9 October
2026; `doctor` matched the original runtime/model hashes and 2,228 installed
AISim file hashes. Future package-index availability is not guaranteed.

`runtime.json` records Python version, native runtime hash, bundled model
configuration hashes, and a separately inspected clean source checkout.
That checkout is not asserted to be the wheel's build commit. The wheel and
model configuration files are not redistributed. Rebuilding an index from
frozen CSVs does not require AISimulate or this full environment; native
re-evaluation does. No GPU, model weights, API keys, or serving engine are
needed for the native analytical calculation.

`docs/METHODOLOGY.txt` and `.html` are exact historical snapshots. Their
references to the old study filenames describe provenance; they are not
entry-point instructions for the extracted repository. Refer to this
repository's README for runnable interfaces.
