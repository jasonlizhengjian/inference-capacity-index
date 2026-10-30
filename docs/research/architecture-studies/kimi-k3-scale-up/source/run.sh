#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIMULATOR_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${SIMULATOR_DIR}/.venv-aic-kimi-k3/bin/python"

"${PYTHON}" "${SCRIPT_DIR}/generate_scaled_configs.py"
exec "${PYTHON}" "${SCRIPT_DIR}/sweep_scaled_decode.py" "$@"
