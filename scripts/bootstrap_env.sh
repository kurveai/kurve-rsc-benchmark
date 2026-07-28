#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRAPHREDUCE_ROOT="$(cd -- "$PROJECT_ROOT/../../graphreduce" && pwd)"

if [[ ! -f "$GRAPHREDUCE_ROOT/setup.py" ]]; then
    echo "Local GraphReduce checkout not found at $GRAPHREDUCE_ROOT" >&2
    exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x "/opt/homebrew/opt/python@3.10/bin/python3.10" ]]; then
    PYTHON_BIN="/opt/homebrew/opt/python@3.10/bin/python3.10"
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.10)"
else
    PYTHON_BIN="$(command -v python3)"
fi

"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] >= (3, 10) and sys.version_info[:2] < (3, 13), sys.version' \
    || { echo "Python 3.10, 3.11, or 3.12 is required" >&2; exit 1; }

"$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install --requirement "$PROJECT_ROOT/requirements.local.lock"
"$PYTHON" -m pip install --no-deps --editable "$PROJECT_ROOT"

"$PYTHON" -c "import graphreduce, pathlib; root = pathlib.Path(r'''$GRAPHREDUCE_ROOT''').resolve(); source = pathlib.Path(graphreduce.__file__).resolve(); assert root in source.parents, f'{source} is not under {root}'; print(f'GraphReduce source: {source}')"
"$PYTHON" -m pip check

echo "Environment ready: $PROJECT_ROOT/.venv"
echo "Activate with: source $PROJECT_ROOT/.venv/bin/activate"
