#!/usr/bin/env bash
# scripts/setup_env.sh — one-command environment bootstrap.
#
# Creates a Python 3.11 venv with uv, installs all dependencies in editable mode,
# and installs the pre-commit hooks.
#
# Usage:
#   bash scripts/setup_env.sh
#   make setup                 # convenience wrapper

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# 1. uv
# ---------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "Installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# 2. .venv
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR with Python $PYTHON_VERSION…"
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# 3. dependencies
# ---------------------------------------------------------------------------
echo "Installing dependencies…"
uv pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# 4. pre-commit
# ---------------------------------------------------------------------------
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
fi

# ---------------------------------------------------------------------------
# 5. sanity
# ---------------------------------------------------------------------------
echo ""
echo "=== Environment sanity check ==="
python -c "import sys; print(f'python: {sys.version.split()[0]}')"
python -m src.utils.device || echo "(device check failed — investigate)"

cat <<'EOF'

==============================================================================
Done. Activate the venv with:

    source .venv/bin/activate

Next:
    make data           # download Norman + DepMap
    make pipeline       # end-to-end (data → vae → dynamics → rl → evaluate)
    make help           # see all targets
==============================================================================
EOF
