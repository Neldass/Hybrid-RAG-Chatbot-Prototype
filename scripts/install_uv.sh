#!/usr/bin/env bash
set -euo pipefail

# Simple bootstrapper for installing project dependencies via uv
# usage: scripts/install_uv.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
cd "${PROJECT_ROOT}"

if [ ! -d ".venv" ]; then
  echo "Creating uv-managed virtual environment (.venv)..."
  uv venv
else
  echo "Found existing .venv; reusing it."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing hybrid-rag-chatbot in editable mode via uv..."
uv pip install -e .

echo "Done. Activate the env with: source .venv/bin/activate"
