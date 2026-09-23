#!/usr/bin/env bash
# One-shot environment bootstrap (also used as the devcontainer postCreateCommand).
# Installs pixi (user-local), the locked pixi env, and builds tiny-cuda-nn. Idempotent.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
if ! command -v pixi >/dev/null 2>&1 && [ ! -x "$HOME/.pixi/bin/pixi" ]; then
  curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="$HOME/.pixi/bin:$PATH"
pixi install --locked
pixi run tcnn-install
pixi run check-cuda
echo "env ready. next: scripts/fetch_assets.sh --dataset <path>, then scripts/reproduce_sledgehammer.sh"
