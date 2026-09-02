#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

test_repo="${tmpdir}/My Project"
mkdir -p "${test_repo}/.devcontainer"
cp "${repo_root}/.devcontainer/init-env.sh" "${test_repo}/.devcontainer/init-env.sh"

bash "${test_repo}/.devcontainer/init-env.sh"

grep -Fx 'COMPOSE_PROJECT_NAME=my-project' "${test_repo}/.devcontainer/.env"

invalid_repo="${tmpdir}/---"
mkdir -p "${invalid_repo}/.devcontainer"
cp "${repo_root}/.devcontainer/init-env.sh" "${invalid_repo}/.devcontainer/init-env.sh"

if bash "${invalid_repo}/.devcontainer/init-env.sh"; then
    echo 'invalid repository name was accepted' >&2
    exit 1
fi
