#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

printf '# test fixture\n' > "${tmpdir}/.zshrc"

(
    cd "${tmpdir}"
    HOST_USER=+ WORKSPACE_DIR="${tmpdir}" \
        bash "${repo_root}/.devcontainer/entrypoint.sh" \
        sh -c 'printf "entrypoint-ok\n" > "$1"' sh "${tmpdir}/result"
)

grep -Fx 'entrypoint-ok' "${tmpdir}/result"
