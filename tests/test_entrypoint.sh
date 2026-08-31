#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mkdir -p "${tmpdir}/bin" "${tmpdir}/.pixi/envs/default/bin"
printf '# test fixture\n' > "${tmpdir}/.zshrc"
printf '[workspace]\nname = "test"\n' > "${tmpdir}/pixi.toml"
printf '[build-system]\nrequires = []\n' > "${tmpdir}/pyproject.toml"

printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$*" > "$TEST_PIXI_ARGS_FILE"\nprintf "pixi failed\\n"\nexit 1\n' > "${tmpdir}/bin/pixi"
printf '#!/usr/bin/env bash\nprintf "editable install failed\\n"\nexit 1\n' > "${tmpdir}/.pixi/envs/default/bin/pip"
printf '#!/usr/bin/env bash\nshift\nexec "$@"\n' > "${tmpdir}/bin/gosu"
chmod +x "${tmpdir}/bin/pixi" "${tmpdir}/.pixi/envs/default/bin/pip" "${tmpdir}/bin/gosu"

output="$(
    cd "${tmpdir}"
    HOST_USER=+ WORKSPACE_DIR="${tmpdir}" PATH="${tmpdir}/bin:/usr/bin:/bin" \
        TEST_PIXI_ARGS_FILE="${tmpdir}/pixi-args" \
        bash "${repo_root}/.devcontainer/entrypoint.sh" true 2>&1
)"
printf '%s\n' "${output}"

case "${output}" in
    *"WARNING: pixi install failed (non-fatal, continuing...)"*) ;;
    *) echo "missing pixi install warning" >&2; exit 1 ;;
esac
case "${output}" in
    *"WARNING: editable install failed (non-fatal, continuing...)"*) ;;
    *) echo "missing editable install warning" >&2; exit 1 ;;
esac

grep -Fx "install --locked --manifest-path ${tmpdir}" "${tmpdir}/pixi-args"
