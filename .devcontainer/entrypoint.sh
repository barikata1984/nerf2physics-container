#!/bin/bash
set -e -o pipefail

# =============================================================================
# Container entrypoint
# - Initialises shell config for the runtime user
# - Drops to non-root user via gosu
# =============================================================================

TARGET_USER="${HOST_USER:-developer}"
TARGET_HOME=$(eval echo "~${TARGET_USER}" 2>/dev/null || echo "/home/${TARGET_USER}")
TARGET_GROUP=$(id -gn "${TARGET_USER}" 2>/dev/null || echo "${TARGET_USER}")
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

# ---- zsh bootstrap (only on first run) --------------------------------------
if [ ! -f "${TARGET_HOME}/.zshrc" ]; then
    mkdir -p "${TARGET_HOME}"
    cat > "${TARGET_HOME}/.zshrc" << 'ZSHRC'
# Minimal zsh config
autoload -Uz compinit && compinit
autoload -Uz vcs_info
precmd() { vcs_info }
zstyle ':vcs_info:git:*' formats ' (%b)'

setopt PROMPT_SUBST
PROMPT='%F{cyan}%~%f${vcs_info_msg_0_} %F{green}>%f '

# History
HISTFILE=~/.zsh_history
HISTSIZE=10000
SAVEHIST=10000
setopt SHARE_HISTORY HIST_IGNORE_DUPS

# Aliases
alias ll='ls -lah --color=auto'
alias la='ls -A --color=auto'

# Note: the `claude` CLI from the VS Code Claude Code extension is put on PATH
# by /etc/zsh/zshenv (installed by the Dockerfile), not here — that file is
# part of the image and refreshes on rebuild, whereas this ~/.zshrc is written
# once and never updated.
ZSHRC
    chown "${TARGET_USER}:${TARGET_GROUP}" "${TARGET_HOME}/.zshrc"
fi

# ---- Ensure user directories exist with correct ownership --------------------
for d in \
    "${TARGET_HOME}/.cache" \
    "${TARGET_HOME}/.local" \
    "${TARGET_HOME}/.config" \
    "${TARGET_HOME}/.claude"; do
    mkdir -p "$d"
    chown "${TARGET_USER}:${TARGET_GROUP}" "$d" 2>/dev/null || true
done

# ---- Drop to non-root user and exec command ---------------------------------
# Doppler secrets are fetched per-shell via /etc/zsh/zshenv (installed by the
# Dockerfile). That ensures `docker compose exec` shells also receive secrets,
# which env vars exported here would not reach.
if [ "$(id -u)" = "0" ] && [ "${TARGET_USER}" != "root" ]; then
    exec gosu "${TARGET_USER}" "$@"
else
    exec "$@"
fi
