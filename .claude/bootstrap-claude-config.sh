#!/usr/bin/env bash
#
# Deploy giulioisac/claude-config into an ephemeral Claude Code sandbox.
#
# The repo's own ./install.sh targets a long-lived workstation: it sudo-installs
# gh, downloads the github-mcp-server binary and registers it over stdio. None of
# that works here — a web sandbox has no gh CLI and no gh token, so the wrapper
# would register an MCP server that cannot authenticate. GitHub access in a
# sandbox comes from the managed MCP server instead.
#
# So this deploys the portable half — skills, agents, hooks, CLAUDE.md, settings
# — and skips the machine provisioning. Idempotent: safe to run at every session
# start.
#
# Never exits non-zero. As a cloud setup script a failure here would stop the
# session from starting at all, and a missing config is not worth that.
set -uo pipefail

REPO_URL="${CLAUDE_CONFIG_URL:-https://github.com/giulioisac/claude-config}"
REPO_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.cache/claude-config}"
CLAUDE_DIR="$HOME/.claude"
SELF="$CLAUDE_DIR/bootstrap-claude-config.sh"

log() { echo "[claude-config] $*"; }

main() {

# --- fetch ------------------------------------------------------------------

if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch --depth 1 origin HEAD -q 2>/dev/null &&
    git -C "$REPO_DIR" reset --hard FETCH_HEAD -q 2>/dev/null ||
    log "fetch failed; using the cached clone"
else
  git clone --depth 1 "$REPO_URL" "$REPO_DIR" -q || { log "clone failed; skipping"; return 0; }
fi
log "config at $(git -C "$REPO_DIR" rev-parse --short HEAD)"

# --- link --------------------------------------------------------------------

# Symlink src -> dst, but never destroy a real file the sandbox put there.
# install.sh backs such files up and replaces them; here they belong to the
# managed image, so we yield instead.
link() {
  local src="$1" dst="$2"
  if [[ -e "$dst" && ! -L "$dst" ]]; then
    log "skip $dst (pre-existing, not ours)"
    return
  fi
  mkdir -p "$(dirname "$dst")"
  ln -sfn "$src" "$dst"
}

# Skills are linked ONE BY ONE. ~/.claude/skills is already populated by the
# sandbox image (docx, pdf, pptx, xlsx, ...) and tracked in its manifest.json,
# so linking the directory wholesale — which is what install.sh does — would
# move those aside and lose them for the session.
mkdir -p "$CLAUDE_DIR/skills"
n=0
for d in "$REPO_DIR"/claude/skills/*/; do
  d="${d%/}"
  link "$d" "$CLAUDE_DIR/skills/$(basename "$d")"
  n=$((n + 1))
done
log "linked $n skills"

# These have no counterpart in the image, so the whole directory can be linked.
link "$REPO_DIR/claude/agents"       "$CLAUDE_DIR/agents"
link "$REPO_DIR/claude/hooks"        "$CLAUDE_DIR/hooks"
link "$REPO_DIR/claude/CLAUDE.md"    "$CLAUDE_DIR/CLAUDE.md"
link "$REPO_DIR/claude/statusline.js" "$CLAUDE_DIR/statusline.js"

# --- settings ----------------------------------------------------------------

# Rendered, not linked — the template carries a per-machine node path. Left
# alongside the image's launcher-settings.json, which is a separate file and is
# not touched.
if [[ -f "$REPO_DIR/claude/settings.template.json" ]]; then
  sed -e "s#__NODE__#$(command -v node)#g" -e "s#__HOME__#${HOME}#g" \
    "$REPO_DIR/claude/settings.template.json" > "$CLAUDE_DIR/settings.json"
  log "rendered settings.json"
fi

# --- self-perpetuate ----------------------------------------------------------

# A cloud setup script runs once, then the filesystem is snapshotted and later
# sessions skip it -- so the clone above would sit frozen at snapshot age for up
# to a week. Keep a copy on disk and register it as a user-level SessionStart
# hook: the snapshot carries both, and every later session re-runs the fetch.
# The hook goes into the rendered settings.json rather than the template, so
# only sandboxes get it.
if [[ -f "${BASH_SOURCE[0]}" && "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")" != "$SELF" ]]; then
  cp "${BASH_SOURCE[0]}" "$SELF" && chmod +x "$SELF"
fi

if [[ -f "$SELF" && -f "$CLAUDE_DIR/settings.json" ]] && command -v python3 >/dev/null; then
  SELF="$SELF" python3 - "$CLAUDE_DIR/settings.json" <<'PY' || log "could not register the refresh hook"
import json, os, sys

path, self_path = sys.argv[1], os.environ["SELF"]
settings = json.load(open(path))
entry = {"type": "command", "command": f"bash {self_path}"}
groups = settings.setdefault("hooks", {}).setdefault("SessionStart", [])
if not any(entry in g.get("hooks", []) for g in groups):
    groups.append({"hooks": [entry]})
    json.dump(settings, open(path, "w"), indent=2)
PY
fi

# --- git identity -------------------------------------------------------------

# Commits from this account go out under the owner's name. The sandbox image
# leaves the global identity set to someone else, so a repo cloned here without
# a repo-local override would be committed under the wrong name.
git config --global user.name "giulioisac"
git config --global user.email "giulioisac@gmail.com"

# --- deps --------------------------------------------------------------------

# Only what the hooks need. ruff backs the PostToolUse hook; without it every
# Edit would fire a hook that cannot run.
command -v ruff >/dev/null || pip install --quiet ruff 2>/dev/null || log "ruff unavailable"

log "ready"
}

main || log "bootstrap failed; continuing without the config"
exit 0
