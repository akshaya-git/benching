#!/usr/bin/env bash
#
# benching — one-shot setup check for Apple Silicon.
#
# benching itself has no Python dependencies (stdlib only). This script
# verifies the runtime prerequisites, seeds config.json from the example on
# first run, and tells you exactly what (if anything) is missing.
#
# Usage:  ./install.sh
#
set -u

cd "$(dirname "$0")"

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m⚠\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; }

have() { command -v "$1" >/dev/null 2>&1; }

echo
echo "  benching — prerequisite check"
echo "  ----------------------------------------"

# --- Python (required) ---
PY="$(command -v python3 || true)"
if [ -z "$PY" ]; then
  fail "python3 not found on PATH — install Python 3.10+ (brew install python)"
else
  PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
  PYOK="$("$PY" -c 'import sys; print(1 if sys.version_info >= (3,10) else 0)' 2>/dev/null)"
  if [ "$PYOK" = "1" ]; then
    ok "python $PYVER ($PY)"
  else
    fail "python $PYVER is too old — benching needs 3.10+ (brew install python)"
  fi
fi

# --- Framework CLIs (optional: only needed for the frameworks you run) ---
echo
echo "  Framework CLIs (install only what you benchmark):"
check_cli() { # name  hint
  if have "$1"; then ok "$1 — $(command -v "$1")"; else warn "$1 not found — $2"; fi
}
check_cli omlx      "brew install omlx  (or see the OMLX repo)"
check_cli mtplx     "see the MTPLX repo for install"
check_cli mlx-serve "npm i -g mlx-serve  (or the MLX-Serve repo)"

# MLX-VLM needs a python interpreter with the mlx_vlm package.
if "$PY" -c 'import mlx_vlm' >/dev/null 2>&1; then
  ok "mlx_vlm importable by $PY"
else
  warn "mlx_vlm not importable by $PY — MLX-VLM framework will be skipped"
  warn "   (pip install mlx-vlm into the interpreter you point 'mlxlm' at)"
fi

# --- Agent harnesses (optional: only needed for the agent rows) ---
echo
echo "  Agent harness CLIs (optional):"
check_cli pi        "npm i -g @earendil-works/pi-coding-agent"
check_cli opencode  "curl -fsSL https://opencode.ai/install | bash"
check_cli goose     "brew install goose  (or the Goose repo)"
have node && ok "node — $(command -v node)" || warn "node not found (needed by pi/opencode)"

# --- Seed config.json on first run ---
echo
if [ -f config.json ]; then
  ok "config.json already present (left untouched)"
else
  if [ -f config.example.json ]; then
    cp config.example.json config.json
    ok "seeded config.json from config.example.json — edit it to match your machine"
  else
    warn "config.example.json missing — server will fall back to built-in defaults"
  fi
fi

# --- Runtime dirs ---
mkdir -p runs outputs logs work harness-configs 2>/dev/null
ok "runtime dirs ready (runs/ outputs/ logs/ work/ harness-configs/)"

echo
echo "  ----------------------------------------"
echo "  Next:"
echo "    1. Edit config.json  (models, ports, start_cmd per framework)"
echo "    2. python3 server.py            (or: make run)"
echo "    3. Open http://localhost:7090"
echo
exit 0
