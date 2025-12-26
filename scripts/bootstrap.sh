#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: ./scripts/bootstrap.sh [--exb] [--no-install]

  --exb         Also installs optional deps for exb (Streamlit + yt-dlp)
  --no-install  Only creates the venv (no pip installs)
EOF
}

INSTALL=true
INSTALL_EXB=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-install) INSTALL=false ;;
    --exb) INSTALL_EXB=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

pick_python() {
  local candidates=(python3.12 python3.11 python3 /usr/bin/python3)
  local p
  for p in "${candidates[@]}"; do
    if command -v "$p" >/dev/null 2>&1; then
      if "$p" -c 'import sys; print(sys.version.split()[0])' >/dev/null 2>&1; then
        echo "$p"
        return 0
      fi
    fi
  done
  return 1
}

PY="$(pick_python || true)"
if [[ -z "${PY:-}" ]]; then
  echo "No working Python found (tried: python3.12, python3.11, python3, /usr/bin/python3)." >&2
  exit 1
fi

echo "Using Python: $PY"

rm -rf .venv
"$PY" -m venv .venv

if [[ "$INSTALL" == "true" ]]; then
  .venv/bin/python -m pip install -U pip
  .venv/bin/python -m pip install -r requirements.txt
  if [[ "$INSTALL_EXB" == "true" ]]; then
    .venv/bin/python -m pip install -r requirements-exb.txt
  fi
fi

cat <<'EOF'

Done.

Activate:
  source .venv/bin/activate

Run a demo:
  python exa/exa.py

Optional (exb Streamlit + yt-dlp):
  python -m pip install -r requirements-exb.txt
EOF
