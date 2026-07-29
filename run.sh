#!/usr/bin/env bash
# One-command benchmark (Linux x64 / Apple Silicon): detect the platform, pull
# the latest release kit from GitHub (checksum-verified), unpack it next to
# this script, and hand off to the kit's own run.sh. Needs curl or wget,
# plus tar + gzip.
#
# Works standalone — no clone required:
#   curl -fsSL https://raw.githubusercontent.com/MonsieurTapir/llm-on-device-survey/main/run.sh | bash
#
# Safe to re-run: an already-unpacked kit is reused, and the kit itself resumes.
set -euo pipefail
REPO_SLUG="MonsieurTapir/llm-on-device-survey"
fail() { printf '\n!! %s\n' "$*" >&2; exit 1; }

# http <url> [<out>] — curl or wget, whichever exists; stdout when no <out>.
if command -v curl >/dev/null; then
  http() { curl -fsSL ${2:+-o "$2"} "$1"; }
elif command -v wget >/dev/null; then
  http() { wget -qO "${2:--}" "$1"; }
else
  fail "need curl or wget to download the benchmark kit"
fi

case "$(uname -s)-$(uname -m)" in
Linux-x86_64) TARGET=linux-x64 ;;
Darwin-arm64) TARGET=macos-arm64 ;;
Darwin-x86_64) fail "Intel Macs aren't supported (no prebuilt kit — Apple Silicon only)" ;;
*) fail "no prebuilt kit for $(uname -s)/$(uname -m) — supported: Linux x64, Apple Silicon, Windows x64 (use run.bat)" ;;
esac

TAG=$(http "https://api.github.com/repos/$REPO_SLUG/releases/latest" |
  sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
[ -n "$TAG" ] || fail "cannot resolve the latest release — check your network connection"

KIT="bench-$TAG-$TARGET"
if [ ! -d "$KIT" ]; then
  BASE="https://github.com/$REPO_SLUG/releases/download/$TAG"
  echo "== downloading $KIT.tar.gz"
  http "$BASE/$KIT.tar.gz" "$KIT.tar.gz"
  http "$BASE/$KIT.tar.gz.sha256" "$KIT.tar.gz.sha256"
  { command -v sha256sum >/dev/null && sha256sum -c "$KIT.tar.gz.sha256" ||
    shasum -a 256 -c "$KIT.tar.gz.sha256"; } >/dev/null ||
    fail "checksum mismatch (partial download?) — delete $KIT.tar.gz and re-run"
  tar -xzf "$KIT.tar.gz"
  rm -f "$KIT.tar.gz" "$KIT.tar.gz.sha256"
fi

[ -n "${BENCH_BOOTSTRAP_ONLY:-}" ] && { echo "== bootstrap ok: $KIT"; exit 0; }
exec "./$KIT/run.sh"
