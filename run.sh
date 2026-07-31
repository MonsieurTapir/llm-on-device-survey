#!/usr/bin/env bash
# One-command benchmark (Linux x64 / Apple Silicon): detect the platform, pull
# the latest release kit from GitHub (checksum-verified), unpack it into the
# llm-on-device-survey/ folder next to this script, and hand off to the kit's
# own run.sh. Kits are versioned subfolders of llm-on-device-survey/ and share
# its models/ cache, so a new release skips the model download. Needs curl or
# wget, plus tar + gzip.
#
# Works standalone — no clone required:
#   curl -fsSL https://raw.githubusercontent.com/MonsieurTapir/llm-on-device-survey/main/run.sh | bash
#
# Safe to re-run: an already-unpacked kit is reused, and the kit itself resumes.
set -euo pipefail
REPO_SLUG="MonsieurTapir/llm-on-device-survey"
if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'; else B= G= R= N=; fi
say()  { printf '\n%s→ %s%s\n' "$B" "$*" "$N"; }
fail() { printf '\n%s✗%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

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

# Must match packaging/package.sh's naming: the release URL is built from
# ARTIFACT, and the archive unpacks to llm-on-device-survey/<tag>-<target>/.
ASSET="llm-on-device-survey-${TAG#v}-$TARGET"
KIT="llm-on-device-survey/$TAG-$TARGET"
if [ ! -d "$KIT" ]; then
  BASE="https://github.com/$REPO_SLUG/releases/download/$TAG"
  say "downloading $ASSET.tar.gz"
  http "$BASE/$ASSET.tar.gz" "$ASSET.tar.gz"
  http "$BASE/$ASSET.tar.gz.sha256" "$ASSET.tar.gz.sha256"
  { command -v sha256sum >/dev/null && sha256sum -c "$ASSET.tar.gz.sha256" ||
    shasum -a 256 -c "$ASSET.tar.gz.sha256"; } >/dev/null ||
    fail "checksum mismatch (partial download?) — delete $ASSET.tar.gz and re-run"
  tar -xzf "$ASSET.tar.gz"
  rm -f "$ASSET.tar.gz" "$ASSET.tar.gz.sha256"
fi

[ -n "${BENCH_BOOTSTRAP_ONLY:-}" ] && { printf '%s✓%s bootstrap ok: %s\n' "$G" "$N" "$KIT"; exit 0; }
exec "./$KIT/run.sh"
