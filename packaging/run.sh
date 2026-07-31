#!/usr/bin/env bash
# One-shot contributor entry point: check the exe runs here, set up a private
# Python, fetch models, measure, and pack a submission tarball. Safe to re-run —
# every step resumes or is idempotent. Nothing is installed outside the
# llm-on-device-survey folder this kit sits in and uv's cache. Models land in
# ../models — the cache every kit version shares, so a newer kit next to this
# one skips the ~8 GB download.
set -euo pipefail
cd "$(dirname "$0")"

UV="bin/uv"
MODELS="../models"
# Same visual language as the survey CLI: bold section arrows, green ✓, red ✗ —
# plain when stdout isn't a terminal.
if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'; else B= G= R= N=; fi
say()  { printf '\n%s→ %s%s\n' "$B" "$*" "$N"; }
done_() { printf '\n%s✓%s %s\n' "$G" "$N" "$*"; }
fail() { printf '\n%s✗%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

[ -x "$UV" ] || fail "bin/uv is missing or not executable — the zip may be
   incomplete; please re-download it."

say "Checking the benchmark exe starts on this machine"
if ! backends/llamacpp/build/survey-llamacpp version >/dev/null 2>exe-error.log; then
  fail "the benchmark exe failed to start (details in exe-error.log).
   Common causes: a Linux distribution older than ~2022 (glibc), or an unusual
   GPU driver stack. Please open an issue and attach exe-error.log:
   https://github.com/MonsieurTapir/llm-on-device-survey/issues/new"
fi

say "Setting up Python (self-contained — nothing touches your system Python)"
"$UV" sync ||
  fail "Python setup failed — check your network connection and re-run ./run.sh
   (it picks up where it left off)."

say "Fetching models (about 8 GB the first time; kit versions share the cache; safe to interrupt and re-run)"
"$UV" run survey fetch --models-dir "$MODELS" ||
  fail "model download failed or was interrupted — re-run ./run.sh to resume."

say "Conformance-checking the exe against the contract"
"$UV" run survey check --backend llamacpp --models "$MODELS" ||
  fail "the exe runs but failed its conformance check — please open an issue
   with the output above."

say "What will be measured on this machine"
"$UV" run survey plan --backend llamacpp --models "$MODELS"
echo
echo "   If a GPU you expected is missing above: on headless Linux boxes your"
echo "   user usually needs the 'render' group (sudo usermod -aG render \$USER,"
echo "   then log out and back in) for the GPU to be visible."

say "Running the benchmark (about 15 min on fast machines, up to an hour on slow ones; keep it plugged in and idle)"
"$UV" run survey run --backend llamacpp --models "$MODELS" --out results/local ||
  fail "the benchmark run failed — please open an issue with the output above."

say "Packing your submission"
"$UV" run survey bundle results/local --out . ||
  fail "bundling failed — please open an issue with the output above."

done_ "All done — attach the submission-*.tar.gz above to a new issue (link above)."
