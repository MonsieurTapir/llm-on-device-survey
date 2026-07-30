#!/usr/bin/env bash
# package.sh <linux-x64|macos-arm64|windows-x64> [tag] — build + stage + zip one
# contributor artifact.
#
# The environment provides the toolchain (cmake, a compiler, the Vulkan SDK on
# linux/windows); this script never installs anything. CI invokes it per-OS
# (see .github/workflows/release.yml); the same invocation works on any
# matching machine. The Linux artifact should be built on the oldest supported
# base (ubuntu-22.04 → glibc 2.35); reproduce locally with:
#
#   docker run --rm -v "$PWD":/src -w /src ubuntu:22.04 bash -c \
#     'apt-get update && apt-get install -y build-essential cmake git curl \
#        libvulkan-dev glslc zip && packaging/package.sh linux-x64'
#
# Staging is a pruned snapshot of the *committed* tree (git archive HEAD) with
# the build products placed exactly where backend.toml already points
# ({dir}/build/bench-llamacpp) — the harness needs no packaged-layout special case.
set -euo pipefail

TARGET="${1:?usage: package.sh <linux-x64|macos-arm64|windows-x64> [tag]}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${2:-$(git -C "$REPO" describe --tags --always)}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

UV_VERSION="0.9.5" # bundled uv release; bump deliberately, the .sha256 asset verifies it

# Windows builds in a SHORT root: llama.cpp's nested vulkan-shaders-gen
# external project alone adds ~150 chars of path, and from a deep checkout
# MSVC's PDB writer (C1041) and CMAKE_OBJECT_PATH_MAX blow the 260-char
# limit — git core.longpaths doesn't cover either.
if [ "$TARGET" = "windows-x64" ]; then
  BUILD="${BENCH_BUILD_DIR:-C:/bb}"
else
  BUILD="$REPO/backends/llamacpp/build-dist/$TARGET"
fi
# Artifact name: the project, then the version, then the platform. The tag is
# the compatibility anchor everywhere else ("v1.0.1"), but the leading v reads
# as noise in a file name, so the artifact carries the bare version.
STAGE_NAME="llm-on-device-survey-${TAG#v}-$TARGET"
DIST="$REPO/dist"
STAGE="$DIST/$STAGE_NAME"

say()  { printf '\n== %s\n' "$*"; }
fail() { printf '\n!! package.sh: %s\n' "$*" >&2; exit 1; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

# ---------------------------------------------------------------- configure + build
# One binary must be right on every contributor box: GGML_BACKEND_DL +
# CPU_ALL_VARIANTS build one dlopen'd CPU module per microarch and pick at
# runtime by feature probe, so GGML_NATIVE=OFF can keep the build host's ISA out
# of the shared code without capping anyone. Every target needs that pair —
# Apple silicon included: it is three microarchs for kernel purposes (M1 is
# ARMv8.5 with no i8mm, M2/M3 add FEAT_I8MM, M4+ add SME), ggml ships
# apple_m1 / apple_m2_m3 / apple_m4 variants accordingly, and with neither
# ALL_VARIANTS nor GGML_CPU_ARM_ARCH set, ggml appends no -mcpu at all and clang
# falls back to its baseline arm64 target — an M1 code path on every Mac, with
# MATMUL_INT8 and SME compiled out. Selection is a real runtime probe there
# (arch/arm/cpu-feats.cpp reads hw.optional.arm.FEAT_*), so the M1 kit still
# picks the M1 module. Vulkan is the universal GPU lane on linux/windows, Metal
# on macOS with embedded shaders.
# GGML_CCACHE off: release builds must be clean and deterministic; on CI
# runners there is no persistent cache to win, only cl.exe flake to lose.
COMMON_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF
  -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DBUILD_SHARED_LIBS=ON)
case "$TARGET" in
linux-x64)
  FLAGS=("${COMMON_FLAGS[@]}" -DGGML_VULKAN=ON
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON "-DCMAKE_INSTALL_RPATH=\$ORIGIN")
  ;;
windows-x64)
  # Ninja + MSVC (cl from the environment — CI runs msvc-dev-cmd first):
  # linear readable logs and single-config output paths; MSBuild's interleaved
  # output has swallowed real errors here before.
  # Compiler pinned to cl: with Ninja, cmake takes the first compiler on
  # PATH, and CI runners carry stray gcc toolchains that would win.
  FLAGS=("${COMMON_FLAGS[@]}" -G Ninja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
    -DGGML_VULKAN=ON)
  ;;
macos-arm64)
  # Deployment target pinned: cmake otherwise defaults to the build host's OS,
  # and a kit built on a macos-15 runner would refuse to launch on older Macs.
  # @loader_path is macOS's $ORIGIN: cmake gives each dylib an
  # install_name of @rpath/<lib>.dylib, and this is the LC_RPATH that resolves
  # it to the flat directory the exe and modules are staged into.
  FLAGS=("${COMMON_FLAGS[@]}" -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON "-DCMAKE_INSTALL_RPATH=@loader_path"
    -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0)
  ;;
*) fail "unknown target $TARGET" ;;
esac

say "configure + build ($TARGET, -j$JOBS)"
cmake -B "$BUILD" -S "$REPO/backends/llamacpp" "${FLAGS[@]}"
cmake --build "$BUILD" --config Release -j "$JOBS"

# ---------------------------------------------------------------- stage
say "stage → $STAGE"
rm -rf "$STAGE" && mkdir -p "$STAGE"
git -C "$REPO" archive HEAD harness schema tasks models.yaml backends/llamacpp/backend.toml \
  | tar -x -C "$STAGE"

EXE_DIR="$STAGE/backends/llamacpp/build"
mkdir -p "$EXE_DIR"
EXE="$(find "$BUILD" -type f \( -name bench-llamacpp -o -name bench-llamacpp.exe \) | head -1)"
[ -n "$EXE" ] || fail "no bench-llamacpp produced under $BUILD"
cp "$EXE" "$EXE_DIR/"
# Shared libs + dlopen'd backend modules, flat next to the exe: the exe's
# $ORIGIN rpath (linux) / @loader_path rpath (macos) / same-dir DLL lookup
# (windows) all resolve there, and ggml_backend_load_all searches the exe's
# directory.
# -type l keeps the soname symlinks (libllama.so.0 → …) the loader asks for;
# they link within the same dir, so they survive the flat copy.
find "$BUILD" \( -type f -o -type l \) \( -name 'libggml*.so*' -o -name 'libllama*.so*' \
  -o -name 'ggml*.dll' -o -name 'llama*.dll' -o -name '*.dylib' \) \
  -exec cp -P {} "$EXE_DIR/" \;

# The whole point of the variant build: a missing module is a silent capability
# downgrade on someone's machine, so count them. x86 declares a dozen variants,
# Apple silicon exactly three (apple_m1 / apple_m2_m3 / apple_m4).
CPU_MODULES=$(find "$EXE_DIR" \( -name '*ggml-cpu-*.so*' -o -name 'ggml-cpu-*.dll' \
  -o -name '*ggml-cpu-*.dylib' \) | wc -l)
case "$TARGET" in
macos-arm64) MIN_CPU_MODULES=3 ;;
*) MIN_CPU_MODULES=4 ;;
esac
[ "$CPU_MODULES" -ge "$MIN_CPU_MODULES" ] ||
  fail "only $CPU_MODULES CPU variant modules staged (expected ≥$MIN_CPU_MODULES)"
case "$TARGET" in
linux-x64 | windows-x64)
  find "$EXE_DIR" \( -name '*ggml-vulkan*' \) | grep -q . || fail "vulkan module missing"
  ;;
macos-arm64)
  find "$EXE_DIR" \( -name '*ggml-metal*' \) | grep -q . || fail "metal module missing"
  ;;
esac

say "smoke-test the staged exe"
case "$TARGET" in
windows-x64) VERSION_JSON="$("$EXE_DIR/bench-llamacpp.exe" version)" ;;
*) VERSION_JSON="$("$EXE_DIR/bench-llamacpp" version)" ;;
esac
[ -n "$VERSION_JSON" ] || fail "staged exe produced no version output"

# `version` reports the features of the CPU module that actually got *loaded*
# (llama_print_system_info queries each registered backend at runtime), so the
# staged kit can be held against the build host's own capabilities. Anything the
# hardware has and the kit doesn't is a variant that failed to build, failed to
# stage, or scored itself out — the silent downgrade this check exists to catch.
# Only ever asserts on features this host has, so an older runner stays valid.
case "$TARGET" in
macos-arm64)
  for feat in DotProd:DOTPROD I8MM:MATMUL_INT8 SME:SME; do
    sysctl_name="hw.optional.arm.FEAT_${feat%%:*}"
    reported="${feat##*:}"
    [ "$(sysctl -n "$sysctl_name" 2>/dev/null || echo 0)" = "1" ] || continue
    case "$VERSION_JSON" in
    *"$reported = 1"*) printf '   host has %s → kit reports %s = 1\n' "${feat%%:*}" "$reported" ;;
    *) fail "host reports $sysctl_name=1 but the staged kit has $reported off —
    the CPU variant build/dispatch is downgrading this machine.
    system_info: $VERSION_JSON" ;;
    esac
  done
  ;;
esac

# ---------------------------------------------------------------- bundle uv
say "bundle uv $UV_VERSION"
mkdir -p "$STAGE/bin"
UV_BASE="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
fetch_verified() { # <asset> — download + verify against its published .sha256
  curl -fsSL -o "$DIST/$1" "$UV_BASE/$1"
  curl -fsSL -o "$DIST/$1.sha256" "$UV_BASE/$1.sha256"
  (cd "$DIST" && sha256 -c "$1.sha256" >/dev/null) || fail "uv checksum mismatch for $1"
}
case "$TARGET" in
linux-x64)
  fetch_verified uv-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf "$DIST/uv-x86_64-unknown-linux-gnu.tar.gz" -C "$STAGE/bin" \
    --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
  ;;
macos-arm64)
  fetch_verified uv-aarch64-apple-darwin.tar.gz
  tar -xzf "$DIST/uv-aarch64-apple-darwin.tar.gz" -C "$STAGE/bin" \
    --strip-components=1 uv-aarch64-apple-darwin/uv
  ;;
windows-x64)
  fetch_verified uv-x86_64-pc-windows-msvc.zip
  # Expand-Archive, not unzip: Git Bash on Windows runners has no unzip.
  powershell.exe -NoProfile -Command \
    "Expand-Archive -Force -Path '$(cygpath -w "$DIST/uv-x86_64-pc-windows-msvc.zip")' -DestinationPath '$(cygpath -w "$STAGE/bin")'"
  ;;
esac

# ---------------------------------------------------------------- entry point + docs
cp "$REPO/packaging/contributor-readme.txt" "$STAGE/README.txt"
case "$TARGET" in
windows-x64) cp "$REPO/packaging/run.ps1" "$REPO/packaging/run.bat" "$STAGE/" ;;
*) cp "$REPO/packaging/run.sh" "$STAGE/" && chmod +x "$STAGE/run.sh" "$STAGE/bin/uv" ;;
esac

mkdir -p "$STAGE/licenses"
LLAMA_LICENSE="$(find "$BUILD/_deps" -maxdepth 2 -name LICENSE -path '*llamacpp*' | head -1)"
[ -n "$LLAMA_LICENSE" ] && cp "$LLAMA_LICENSE" "$STAGE/licenses/llama.cpp-LICENSE"
cat >"$STAGE/licenses/THIRD_PARTY.md" <<'EOF'
Bundled third-party components:
- llama.cpp (MIT) — https://github.com/ggml-org/llama.cpp (LICENSE alongside)
- CLI11 (BSD-3-Clause) — https://github.com/CLIUtils/CLI11 (license header embedded in the built exe's source)
- uv (MIT OR Apache-2.0) — https://github.com/astral-sh/uv
EOF

(cd "$STAGE" && find . -type f | sort | xargs -I{} sh -c 'echo {}') >"$STAGE/MANIFEST.txt"

# ---------------------------------------------------------------- archive
say "archive"
case "$TARGET" in
windows-x64)
  ARCHIVE="$DIST/$STAGE_NAME.zip"
  (cd "$DIST" && powershell.exe -NoProfile -Command \
    "Compress-Archive -Force -Path '$STAGE_NAME' -DestinationPath '$STAGE_NAME.zip'")
  ;;
*)
  ARCHIVE="$DIST/$STAGE_NAME.tar.gz"
  tar -czf "$ARCHIVE" -C "$DIST" "$STAGE_NAME"
  ;;
esac
(cd "$DIST" && sha256 "$(basename "$ARCHIVE")" >"$ARCHIVE.sha256")
say "done: $ARCHIVE"
cat "$ARCHIVE.sha256"
