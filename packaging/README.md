# packaging/ — the contributor artifact

One zip per platform turns "clone + toolchain + uv" into "unzip + run". The
zip is a pruned snapshot of the committed tree (`git archive HEAD`) with the
build products staged exactly where `backend.toml` already points
(`backends/llamacpp/build/survey-llamacpp`) plus a bundled `uv` — so the survey tool has no
packaged-layout special case, and the contributor needs nothing preinstalled.

Every archive unpacks into the same `llm-on-device-survey/` folder, one
versioned kit per subfolder. The kits' run scripts fetch models into that
folder's shared `models/` cache (`--models-dir ../models`), so a newer release
extracted next to an older one reuses the ~8 GB already downloaded.

```
llm-on-device-survey/
├── models/                        the shared model cache (fetched on first run)
└── <tag>-<target>/                one kit per release, e.g. v1.0.0-linux-x64
    ├── run.sh | run.bat + run.ps1    the only thing a contributor touches
    ├── README.txt                     ← contributor-readme.txt
    ├── bin/uv[.exe]                   pinned, checksum-verified at package time
    ├── backends/llamacpp/{backend.toml, build/survey-llamacpp + shared libs/modules}
    ├── pyproject.toml  uv.lock  survey/  schema/  tasks/  models/
    ├── licenses/  MANIFEST.txt
```

- **`package.sh <target> [tag]`** — configure → build → stage → verify → zip
  (+ `.sha256`). Targets: `linux-x64`, `macos-arm64`, `windows-x64`. The
  toolchain comes from the environment (CI workflow or your shell); the
  script installs nothing. It hard-fails if the CPU-variant / GPU module set is
  short (a missing dlopen'd module is a silent capability downgrade on someone's
  box, i.e. wrong data), smoke-runs the staged exe, and on macOS holds the
  features the staged kit reports against the ones the build host advertises via
  `hw.optional.arm.FEAT_*` — hardware capability the kit can't use is the same
  wrong data, arriving quietly.
- **Build shape**: every target uses `GGML_BACKEND_DL` +
  `GGML_CPU_ALL_VARIANTS` + shared libs, with `GGML_NATIVE=OFF` and an
  `$ORIGIN` (linux) / `@loader_path` (macOS) rpath, so one binary per platform
  is *correct* on every microarch instead of pinned to the build host's. Plus
  Vulkan on linux/windows, Metal with embedded shaders on macOS. Apple silicon
  needs the variant machinery as much as x86 does: ggml ships `apple_m1` /
  `apple_m2_m3` / `apple_m4` CPU modules because i8mm arrives with M2 and SME
  with M4, and without them clang targets its baseline arm64 CPU — an M1 code
  path on every Mac. Selection is a runtime feature probe on all platforms, so a
  wider kit never breaks an older machine.
- **`run.sh` / `run.ps1`** — the contributor flow: exe smoke test → `uv sync`
  → `survey fetch` → `survey check` → `survey plan` → `survey run` →
  `survey bundle`. Every failure message says what to do next; every step
  resumes on re-run.
- Release builds run in CI (`.github/workflows/release.yml`), one job per
  target, on a tag push. `git archive HEAD` means only *committed* files ship.
