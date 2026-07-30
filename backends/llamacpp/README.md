# `llamacpp` backend — llama.cpp

The native backend over [llama.cpp](https://github.com/ggml-org/llama.cpp),
driving the low-level `llama.h` API (not `generate()`). Model artifact: a
single `.gguf` file that runs on any provider the build supports (`cpu`,
`cuda`, `metal`, …) — chosen at runtime via `--ep`. Contract:
[ARCHITECTURE.md](../../ARCHITECTURE.md).

## Build

Prerequisites: CMake ≥ 3.18, a C++17 compiler, `git`. A GPU build also needs
its toolchain (CUDA toolkit for `-DGGML_CUDA=ON`, Xcode for `-DGGML_METAL=ON`).

```sh
cd backends/llamacpp
cmake -B build -S .          # add -DGGML_CUDA=ON / -DGGML_VULKAN=ON / -DGGML_METAL=ON
cmake --build build -j
./build/bench-llamacpp version   # sanity check: prints versions JSON
```

`CMakeLists.txt` fetches its own dependencies at configure time, pinned for
reproducibility: llama.cpp at `LLAMACPP_GIT_TAG` (embedded as a subproject,
tests/tools/server off), CLI11 by SHA256; nlohmann/json comes from llama.cpp's
vendored copy. The first configure needs network (cached under `build/`
afterward); pass `-DLLAMACPP_SOURCE_DIR=<path>` (e.g. a
`third_party/llama.cpp` checkout) to build offline or hack on the stack. The
output is the exe named in [`backend.toml`](backend.toml) —
**`build/bench-llamacpp`** — keep the two in sync.

[`patches/`](patches/) carries local fixes applied to the fetched llama.cpp
tree, in listed order, by [`patches/apply.cmake`](patches/apply.cmake). Each
file states the bug, the evidence, and what retires it — read one before
touching it. They are deltas on the compatibility anchor, so the list stays
short: drop a patch as soon as the pin moves past it. A patch is applied at
populate time only — delete the build directory to re-patch — and only to the
*fetched* tree, never to a `LLAMACPP_SOURCE_DIR` working copy.

Format with `clang-format -i main.cpp`, run from this directory so
[`.clang-format`](.clang-format) is the config that applies — it resolves
relative to the file, and pointing the tool at a copy somewhere else silently
reformats to LLVM defaults at 80 columns instead. Validate conformance from the repo
root: `uv run --project harness bench check --backend llamacpp --models models`.

## Contract notes specific to llama.cpp

- **mmap stays off** (`use_mmap=false`) — the shipped deployment
  configuration. Weights are read into allocated buffers at load: the load
  phase pays the full file read, RSS carries the whole weight footprint, and
  the CPU backend's repacked quant tensors exist only in their repacked form
  (no mapped originals kept alongside). See
  [harness/README.md](../../harness/README.md) for how memory is reported.
- `providers` walks the ggml backend registry (`ggml_backend_reg_*`,
  `ggml_backend_dev_get_props`).
- Render prompts via `common_chat_templates_apply` (the model's own jinja
  template) with `enable_thinking=false`. Do **not** use
  `llama_chat_apply_template`: it's a non-jinja built-in approximation with no
  thinking knob that silently diverges from some real templates.
  `enable_thinking=false` lets the template emit its own thinking-off block
  inline; nothing is hardcoded, so the rendered token ids match the other
  stacks.
- Prefill is isolated from the first decode step, so `prefill_tps` and
  `ttft_ms` are both reported.
- `n_threads` / `n_threads_batch` default to `common_cpu_get_num_physical_cores()`
  — every physical core on linux/windows, but only the top performance cluster on
  macOS (`hw.perflevel0.physicalcpu`), so the same call means different silicon per
  platform. `--threads` / `--threads-batch` override each pool independently
  (`0` = that default); the probe's GEMM runs the batch pool, since its shapes are
  batched work. Both resolved counts are recorded in `versions.threads`, alongside
  the llama.cpp commit, compiled backends + toolkit versions, build flags, and
  `use_mmap`. Thread affinity is *not* set — and on Apple it cannot be
  (`ggml_thread_apply_affinity` is a no-op there), so a count is a request for N
  threads, not a choice of which cores run them.
- On CPU lanes `sweep` also walks a thread ladder (`thread_prefill` /
  `thread_decode`), repointing both pools on the live context with
  `llama_set_n_threads` so the loaded model is reused — load + warmup is ~90% of a
  cold spawn and none of the measurement, which is what makes the indicator cost
  seconds. Widths go down from every physical core (`hw.physicalcpu` on Apple,
  where llama.cpp's own default covers the performance cluster only) and pass
  through the lane's default on the way; prefill is timed from an empty cache,
  decode at a fill the pass already primed (deep by preference — shallow, there is
  almost no KV to attend over and the scaling flattens into noise).
- The binary is ad-hoc signed on macOS (`entitlements.plist`,
  `com.apple.security.get-task-allow`) so `task_for_pid` stays available to
  the sampler.
