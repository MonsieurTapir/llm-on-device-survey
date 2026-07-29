# On-device LLM inference benchmark

A harness that measures the cost of running small LLMs on consumer hardware
with [llama.cpp](https://github.com/ggml-org/llama.cpp). Per device it
records a bare compute/bandwidth **probe**; per model, the runtime-reported
**geometry**, prefill/decode **cost curves** to 8k context, and one
end-to-end **job** measuring load time, time-to-first-token, generation
speed, and memory.

**[ARCHITECTURE.md](ARCHITECTURE.md)** is the map — components, the contract,
the measurement model. This file is how to run the thing. Each component's
README has its specifics.

## Layout

```
ARCHITECTURE.md    the map — read this first
schema/            JSON Schemas for the events + results objects — the wire contract
tasks/             task catalog, corpora, brain-check gate
models.yaml        model registry + fetch spec (per model: gguf repo + quants)
backends/
  llamacpp/          llama.cpp backend (C++) — builds build/bench-llamacpp
harness/            Python tool (uv) — bench fetch/check/plan/run/aggregate/publish
results/           harness output; local & gitignored except published/ submissions
analysis/          Python project (uv) — cross-machine comparison notebook
```

Untracked, you provide: `models/` (artifacts pulled by `bench fetch`) and
optionally `third_party/` (local stack checkouts for offline builds or hacking
— each backend otherwise fetches its own pinned stack at build time).

## Contribute a run

One command downloads the latest prebuilt kit for your platform
(checksum-verified), fetches the models, measures, and packs a submission
file to attach to a [submission issue][si]:

```sh
# Linux x64 / Apple Silicon — from a clone, or standalone:
curl -fsSL https://raw.githubusercontent.com/MonsieurTapir/llm-on-device-survey/main/run.sh | bash
```

```powershell
# Windows x64, in PowerShell — from a clone, or standalone:
irm https://raw.githubusercontent.com/MonsieurTapir/llm-on-device-survey/main/run.ps1 | iex
```

On Windows you can also save [`run.bat`][rb] (right-click the link → *Save link
as*) and double-click it; it fetches the rest itself.

[si]: https://github.com/MonsieurTapir/llm-on-device-survey/issues/new?template=submission.yml
[rb]: https://raw.githubusercontent.com/MonsieurTapir/llm-on-device-survey/main/run.bat

## Run from source

1. **Build the backend** — an independent unit with its own README and
   toolchain ([backends/llamacpp](backends/llamacpp/README.md)). The harness
   skips a backend whose exe isn't runnable.

2. **Fetch models and run** — the harness is a [uv](https://docs.astral.sh/uv/)
   project:

   ```sh
   cd harness
   uv sync

   uv run bench fetch gemma4-E2B                              # pull artifacts into ../models
   uv run bench check --backend llamacpp --models ../models   # conformance-check a built exe
   uv run bench plan  --backend llamacpp --models ../models   # enumerate cells, don't run
   uv run bench run   --backend llamacpp --models ../models --tasks ../tasks \
                      --out ../results --machine my-box
   ```

   `run` writes two files per backend: `<backend>-raw.json.gz` (raw per-spawn
   traces) and `<backend>-results.json` (aggregated `[p50, max]`).
   `bench aggregate` recomputes the second from the first — no re-inference.
   `--machine` names the box in the results (default: hostname).

3. **Compare** — load one or many machines' results with the separate
   [`analysis/`](analysis/README.md) project; share a run via
   [`bench publish`](results/published/README.md). The report is rebuilt
   from the published submissions and deployed to GitHub Pages on every
   push to `main`.

## Adding a backend

A backend is one directory under `backends/` that builds to an exe, implements
the `providers` / `run` / `version` CLI, emits schema-valid events on stdout,
and registers itself with a `backend.toml`:

```toml
key  = "llamacpp"                       # must match the events object's `backend`
name = "llama.cpp"                      # human label
cmd  = ["{dir}/build/bench-llamacpp"]   # argv prefix; harness substitutes {dir}, appends the subcommand
```

The `key` also selects the backend's `models.yaml` block (`llamacpp`→`gguf`);
the directory name is free. The contract rules are summarized in
[ARCHITECTURE.md](ARCHITECTURE.md) and [CLAUDE.md](CLAUDE.md); the schemas in
[`schema/`](schema/) are the field-level truth. `bench check` is the
conformance gate.

## Reproducibility

Every events object embeds `<exe> version` — exact library commits, build
flags, `use_mmap`, thread count. A quant label names stack-specific math
(`q4` = Q4_K_M), so results are compared within a label, and each result's
stack versions qualify it.
