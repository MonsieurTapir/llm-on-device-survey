# On-device LLM inference survey

A survey of what small LLMs cost to run on consumer hardware, measured with
[llama.cpp](https://github.com/ggml-org/llama.cpp). Per device it records a
bare compute/bandwidth **probe**; per model, the runtime-reported
**geometry**, prefill/decode **cost curves** to 8k context, and one
end-to-end **job** measuring load time, time-to-first-token, generation
speed, and memory. Not just what a task took — the parameters of the cost
function, so a task that was never run can be priced.

Each folder is one component with its own README; this file is the map and
how to run the thing.

## Layout

```
schema/        the wire contracts — events (backend → survey) and results
               (survey → analysis) JSON Schemas
tasks/         what the benchmark runs — timed catalog, corpora, the
               brain-check health gate
models/        what it runs on — registry + Hub fetch spec (the model files
               themselves are fetched locally, never tracked)
backends/      one independently-buildable exe per stack (llamacpp/)
survey/        the measuring tool (Python, CLI `survey`) — enumerates cells,
               spawns backends, samples memory, aggregates, publishes
submissions/   published benchmark runs, version-controlled
analysis/      loads submissions into pandas and builds the report
site/          the generated report (build output, never committed)
packaging/     the contributor kit build + the root run.* bootstrap
```

Untracked, you provide: the model files under `models/` (pulled by
`survey fetch`) and optionally `third_party/` (local stack checkouts for
offline builds — each backend otherwise fetches its own pinned stack at
build time).

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
   toolchain ([backends/llamacpp](backends/llamacpp/README.md)). The survey
   tool skips a backend whose exe isn't runnable.

2. **Fetch models and run** — one [uv](https://docs.astral.sh/uv/) project,
   rooted here:

   ```sh
   uv sync

   uv run survey fetch gemma4-E2B          # pull artifacts into models/
   uv run survey check --backend llamacpp  # conformance-check a built exe
   uv run survey plan  --backend llamacpp  # enumerate cells, don't run
   uv run survey run   --backend llamacpp --machine my-box
   ```

   `run` writes two files per backend: `<backend>-raw.json.gz` (raw per-spawn
   traces) and `<backend>-results.json` (aggregated `[p50, max]`).
   `survey aggregate` recomputes the second from the first — no re-inference.

3. **Publish and compare** — stage a run as a submission with
   [`survey publish`](submissions/README.md); build the report locally with
   the [`analysis/`](analysis/README.md) project. CI rebuilds the report from
   the published submissions and deploys it to GitHub Pages on every push to
   `main`.

## The protocol

Per provider, once: a **probe** — bare f16 GEMM and buffer-copy throughput on
the exact device inference selects, no model loaded. The ceilings every model
number is later divided by.

Per `(model, variant, provider)` cell, in order:

1. **Gated sweep** — one spawn, one model load. First the gate: a trivial
   three-turn brain-check, every turn expect-checked — any miss marks the
   cell `unhealthy` and nothing synthetic is measured. Then the sweep, the
   cost function in one instrumented pass: a full-context prefill timed per
   ubatch chunk (the chunk series is the marginal cost curve; its cumulative
   sum is TTFT vs depth), then decode rate at the reached depth and at fills
   below it, walked down the already-primed cache by trimming. Two chunks
   are re-run as half-width dispatches (the lane's dispatch-width
   sensitivity), CPU lanes walk a short thread ladder, and the spawn reports
   the model geometry with allocator-measured memory at a ladder of context
   sizes.
2. **Job** — one real end-to-end task, the per-machine check that the
   sweep-derived parameters reproduce an actual workload. The only spawn the
   memory sampler watches. Skipped when the gate said unhealthy.

One process = one measurement unit — it loads at most one model, on one
provider. Memory stays attributable, failures isolate, and the exe stays
simple: the survey tool owns the matrix.

**Nothing is measured twice.** The sweep times the chunks of one pass rather
than re-running prompts at doubling lengths, and the decode ladder reuses
that pass's primed cache. A soft budget bounds the prefill ladder — on slow
silicon the measured envelope shrinks instead of the time growing, and
everything measured is emitted.

## The contract

The survey tool and the backends couple through exactly three things
(field-level truth: [`schema/`](schema/)):

1. a **CLI** every exe exposes — `providers` / `run` / `sweep` / `probe` /
   `version`;
2. the **events schema** the exe emits on stdout (stderr is for logging;
   nothing is downloaded at runtime);
3. a **`backend.toml`** telling the tool how to invoke the exe:

   ```toml
   key  = "llamacpp"                        # must match the events object's `backend`
   name = "llama.cpp"                       # human label
   cmd  = ["{dir}/build/survey-llamacpp"]   # argv prefix; {dir} substituted, subcommand appended
   ```

Backends are otherwise free: own language, own build, own deps. A versioned
JSON schema — rather than a shared library interface — lets each side evolve
independently and fail loudly at the seam. The `key` also selects the
backend's block in `models/models.yaml` (`llamacpp`→`gguf`); the directory
name is free. `survey check` is the conformance gate.

## Measurement model

The principles; the mechanics live in [survey/README.md](survey/README.md)
and the backend READMEs.

- **Self-timed compute, externally-observed memory.** The exe times its own
  ops with a monotonic clock; the tool samples memory over the process tree
  from outside, on job spawns only.
- **Equal work, deterministically.** Greedy decode, exactly `nb_tokens` per
  turn, EOS ignored, thinking disabled through the model's own chat
  template — every config does the same token count.
- **Equal work is not equal width.** CPU thread counts are the runtime's own
  per-platform defaults; both phase widths travel with the numbers, and a
  thread ladder measures what each core buys. Only measured widths are
  published — no asymptotes fitted past the evidence.
- **First-touch setup is not inference.** Warmup owns pipeline compilation
  and context allocation; the shader cache is pinned per spawn, so the
  sweep's warmup span is a from-scratch first launch and the job runs warm,
  like a user's second launch.
- **Correctness gates timing.** A brain-check must pass before anything is
  timed; this is a timing survey gated by a plumbing smoke test, not a
  quality benchmark.
- **Slow cells aren't ground.** Deadlines and backstops keep slow stacks
  from grinding; unusable cells are listed as `too_slow` or `errored`, never
  given an invented number.
- **Raw traces first, results second.** Aggregation re-runs from the
  persisted raw trace; a metric change is a re-aggregate, not a re-run.
- **Comparison is within a quant label** (`q4` = Q4_K_M); every events
  object embeds the exact stack versions that qualify it.

## Reproducibility

Every events object embeds `<exe> version` — exact library commits, build
flags, `use_mmap`, thread count. Model artifacts are pinned by the registry;
backends fetch their inference stack at build time, pinned; nothing is ever
downloaded at runtime.
