# Architecture

A harness to measure the cost of on-device inference across machines — not
just what a task took, but the **parameters of the cost function**: what the
silicon can do (ceiling probes), what the model is (runtime-reported
geometry), how cost scales (prefill/decode sweeps to 8k context), and whether
those parameters reproduce a real workload (one validation job).

| backend | stack | language | artifact |
|---------|-------|----------|----------|
| `llamacpp` | [llama.cpp](https://github.com/ggml-org/llama.cpp) | C++ (native) | one `.gguf`, runs on any provider |

## Components

```
schema/            the wire contract — events (backend → harness) and results
                   (harness → analysis) JSON Schemas
tasks/             what the benchmark runs — timed catalog, corpora, the
                   brain-check provider-health gate
models.yaml        what it runs on — model registry + Hub fetch spec
backends/          one independently-buildable exe per stack (llamacpp)
harness/            backend-agnostic Python tool — enumerates cells, spawns
                   backends, samples memory, aggregates, publishes
analysis/          separate Python project — loads results into pandas,
                   builds the static report site (Models / Fleet / Evidence)
results/published/ version-controlled benchmark submissions
packaging/         the contributor kit: per-platform zips (prebuilt exe +
                   bundled uv + this tree) so a submission needs no toolchain
```

Each component has a README with its specifics; this file is the map.

## The protocol

Per provider, once: a **probe** — bare f16 GEMM and buffer-copy throughput on
the exact device inference selects, no model loaded. The ceilings every model
number is later divided by.

Per `(model, variant, provider)` cell, in order:

1. **Gated sweep** — one spawn, one model load. First the **gate**: the
   brain-check, a trivial three-turn task, every turn expect-checked — any
   miss marks the cell `unhealthy` and the spawn measures nothing synthetic.
   Then the **sweep**, the cost function in one instrumented pass, synthetic
   tokens, no chat semantics: a full-context prefill timed per ubatch chunk
   (the chunk series is the marginal cost curve — its slope is the attention
   term; its cumulative sum is TTFT vs depth), then decode rate at the
   reached depth and at fills below it, walked down the already-primed cache
   by trimming. At two depths the chunk goes in as two half-width dispatches
   instead of one — same tokens, same envelope, so it costs nothing extra —
   and the pair against the full-width cost is the lane's **dispatch-width
   sensitivity**. This spawn is also handed an empty shader-cache directory, so
   its warmup span is the driver compiling this model's pipeline set from
   scratch: the **first-launch cost**, which is otherwise invisible. The spawn
   also reports the model **geometry**: scalars,
   per-layer attention typing, tensor inventory, and the allocator's actual
   buffer sizes — including a **memory cost curve**, the breakdown
   re-measured at a ladder of context sizes. Counted by the runtime, never
   hand-maintained.
2. **Job** — one real end-to-end task, the per-machine check that the
   sweep-derived parameters reproduce an actual workload. The only spawn the
   memory sampler watches. Reuses the shader cache the sweep just populated, so
   it measures a warm process — a second launch, the way a user's would be.
   Skipped when the gate said unhealthy.

One process = one measurement unit — it loads at most one model, on one
provider. Memory stays attributable (one clean timeline per address space,
sampled from outside), failures isolate, and the exe stays simple: no provider
loop, no model cache; the harness owns the matrix.

**Nothing is measured twice.** A prompt is ingested ubatch-by-ubatch anyway,
so separate prefill points at doubling lengths would re-run the same loop —
the sweep instead times the chunks of one pass, and the decode ladder reuses
that pass's primed cache. Under the soft sweep budget (default 60 s) the
ladder stops between items, and each decode point stops early past its own
soft time budget (never below the steady-state minimum): on slow silicon the
measured envelope shrinks instead of the time growing, and everything
measured is emitted. Probe points repeat adaptively (spread ≤ 5% judged from
3 repeats, capped at 5; a ≥ 20 s point runs once); the job keeps fixed **K
in-process iterations × S spawns** (defaults 2×1).

```
models.yaml + models/<M>/gguf/ ──► harness
                                   ├─ <exe> providers --model …      → [cpu, cuda, …]
   per provider:                   ├─ spawn <exe> probe --ep <p>     → ceilings (GEMM, copies)
   per cell (sweep → job):         ├─ spawn <exe> sweep … --gate …   → healthy? + geometry + curve points
   sample rss/vram every ~10 ms ───┤─ spawn <exe> run   … --iters K  → job events   (×S spawns)
   align samples to event windows ─┘
   aggregate ──► results/          (probe points carry adaptive repeats)
```

## The contract

The harness and the backends couple through exactly three things
(detail: [CLAUDE.md](CLAUDE.md), field-level truth: [`schema/`](schema/)):

1. a **CLI** every exe exposes — `providers` / `run` / `sweep` / `probe` /
   `version`;
2. the **events schema** the exe emits on stdout (stderr is for logging;
   nothing is downloaded at runtime);
3. a **`backend.toml`** telling the harness how to invoke the exe.

Backends are otherwise free: own language, own build, own deps. A versioned
JSON schema — rather than a shared library interface — lets each side evolve
independently and fail loudly at the seam: the harness validates events on the
way in and results on the way out. `bench check` conformance-tests a built exe.

## Measurement model

**Self-timed compute, externally-observed memory.** The exe times its own ops
with a monotonic clock; the harness samples memory over the process tree and
never touches the measured process. Only the job spawns are sampled — the
sampled footprint is the validation tick against the allocator-reported
memory model; sweep and probe spawns run unsampled. A single
wall-clock anchor captured at startup maps events onto the memory timeline.

**Equal work, deterministically.** Greedy/argmax decode, exactly `nb_tokens`
per turn, EOS ignored — every config does the same token count. The exe pins
intra-op threads to physical cores. Thinking is disabled
(`enable_thinking=false` through the model's own jinja chat template), so a
reasoning model doesn't burn its decode budget inside `<think>…</think>`.

**One canonical loop.** The exe drives its library's
low-level primitives (not `generate()`), isolating prefill from the first
decode step — TTFT (prefill start → first token) is measured identically
everywhere; decode throughput is steady-state over steps 2..N.

**A real operating point.** The micro-batch is the deployment default
(`n_ubatch = 512`), never tied to context size — measured rates and compute
buffers describe how the stack actually ships. Every output records its
`n_ctx / n_batch / n_ubatch` in the geometry block, so each number is
qualified by the operating point that produced it. One fixed width can't say
how much a lane *minds* that width, so the sweep also splits two of its chunks
into half-width dispatches; a wide GPU pays double digits for the narrower
dispatch, a CPU with a handful of cores pays nothing measurable. It is an
indicator, not a second operating point — the context is still open at the full
`n_ubatch`, and each part pays its own synchronization.

**First-touch setup is not inference.** A GPU backend builds compute pipelines
lazily, on first use, *inside* the graph compute that needs them, and which
pipeline that is depends on the dispatch width. Left alone, seconds of shader
compilation land in whichever measured span happens to run a width first. So the
warmup deliberately walks every width the pass will use — a full ubatch, a full
one over existing history, a half, a short ragged one, then a single-token
decode — and the job's warmup walks its task's own prompts as well. Nothing
measured is charged for setup, and because the harness pins the shader cache to
a directory it controls, the warmup span becomes a *measurement*: the sweep's is
a from-scratch compile, the job's is the warm second launch. macOS exposes no
way to pin Metal's cache, so results record whether the location was pinnable
and only pinned numbers compare across machines.

**Parameters, not just points.** The full-width chunks reduce to a line whose
intercept is the per-dispatch cost and whose slope is the attention term —
between them, the prefill cost function, evaluable at prompt lengths that were
never measured. The prefill pass is also the one measurement family with no
repeats (nothing is measured twice), so the fit's R² and worst residual stand in
for the spread the other families get from repeating.

**Comparison is within a quant label.** A label names stack-specific math
(`q4` = Q4_K_M); every events object embeds
the exact stack versions to qualify a result.

**Correctness gates timing; it is not a quality benchmark.** One trivial
three-turn brain-check runs once per `(model, provider)` before anything
timed — inside the sweep spawn, on the same model load — and every turn must
pass or the cell's sweep and job are skipped.
`expect` strings on trivially-knowable prompts catch plumbing failures (wrong
template, misconfigured provider, degenerate decode). Accepted limitation: a
provider that passes the gate but degenerates at a long-context prefill isn't
caught.

**Slow cells aren't ground.** The exe honours a soft `--deadline-ms`; the
harness hard-kills a pathologically slow iteration and skips monotonically
costlier tasks once one is unusable. Unusable cells are listed explicitly —
split into too-slow vs errored — never given an invented number.

**Memory figures are phase footprints.** Prefill reports its high-water mark;
decode reports both a **peak** (what must fit on the device) and a
**sustained** median (what generation occupies steady-state) — they diverge
when a transient, e.g. an EP compile spike, rides into the decode window. The
reported footprint is RSS, with per-PID NVML VRAM as its own pool and
unified-memory GTT folded into RSS; the why and the platform traps live in
[harness/README.md](harness/README.md).

**Raw traces first, results second.** Inference is expensive; aggregation is
cheap and changes often. The harness persists raw per-spawn traces
(`<backend>-raw.json.gz`), then derives the `[p50, max]` results from them —
`bench aggregate` re-runs only that second step, so a metric change is a
re-aggregate, not a re-run.
