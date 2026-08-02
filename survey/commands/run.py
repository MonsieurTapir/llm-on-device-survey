"""`survey run` — the benchmark proper.

Per provider: one device-ceiling probe. Per `(model, variant, provider)`: one
sweep spawn that first runs the brain-check gate on its own model load (the
exe measures nothing synthetic on an unhealthy provider), then — for healthy
cells — S job spawns of the single validation task. Persists the raw traces,
then aggregates them through the same `aggregate.build` that `survey aggregate`
uses.

Every finished cell is checkpointed to `<backend>-checkpoint.json.gz` in
`--out`: an interrupted run resumes on the next invocation, skipping the cells
already measured, and a completed run deletes the file. A checkpoint from a
different experiment (machine identity, job shape, backend stack) refuses to
resume; `--fresh` discards it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import tempfile
from pathlib import Path
from statistics import median

from .. import aggregate, config, machine, metrics, registry, sampling, schema, spawn
from .._log import log, working
from ..tasks import Task
from ..tasks import load as load_tasks
from .aggregate import RAW_SCHEMA_VERSION
from .merge import MACHINE_IDENTITY

# Below ~4 tok/s a stack is slower than a person reads — effectively unusable.
UNUSABLE_TPS = 4

PROBE_BACKSTOP_S = 300.0

# The sweep's hard kill is a hang guard, never a routine stop (a kill loses
# every completed point). Budgeted sweeps get the budget plus a tail — one
# in-flight chunk and the decode/memory ladders run past the budget by design.
SWEEP_HANG_BACKSTOP_S = 3600.0
SWEEP_TAIL_S = 300.0


def _decode_tps(result: spawn.SpawnResult) -> float | None:
    """Median decode tok/s over a spawn's iterations — the progress heartbeat, not
    the aggregated stat."""
    if not result.events:
        return None
    vals = [t["decode_tps"] for t in metrics.timing_samples(result.events) if t["decode_tps"]]
    return median(vals) if vals else None


def _gate_reason(events: dict) -> str:
    """The unhealthy verdict's evidence: each failed gate turn's decoded text."""
    turns = [
        e
        for e in (events.get("gate") or {}).get("events", [])
        if e["type"] == "turn-end" and not e["expect_pass"]
    ]
    if not turns:
        return "expect failed"
    return "; ".join(f"expect failed (got {t['completion'].strip()!r})" for t in turns)


def _sweep(backend, v, ep, gate: Task, *, cold, deadline_ms, backstop_s, shader_cache):
    """One sweep spawn, carrying the brain-check gate → (status, result). A
    killed sweep is `too_slow`, a spawn that died emitting nothing is
    `errored`, a gate miss is `skipped` (the exe measured no point); partial
    points still count as ok — each point carries its own truth.

    `shader_cache` is this cell's own (empty) cache directory, so the spawn pays
    the pipeline compile deterministically into its `warmup` span instead of
    inheriting whatever the machine happens to have compiled before."""
    s = spawn.sweep(
        backend.cmd,
        model_path=v.model_path,
        quant=v.quant,
        ep=ep,
        gate=gate.spec,
        cold=cold,
        deadline_ms=deadline_ms,
        backstop_s=backstop_s,
        shader_cache=shader_cache,
    )
    if s.events:
        return ("ok" if s.healthy else "skipped"), s
    return ("too_slow" if s.timed_out else "errored"), s


def _job(backend, v, ep, task: Task, *, spawns, iters, deadline_ms, backstop_s, shader_cache):
    """S spawns of the validation job → (status, results) — the only sampled
    spawns (memory is measured here and nowhere else). A bad first spawn is
    not re-ground; a scored job below the usable floor is too_slow, kept apart
    from errored (crash/OOM) so the report can be too.

    Shares the cell's `shader_cache` with the sweep that ran before it, which by
    now has populated it — so the job measures a warm process, the way a second
    launch on a user's machine would."""
    sp: list[spawn.SpawnResult] = []
    for j in range(spawns):
        with working(f"job {task.name} {j + 1}/{spawns}"):
            s = spawn.run(
                backend.cmd,
                model_path=v.model_path,
                quant=v.quant,
                ep=ep,
                task=task.spec,
                iters=iters,
                deadline_ms=deadline_ms,
                backstop_s=backstop_s,
                sample=True,
                shader_cache=shader_cache,
            )
        sp.append(s)
        d = _decode_tps(s)
        note = (
            f"{d:.0f} tok/s"
            if d
            else ("⏱ too slow" if s.timed_out else f"<{s.error}>" if s.error else "—")
        )
        iters_done = len(s.events["iterations"]) if s.events else 0
        tail = f" ({iters_done}/{iters} iters)" if s.truncated else ""
        log(f"    job {task.name} {j + 1}/{spawns}: decode {note}{tail}")
        if j == 0 and (s.timed_out or s.truncated or not s.events):
            log("    job: bad first spawn — skipping remaining spawns")
            break

    if not any(s.events for s in sp):
        status = "too_slow" if any(s.timed_out for s in sp) else "errored"
    elif max((_decode_tps(s) or 0.0) for s in sp) < UNUSABLE_TPS:
        status = "too_slow"
    else:
        status = "ok"
    return status, sp


def _versions_of(raw: dict) -> dict | None:
    """The stack identity from the first events object in a raw/checkpoint doc,
    minus the run-shaped `threads` block — None when nothing carried events."""
    traces = [p.get("trace") for p in raw.get("probes", [])]
    for cell in raw.get("cells", []):
        traces.append((cell.get("sweep") or {}).get("trace"))
        traces += (cell.get("job") or {}).get("spawns", [])
    for t in traces:
        if t and t.get("events"):
            return {k: v for k, v in t["events"]["versions"].items() if k != "threads"}
    return None


def _exe_versions(cmd: list[str]) -> dict | None:
    """The live exe's `version` verb — the same identity object the events
    embed — minus `threads`, which depends on the invocation, not the stack."""
    out = subprocess.run([*cmd, "version"], capture_output=True, text=True, errors="replace")
    try:
        ver = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None  # `survey check` polices this verb; don't also die here
    return {k: v for k, v in ver.items() if k != "threads"}


def _read_checkpoint(path: Path, expect: dict, exe_versions) -> dict | None:
    """An interrupted run's persisted state, or None. A checkpoint that measured
    a different experiment — machine identity, job shape, backend stack —
    refuses to resume rather than mix conditions (same bar as `survey merge`)."""
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            ckpt = json.load(fh)
    except (OSError, EOFError, json.JSONDecodeError) as err:
        log(f"checkpoint {path.name} is unreadable ({err}) — starting fresh")
        return None

    def check(field: str, old, new) -> None:
        if old != new:
            raise SystemExit(
                f"refusing to resume from {path.name}: {field} differs\n"
                f"  checkpoint: {old!r}\n  this run:   {new!r}\n"
                f"re-run with --fresh to discard the checkpoint"
            )

    for field in ("schema_version", "backend", "sampling", "job_spawns", "job_iters"):
        check(field, ckpt.get(field), expect[field])
    for field in MACHINE_IDENTITY:
        check(f"machine.{field}", ckpt["machine"].get(field), expect["machine"].get(field))
    old_versions = _versions_of(ckpt)
    new_versions = exe_versions() if old_versions is not None else None
    if old_versions is not None and new_versions is not None:
        check("versions", old_versions, new_versions)
    return ckpt


def _write_checkpoint(path: Path, doc: dict) -> None:
    """Write + rename, so a kill mid-write never leaves a corrupt checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)
    tmp.replace(path)


def cmd_run(args: argparse.Namespace) -> None:
    backend = config.load_backend(args.backend)
    tasks = load_tasks(args.tasks)
    gate = [t for t in tasks if t.role == "gate"]
    timed = [t for t in tasks if t.role == "timed"]
    if len(gate) != 1:
        raise SystemExit(
            f"the task catalog must define exactly one gate (the sweep spawn carries it), "
            f"found {[t.name for t in gate]}"
        )
    if len(timed) != 1:
        raise SystemExit(
            f"the task catalog must define exactly one validation job, found "
            f"{[t.name for t in timed]}"
        )
    gate_task = gate[0]
    job_task = timed[0]
    variants = registry.variants(args.models, backend.key)
    if not variants:
        raise SystemExit(f"no {backend.key!r} variants under {args.models}")
    variants = registry.select(variants, args.model)

    # Pre-resolve the (variant, lane) cells so progress can show [i/N]; this
    # asks each artifact's `providers` exactly once.
    cells: list[tuple[registry.Variant, str]] = []
    for v in variants:
        lanes = registry.filter_lanes(registry.providers(backend, v.model_path), args.providers)
        cells += [(v, lane.id) for lane in lanes]
    deadline_ms = args.max_ms or None  # soft per-job-spawn time-box
    backstop_s = args.backstop_ms / 1000  # hard kill floor for one runaway iteration
    sweep_deadline_ms = args.sweep_ms or None
    sweep_backstop_s = (
        args.sweep_ms / 1000 + SWEEP_TAIL_S if args.sweep_ms else SWEEP_HANG_BACKSTOP_S
    )
    log(
        f"{len(cells)} cells (gated sweep + job '{job_task.name}' × {args.spawns} spawns); "
        f"probe per provider"
        + (f"; sweep deadline {sweep_deadline_ms / 1000:.0f}s" if sweep_deadline_ms else "")
    )

    touched: set[Path] = set()  # model files already loaded once on this machine
    cold_load: dict[Path, float] = {}  # genuine first-touch load (cold page cache)
    cold_used: set[Path] = set()  # cold_start already attributed
    overruns: list[str] = []  # cells whose rendered prompt exceeded the job's context
    probes: list[dict] = []  # one ceiling probe per provider
    probed: set[str] = set()
    raw_cells: list[dict] = []  # raw per-cell traces → persisted, then aggregated

    # The checkpoint doc is the raw doc plus a `resume` block of run bookkeeping;
    # finishing a run deletes it, so its existence means an interrupted one.
    base = {
        "schema_version": RAW_SCHEMA_VERSION,
        "backend": backend.key,
        "machine": machine.info(args.machine),
        # The run box's sampling sources, recorded so re-aggregating this raw on a
        # different box reproduces the same vram_method (aggregate.sampling_sources).
        "sampling": {"nvml": sampling.NVML_AVAILABLE, "drm": sampling.DRM_AVAILABLE},
        "job_spawns": args.spawns,
        "job_iters": args.iters,
    }
    ckpt_path = args.out / f"{backend.key}-checkpoint.json.gz"
    ckpt = (
        None
        if args.fresh
        else _read_checkpoint(ckpt_path, base, lambda: _exe_versions(backend.cmd))
    )
    done: set[tuple[str, str, str]] = set()
    if ckpt:
        resume = ckpt["resume"]
        touched = {Path(p) for p in resume["touched"]}
        cold_load = {Path(p): ms for p, ms in resume["cold_load"].items()}
        cold_used = {Path(p) for p in resume["cold_used"]}
        overruns = list(resume["overruns"])
        probes = ckpt["probes"]
        probed = {p["provider"] for p in probes}
        raw_cells = ckpt["cells"]
        done = {(c["model"], c["quant"], c["provider"]) for c in raw_cells}
        log(f"resuming from {ckpt_path.name}: {len(done)} cells already measured")
    args.out.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        _write_checkpoint(
            ckpt_path,
            {
                **base,
                "probes": probes,
                "cells": raw_cells,
                "resume": {
                    "touched": sorted(str(p) for p in touched),
                    "cold_load": {str(p): ms for p, ms in cold_load.items()},
                    "cold_used": sorted(str(p) for p in cold_used),
                    "overruns": overruns,
                },
            },
        )

    # Shader caches live under one scratch root for the run and are thrown away
    # with it: every cell's sweep gets an empty directory (so its warmup pays a
    # deterministic compile), its job spawns reuse the now-populated one (so they
    # run warm), and each provider's probe gets its own so it never pre-warms a
    # cell. Nothing outside this root is read or written — a contributor's own
    # driver cache is untouched.
    shader_root = tempfile.TemporaryDirectory(prefix="survey-shaders-")
    cache_root = Path(shader_root.name)
    cache_control = spawn.shader_cache_control()
    if cache_control == "unavailable":
        log(
            "shader cache: not pinnable on this platform — warmup spans reflect "
            "whatever the OS had already compiled"
        )

    for idx, (v, ep) in enumerate(cells, 1):
        head = f"[{idx}/{len(cells)}] {v.model} {v.quant} {ep}"
        if (v.model, v.quant, ep) in done:
            log(f"{head}  already measured — skipped (checkpoint)")
            continue
        lane_dir = ep.replace(":", "-")
        cell_cache = cache_root / f"cell-{idx}-{lane_dir}"

        # Checkpoint before the first spawn touches the model: if this cell is
        # interrupted mid-flight, the resumed run must not call the model cold —
        # the page cache is already warm with it.
        is_cold = v.model_path not in touched
        touched.add(v.model_path)
        checkpoint()

        # 0. one ceiling probe per provider, ahead of its first cell.
        if ep not in probed:
            probed.add(ep)
            with working(f"probe {ep} — device ceilings"):
                p = spawn.probe(
                    backend.cmd,
                    ep=ep,
                    backstop_s=PROBE_BACKSTOP_S,
                    shader_cache=cache_root / f"probe-{lane_dir}",
                )
            probes.append({"provider": ep, "trace": aggregate.trace_of(p)})
            log(f"probe {ep}: {'ok' if p.events else f'✗ {p.error}'}")

        # 1. the sweep — its spawn runs the brain-check gate on its own model
        # load first, then measures its points; track the genuine cold first-touch.
        with working(f"{head}  gate + sweep"):
            sweep_status, sweep_res = _sweep(
                backend,
                v,
                ep,
                gate_task,
                cold=is_cold,
                deadline_ms=sweep_deadline_ms,
                backstop_s=sweep_backstop_s,
                shader_cache=cell_cache,
            )
        # Read straight after the sweep: this is what its compile produced, before
        # the job spawns touch the directory.
        shader_bytes = spawn.shader_cache_bytes(cell_cache)
        if is_cold and sweep_res.events:
            cold_load[v.model_path] = metrics.load_ms(sweep_res.events)
        healthy = sweep_res.events is not None and sweep_res.healthy

        job_status, job_spawns = "skipped", []
        reason = None
        if sweep_res.events is None:
            # No events means no gate verdict either — the cell is unusable and
            # the job is skipped (the same load path would fail again).
            reason = f"{gate_task.name}: no verdict — sweep {sweep_status} ({sweep_res.error})"
            log(f"{head}  gate ? — sweep ✗ {sweep_status} ({sweep_res.error}); skipping job")
        elif not healthy:
            reason = f"{gate_task.name}: {_gate_reason(sweep_res.events)}"
            log(f"{head}  gate ✗ — UNHEALTHY ({reason}); skipping job")
        else:
            pts = sweep_res.events
            depth = sum(p["tokens_count"] for p in pts["prefill_chunks"])
            log(
                f"{head}  gate ✓  sweep: prefill to {depth} tokens "
                f"({len(pts['prefill_chunks'])} chunks) + "
                f"{len(pts['decode_points'])} decode points"
            )
            # 2. the validation job.
            job_status, job_spawns = _job(
                backend,
                v,
                ep,
                job_task,
                spawns=args.spawns,
                iters=args.iters,
                deadline_ms=deadline_ms,
                backstop_s=backstop_s,
                shader_cache=cell_cache,
            )
            # Flag (loudly) any cell whose rendered prompt overran its context,
            # from the exe's own token counts; never trim — adjust by hand.
            budget = job_task.spec.get("max_context_length")
            first = next((s for s in job_spawns if s.events), None)
            if budget and first and (peak := metrics.peak_context(first.events)) > budget:
                log(f"    ⚠ {job_task.name}: sequence {peak} tok > max_context_length {budget}")
                overruns.append(f"{v.model} {v.quant} {ep}: {peak} > {budget}")

        cell = {
            "model": v.model,
            "quant": v.quant,
            "provider": ep,
            "healthy": healthy,
            "reason": reason,
            "cold_ms": None,
            "shader_cache": cache_control,
            "shader_bytes": shader_bytes,
            "sweep": {"status": sweep_status, "trace": aggregate.trace_of(sweep_res)},
            "job": {
                "task": job_task.name,
                "status": job_status,
                "spawns": [aggregate.trace_of(s) for s in job_spawns],
            },
        }
        # The one genuine cold first-touch is attributed once, to the first cell
        # whose job scored (cold_used is run-global).
        if v.model_path in cold_load and v.model_path not in cold_used and job_status == "ok":
            cell["cold_ms"] = cold_load[v.model_path]
            cold_used.add(v.model_path)
        raw_cells.append(cell)
        checkpoint()

    shader_root.cleanup()  # the compiled pipelines were a measurement, not an artifact

    raw = {**base, "probes": probes, "cells": raw_cells}
    raw_path = args.out / f"{backend.key}-raw.json.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as fh:
        json.dump(raw, fh)
    log(f"wrote {raw_path}  (raw traces, {len(raw_cells)} cells)")

    results = aggregate.build(raw)
    schema.validate_results(results)
    out_path = args.out / f"{backend.key}-results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}  ({len(results['runs'])} runs)")
    ckpt_path.unlink(missing_ok=True)  # the run completed; the checkpoint has served

    if overruns:
        log("")
        log(
            f"⚠ {len(overruns)} prompt overrun(s) — these cells exceeded max_context_length; "
            "inference may have truncated. Trim the corpus or raise the task's budget:"
        )
        for line in overruns:
            log(f"    {line}")
