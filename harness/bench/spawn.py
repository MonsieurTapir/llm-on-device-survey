"""Run one backend process, sampling memory from the outside.

A spawn is the atomic unit: one `run`, `sweep`, or `probe` invocation. The
memory sampler is attached only where its samples are consumed — the
validation-job spawns (`sample=True`); sweep and probe spawns run unsampled
and carry an empty sample series. One process loads at most one model and
runs one provider, so a sampled spawn is a single clean memory timeline.
stdout is the events object and nothing else (contract); we parse and
schema-validate it before anyone downstream trusts a number.

A failed `expect` exits nonzero but still emits a valid events object (it
carries the decoded text) — we keep that. Only missing/garbled stdout is a
hard error.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import psutil

from . import schema
from .sampling import Sampler

# Where each driver family keeps its compiled-shader cache. A GPU backend builds
# compute pipelines lazily, on first use, inside the graph compute that needs
# them, and the driver then caches the result on disk — making the cost one-time
# per machine per exe and therefore invisible on any run but the first. Pointing
# these at a directory we control makes it deterministic instead: a spawn handed
# an empty directory always pays the compile (into its `warmup` span, which is
# built to absorb it), and a later spawn handed the now-populated directory
# always runs warm.
#
# We set all of them unconditionally rather than detecting the driver. Each is
# ignored by the families it doesn't belong to — verified both ways between the
# NVIDIA and Mesa knobs — so a lookup table would be maintenance with no payoff.
_SHADER_CACHE_VARS = {
    "__GL_SHADER_DISK_CACHE_PATH": "nvidia",  # NVIDIA proprietary (Unix)
    "MESA_SHADER_CACHE_DIR": "mesa",  # RADV / ANV / NVK
    "CUDA_CACHE_PATH": "cuda",  # CUDA JIT cache
}


def shader_cache_control() -> str:
    """Whether this platform lets us pin the shader cache location.

    macOS keeps Metal's pipeline cache under OS control with no documented path
    or disable knob, so a Mac warmup span is cold or warm depending on machine
    history we can't see or set. Recorded in the results so a reader knows which
    kind of number they're looking at rather than comparing the two."""
    return "unavailable" if sys.platform == "darwin" else "redirected"


def _shader_cache_env(cache_dir: Path | None) -> dict[str, str]:
    """Environment overrides pointing every known shader cache at `cache_dir`."""
    if cache_dir is None:
        return {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    return dict.fromkeys(_SHADER_CACHE_VARS, str(cache_dir))


def shader_cache_bytes(cache_dir: Path | None) -> int | None:
    """Total bytes a spawn's compile left in its cache directory — the on-disk
    size of this model's compiled pipeline set. None when unmeasurable."""
    if cache_dir is None or not cache_dir.exists():
        return None
    return sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and any children, then reap, so a backstop'd spawn leaves
    nothing behind (the backends are single-process, but be defensive)."""
    try:
        root = psutil.Process(proc.pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        procs = []
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    proc.kill()  # ensure the Popen handle itself is signalled


@dataclass
class SpawnResult:
    events: dict | None  # validated events object, or None on hard failure
    samples: list[dict]  # (wall_unix_ns, rss, vram) over the process tree
    cold: bool  # first process to touch this model file (cold page cache)
    error: str | None  # last stderr line when stdout wasn't a valid events object
    timed_out: bool = False  # killed at the harness backstop

    @property
    def healthy(self) -> bool:
        # sweep/probe events carry no expects — they are healthy by existing.
        return self.events is not None and self.events.get("healthy", True)

    @property
    def truncated(self) -> bool:
        """Soft deadline cut the in-process loop below the requested K — a signal
        the cell is slow, so the harness can stop re-spawning it."""
        return (
            self.events is not None
            and len(self.events.get("iterations") or []) < self._iters_requested
        )

    _iters_requested: int = 1


def _execute(cmd: list[str], *, backstop_s: float | None, cold: bool = False,
             iters: int = 1, sample: bool = False,
             shader_cache: Path | None = None) -> SpawnResult:
    """Spawn, sample (job spawns only), backstop-kill if needed, parse +
    schema-validate stdout. `shader_cache` pins where the driver caches compiled
    pipelines, so whether this spawn pays the compile is our choice, not history."""
    killed = False
    env = {**os.environ, **_shader_cache_env(shader_cache)}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            errors="replace", env=env)
    with Sampler(proc.pid) if sample else nullcontext() as sampler:
        try:
            stdout, stderr = proc.communicate(timeout=backstop_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            stdout, stderr = proc.communicate()
            killed = True
    samples = sampler.samples if sampler else []

    if killed:
        reason = f"killed at backstop ({backstop_s:.0f}s)"
        return SpawnResult(events=None, samples=samples, cold=cold, error=reason,
                           timed_out=True, _iters_requested=iters)

    try:
        events = json.loads(stdout)
    except json.JSONDecodeError:
        reason = (stderr.strip().splitlines() or ["no stdout"])[-1][:200]
        return SpawnResult(events=None, samples=samples, cold=cold, error=reason,
                           _iters_requested=iters)

    schema.validate_events(events, label=" ".join(cmd[:2]))
    return SpawnResult(events=events, samples=samples, cold=cold, error=None,
                       _iters_requested=iters)


def run(
    cmd_prefix: list[str],
    *,
    model_path: Path,
    quant: str,
    ep: str,
    task: dict,
    iters: int,
    cold: bool = False,
    deadline_ms: int | None = None,
    backstop_s: float | None = None,
    sample: bool = False,
    shader_cache: Path | None = None,
) -> SpawnResult:
    """One chat-task spawn (the validation job). `task` is already resolved
    (documents inlined); we hand the exe everything — it does no path/template
    resolution beyond its own tokenizer.

    `sample` attaches the memory sampler — the job spawns only. `deadline_ms`
    soft-caps the in-process loop (the exe stops below K); `backstop_s` is the
    hard floor — if even one iteration outlives it we kill the tree and return
    a timed_out result with no events."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(task, fh)
        task_path = fh.name

    cmd = [
        *cmd_prefix, "run",
        "--model", str(model_path),
        "--quant", quant,
        "--ep", ep,
        "--task", task_path,
        "--iters", str(iters),
        *(["--deadline-ms", str(deadline_ms)] if deadline_ms else []),
        "--out", "-",
    ]
    try:
        return _execute(cmd, backstop_s=backstop_s, cold=cold, iters=iters, sample=sample,
                        shader_cache=shader_cache)
    finally:
        Path(task_path).unlink(missing_ok=True)


def sweep(
    cmd_prefix: list[str],
    *,
    model_path: Path,
    quant: str,
    ep: str,
    gate: dict | None = None,
    cold: bool = False,
    deadline_ms: int | None = None,
    backstop_s: float | None = None,
    shader_cache: Path | None = None,
) -> SpawnResult:
    """One sweep spawn. `gate` is the resolved health-check task the exe runs
    through the chat path before anything synthetic — sharing the sweep's model
    load; a missed expect makes the events unhealthy and the point arrays come
    back empty. `deadline_ms` soft-caps the prefill envelope (its first chunk
    always completes; the decode ladder completes regardless); `backstop_s`
    hard-kills a hang."""
    gate_path: str | None = None
    if gate is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(gate, fh)
            gate_path = fh.name
    cmd = [
        *cmd_prefix, "sweep",
        "--model", str(model_path),
        "--quant", quant,
        "--ep", ep,
        *(["--gate", gate_path] if gate_path else []),
        *(["--deadline-ms", str(deadline_ms)] if deadline_ms else []),
        "--out", "-",
    ]
    try:
        return _execute(cmd, backstop_s=backstop_s, cold=cold, shader_cache=shader_cache)
    finally:
        if gate_path:
            Path(gate_path).unlink(missing_ok=True)


def probe(
    cmd_prefix: list[str],
    *,
    ep: str,
    backstop_s: float | None = None,
    shader_cache: Path | None = None,
) -> SpawnResult:
    """One device-ceiling probe spawn — no model. Takes its own cache directory:
    it compiles a small pipeline set of its own, and sharing a directory with a
    cell would leave that cell's sweep partly warm."""
    cmd = [*cmd_prefix, "probe", "--ep", ep, "--out", "-"]
    return _execute(cmd, backstop_s=backstop_s, shader_cache=shader_cache)
