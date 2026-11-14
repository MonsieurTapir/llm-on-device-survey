"""Build the report: one self-contained HTML page.

The page answers one question — how much of a fleet of machines finishes a task
in time — and lets the reader change every part of it. A rail on the left holds the
model, the task (a prompt to read, a reply to write), two time limits (optimal,
tolerable), the backends in use and the fleet's composition. Two numbers answer it
— the share of the fleet within the limits, and the memory the task reserves — and
three panels show what is behind them:
coverage (each class of machine as a distribution over whole-task time, with
the measured machines on it), a side-by-side comparison of chosen machines, and a
searchable table of every device.

Nothing on the page is precomputed for a task. The builder embeds one data pack
(`_pack`): per (lane, model) the sweep's two measured cost curves, and around them
the lanes, machines and models they belong to. The page prices a task from the
curves (`assets/tasks.js`), spreads each class of machine over time
(`assets/fleet.js`) and draws the result (`assets/report.js`).

A lane is one device on one machine. Each lane has a device class (discrete GPU,
integrated GPU, CPU); each machine has a class of its own (`CLASSES`), which is what
a fleet is made of and the one thing hue means on the page.

Rendering is escaping-by-construction: jinja autoescape for HTML, `tojson` for the
data island (escapes `<` as \\u003c, so a submission string can never terminate a
script element), and the page writes every string from the pack through its own
escape. Submission data never lands in a template `|safe`; only repo-owned assets
(css/js) do.

    uv run --group analysis python -m analysis.site   # writes the default
    uv run --group analysis python -m analysis.site --out path.html

Nothing is fetched, at build time or when the page is viewed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from . import load_memory, load_probes, load_results, load_sweeps, load_thread_scaling

PKG = Path(__file__).parent
REPO = PKG.parent  # …/analysis → the repo root
PROJECT_URL = "https://github.com/MonsieurTapir/llm-on-device-survey"

INSTALL_BASH = (
    "curl -fsSL https://raw.githubusercontent.com/MonsieurTapir/"
    "llm-on-device-survey/main/run.sh | bash"
)
INSTALL_PS = (
    "irm https://raw.githubusercontent.com/MonsieurTapir/llm-on-device-survey/main/run.ps1 | iex"
)

# The classes a fleet is made of, in page order: key, label, the share of the fleet
# the page opens on, and the CPUs the class draws on. The shares are a starting
# point for the reader to move, not a measurement of any market. Classes that name
# the same CPUs are one population on the CPU backend: having a card says nothing
# about the processor beside it, so every PC's CPU counts for both PC classes.
# `assets/report.css` carries one colour per key.
CLASSES = (
    ("pc-gpu", "PC with GPU", 15, "pc"),
    ("pc", "PC without GPU", 70, "pc"),
    ("mac", "Apple Silicon Mac", 15, "mac"),
)

# The cores a typical machine has, and a weak one, per group of CPUs. A class is
# centred on its CPUs held to the typical count, by what their thread ladders say
# that costs, so a shelf of many-core workstations does not pass for the fleet. It
# reaches as far as any of them gets from there, as it ran or held to the weak
# count. A group not named here is taken as measured.
TIER_CORES = {"pc": {"typical": 6, "weak": 4}}

# The backends a lane can run on, in page order, as the reader knows them. A family
# the shelf holds and this does not name is shown under its own key.
FAMILIES = (("cpu", "CPU"), ("vulkan", "Vulkan"), ("mtl", "Metal"), ("cuda", "CUDA"))

# The model the page opens on, where the shelf holds it: the smallest of the set,
# so the widest spread of machines has something to show.
DEFAULT_MODEL = "gemma4-E2B"


# ---------------------------------------------------------------- lane identity
_TRADEMARK = re.compile(r"\s*\((?:R|TM|C)\)|®|™", re.IGNORECASE)
# A Vulkan/OpenGL driver appends its own identity in parens: "(RADV PHOENIX)".
_DRIVER_TAG = re.compile(
    r"\s*\((?:RADV|ANV|NVK|LLVM|MESA|SWIFTSHADER|LAVAPIPE|GFX)[^)]*\)", re.IGNORECASE
)
_VENDOR = re.compile(
    r"^(?:NVIDIA|AMD|Intel|Advanced Micro Devices,?(?: Inc\.?)?)\s+", re.IGNORECASE
)
# What a CPU brand string pads its model with — including the iGPU it mentions
# ("Ryzen 7 255 w/ Radeon 780M Graphics"), which belongs to the GPU lane, not here.
_CPU_TAIL = re.compile(r"\s*(?:\bw/\s.*|\d+-Core Processor|Processor|CPU @.*)$", re.IGNORECASE)
_GPU_TAIL = re.compile(r"\s+Graphics$", re.IGNORECASE)
_GEFORCE = re.compile(r"\bGeForce\s+", re.IGNORECASE)


def _chip(text: str) -> str:
    """A short silicon label from an exe- or OS-reported string:
    'AMD Ryzen 9 9950X 16-Core Processor' → 'Ryzen 9 9950X',
    'Intel(R) Core(TM) Ultra 5 125U' → 'Core Ultra 5 125U',
    'AMD Radeon 760M Graphics (RADV PHOENIX)' → 'Radeon 760M'."""
    # Whitespace first: a brand string padded to a fixed width ('… 8-Core Processor   ')
    # would otherwise keep its tail from matching.
    short = " ".join(_TRADEMARK.sub("", text or "").split())
    for pattern in (_DRIVER_TAG, _VENDOR, _CPU_TAIL, _GPU_TAIL, _GEFORCE):
        short = pattern.sub(" ", short)
    return " ".join(short.split()).strip(" ,") or (text or "?")


def _integrated(device: str, cpu: str) -> bool:
    """Whether a GPU lane's silicon lives in the CPU package. True when the runtime
    reports the host chip's own string (Apple silicon, RADV on a desktop APU),
    reports no model at all ('Intel(R) Graphics (MTL)'), or reports a name the CPU
    string already carries ('Radeon 760M' in 'Ryzen 5 PRO 230 w/ Radeon 760M
    Graphics'). A discrete card names a product the host chip never mentions."""
    chip = _chip(device)
    if chip == _chip(cpu):
        return True
    if not any(c.isdigit() for c in chip):
        return True
    return chip.lower() in (cpu or "").lower()


def _lane_chip(device: str, cpu: str, family: str) -> str:
    """The silicon a lane ran on. A discrete card and a CPU wear their own label; an
    integrated GPU wears the host chip's, marked 'iGPU' — unless the runtime
    reported that same string, which is Apple silicon naming the whole SoC."""
    if family == "cpu" or not _integrated(device, cpu):
        return _chip(device)
    if (device or "").strip() == (cpu or "").strip():
        return _chip(cpu)
    return f"{_chip(cpu)} iGPU"


def _width(family: str, batch, decode, cores) -> str:
    """The intra-op width a CPU lane ran, as a label suffix (' 6t', or ' 6t/18t'
    when the phases differ). CPU rates are per-thread-count results and the count
    is not implied by the chip — llama.cpp asks for every physical core on linux
    but only the top performance cluster on macOS, so an 18-core M-series lane
    runs 6 threads next to an 8-core linux lane running 8. Unlabelled, the two
    read as peers. And where the width undercounts the machine, the machine is
    said too (' 4t (8C)') — a bare '4t' reads as the size of the chip rather
    than the share of it the lane used. Empty for GPU lanes, where the pool only
    serves leftover ops."""
    if family != "cpu":
        return ""
    batch, decode, cores = (int(t) if t == t and t else None for t in (batch, decode, cores))
    if batch is None or decode is None:
        return ""
    width = f" {decode}t" if batch == decode else f" {decode}t/{batch}t"
    if cores and cores > max(batch, decode):
        width += f" ({cores}C)"
    return width


def _dev_class(device: str, cpu: str, family: str) -> str:
    if family == "cpu":
        return "CPU"
    return "integrated GPU" if _integrated(device, cpu) else "discrete GPU"


def _family(provider: str) -> str:
    return provider.split(":")[0]


def _with_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """Add `family`, `dev_class` and the display `lane` ('RTX 5080 · vulkan', a CPU
    lane carrying its thread width: 'Apple M5 Pro · cpu 6t'). A label two machines
    (or two lanes of one machine) would share is qualified until it is unique —
    rows are never silently pooled."""
    df = df.copy()
    df["family"] = [_family(p) for p in df.provider]
    df["dev_class"] = [
        _dev_class(d, c, f) for d, c, f in zip(df.device, df.cpu, df.family, strict=True)
    ]
    df["lane"] = [
        f"{_lane_chip(d, c, f)} · {f}{_width(f, b, k, n)}"
        for d, c, f, b, k, n in zip(
            df.device,
            df.cpu,
            df.family,
            df.threads_batch,
            df.threads_decode,
            df.cpu_cores,
            strict=True,
        )
    ]
    for qualifier in ("machine", "provider"):
        shared = df.groupby("lane")[qualifier].transform("nunique") > 1
        df.loc[shared, "lane"] = [
            f"{lane} ({q})" for lane, q in zip(df.lane[shared], df[qualifier][shared], strict=True)
        ]
    return df


# ------------------------------------------------------------- display names
# What a chip label still carries that a reader does not need: a generation prefix,
# a core count, a product line the model number already implies.
_GENERATION = re.compile(r"^\d+(?:st|nd|rd|th) Gen (?:Intel )?", re.IGNORECASE)
_CORES = re.compile(r"\s+\d+-Cores?$", re.IGNORECASE)
_SERIES = re.compile(r"\s+Series$", re.IGNORECASE)


def _brief(chip: str) -> str:
    """A chip label at its shortest: '13th Gen Intel Core i7-1370P' → 'Core i7-1370P',
    'Ryzen Threadripper PRO 3975WX 32-Cores' → 'Threadripper 3975WX',
    'Radeon Pro WX 3200 Series' → 'Radeon Pro WX 3200'."""
    short = _SERIES.sub("", _CORES.sub("", _GENERATION.sub("", chip)))
    return short.replace("Ryzen Threadripper PRO", "Threadripper")


def _lane_short(device: str, cpu: str, family: str) -> str:
    """A lane without its backend, for a reader who already sees which backends are
    in use: 'RTX 5080', 'Ryzen 7 255 iGPU', 'Apple M4 Pro GPU', 'Ryzen 7 255 CPU'."""
    chip = _brief(_lane_chip(device, cpu, family))
    if family == "cpu":
        return f"{chip} CPU"
    if _integrated(device, cpu) and not chip.endswith("iGPU"):
        return f"{chip} GPU"
    return chip


def _machine_class(os: str, cpu: str, dev_classes: set[str]) -> str:
    """Which class of the fleet a machine belongs to: an Apple Silicon Mac, a PC with
    a discrete GPU, or a PC without one."""
    if os == "macos" and _chip(cpu).startswith("Apple"):
        return "mac"
    return "pc-gpu" if "discrete GPU" in dev_classes else "pc"


# ----------------------------------------------------------------- the pack
def _plain(value, digits: int = 1):
    """A number the data island can carry: pandas hands out NaN for a stat that was
    never measured, and the island is strict JSON, so an absent number is None."""
    if value is None or value != value:  # NaN-safe
        return None
    return round(float(value), digits)


def _seconds(ms) -> float | None:
    """A millisecond stat in seconds, or None where it was never measured."""
    value = _plain(ms, 3)
    return None if value is None else round(value / 1e3, 2)


def _task_geometry(sweeps: pd.DataFrame) -> dict[tuple, dict]:
    """Per cell, the two measured cost curves the calculator prices tasks from,
    and how far each was measured.

    `curve` is the sweep's own cumulative time to first token over prompt depth:
    `[depth, ms]` pairs ascending from the `[0, 0]` origin, one per instrumented
    chunk boundary. It is the measurement itself, not a fit of it — the page
    interpolates between the points, so a prompt inside the measured range costs
    what the sweep clocked for that depth, nonlinearities and all. `ladder` is the
    decode side: `(fill, tok/s)` pairs ascending. `pre_max` / `kv_max` are the
    deepest prompt and the deepest fill measured: past them the page carries each
    curve's last measured rate onward and marks what it prints as an estimate."""
    out: dict[tuple, dict] = {}
    if sweeps.empty:
        return out
    key = ["machine", "backend", "provider", "model", "quant"]
    if "ttft_ms" in sweeps:
        pre = sweeps[(sweeps.kind == "prefill") & sweeps.ttft_ms.gt(0)]
        for cell, g in pre.groupby(key, sort=False):
            points = sorted([int(p.tokens), round(float(p.ttft_ms), 1)] for p in g.itertuples())
            out.setdefault(cell, {}).update(curve=[[0, 0], *points], pre_max=points[-1][0])
    if "tps_p50" in sweeps:
        dec = sweeps[(sweeps.kind == "decode") & sweeps.tps_p50.gt(0)]
        for cell, g in dec.groupby(key, sort=False):
            points = sorted([int(p.kv_fill), round(float(p.tps_p50), 2)] for p in g.itertuples())
            out.setdefault(cell, {}).update(ladder=points, kv_max=points[-1][0])
    return out


# --------------------------------------------------------------- first launch
# The sweep's warm pass walks a fixed set of dispatch widths before it measures
# anything: a full ubatch from an empty cache, a second over that history, a half
# ubatch, then a short ragged one — see `warmup` in backends/llamacpp/main.cpp,
# whose kWarmupRaggedTokens this mirrors. The full width is the fit's own, so only
# the ragged tail is a constant here.
WARMUP_RAGGED_TOKENS = 32


def _warm_pass_prefill_ms(row) -> float | None:
    """What the warm pass's width walk costs as plain inference, priced by the
    lane's own prefill cost function. None where there is no fit to price it with.

    This is the term that has to come out of the span before what is left can be
    called compilation. It is most of the span on most lanes — a 512-token ubatch
    walked two and a half times over is real prefill, and on a slow lane that is
    tens of seconds."""
    width = _plain(getattr(row, "fit_width", None), 0)
    intercept = _plain(getattr(row, "fit_intercept_ms", None), 6)
    slope = _plain(getattr(row, "fit_slope_ms_per_1k", None), 6)
    if not width or intercept is None or slope is None:
        return None
    total, depth = 0.0, 0
    for chunk in (width, width, max(1, width // 2), WARMUP_RAGGED_TOKENS):
        total += (intercept + slope * depth / 1e3) * (chunk / width)
        depth += chunk
    return total


def _compile_seconds(row) -> float | None:
    """What a first launch pays to build this model's pipelines on this lane — the
    warm pass with its own prefill netted out.

    The span itself is not that number. It walks dispatch widths by running tokens
    through them, so it is compilation *and* inference, and the inference part
    dominates: the sweep's walk alone came to 8.1 s of a 8.9 s span on a Core Ultra
    125U. `_warm_pass_prefill_ms` prices that part from the lane's own fit and it is
    subtracted here, which makes this an estimate — a difference of two numbers of
    similar size, so a small result is not a precise small result.

    Not a CPU lane: the GPU backend leaves a fixed, model-independent pipeline set
    behind at registry init, so a CPU lane's non-zero `shader_bytes` is that
    artifact and none of its warmup is compilation.

    Where the survey tool pinned the cache (`shader_cache == "redirected"`) an empty
    cache afterwards means the run genuinely compiled nothing — the one state in
    which that can be said. Where it could not (macOS, windows) the machine may have
    handed the driver pipelines it built earlier, so what is left after the
    subtraction is a floor of unknown tightness. It is reported all the same:
    refusing to would only replace an uncertain number with a wrong claim."""
    if row.family == "cpu":
        return None
    if row.shader_cache == "redirected" and not _plain(getattr(row, "shader_bytes", None), 0):
        return None
    span = _plain(getattr(row, "shader_warmup_ms", None), 3)
    prefill = _warm_pass_prefill_ms(row)
    if span is None or prefill is None:
        return None
    # The fit can over-predict its own walk; a compile cannot take negative time.
    return _seconds(max(0.0, span - prefill))


def _model_ids(df: pd.DataFrame) -> dict[tuple[str, str], str]:
    """The name each (model, quant) goes by on the page: the model's own, with the
    quant added only where the shelf holds the model at more than one."""
    quants = df.groupby("model").quant.nunique()
    return {
        (model, quant): model if quants[model] == 1 else f"{model} {quant}"
        for model, quant in df[["model", "quant"]].drop_duplicates().itertuples(index=False)
    }


def _memory(mem: pd.DataFrame) -> dict[tuple, dict]:
    """Per (lane, model, quant), what the allocator reserves as a function of context
    size: the weights and the compute buffers, which do not move, and the context
    buffer as a line through the shallowest and the deepest context the sweep sized
    (`kv_base` megabytes plus `kv_per_tok` for each token of depth). These are the
    allocator's own numbers, so the line is exact at both ends."""
    out: dict[tuple, dict] = {}
    if mem.empty:
        return out
    for cell, g in mem.groupby(["lane", "model", "quant"], sort=False):
        shallow, deep = g.loc[g.n_ctx.idxmin()], g.loc[g.n_ctx.idxmax()]
        if not deep.n_ctx > 0:
            continue
        slope = (
            (deep.kv_mb - shallow.kv_mb) / (deep.n_ctx - shallow.n_ctx)
            if deep.n_ctx > shallow.n_ctx
            else deep.kv_mb / deep.n_ctx
        )
        out[cell] = {
            "weights": _plain(deep.weights_mb),
            "compute": _plain(deep.compute_mb),
            "kv_base": _plain(max(0.0, deep.kv_mb - slope * deep.n_ctx)),
            "kv_per_tok": _plain(slope, 5),
        }
    return out


def _thread_ladders(scaling: pd.DataFrame) -> dict[tuple, dict]:
    """Per (lane, model, quant), what each phase's rate does as a CPU lane is given
    fewer threads: `{phase: {"threads", "points"}}`, `points` being (threads, tok/s)
    pairs ascending and `threads` the width the lane itself ran that phase at. The
    rates compare with each other and with nothing else — the ladder's work is not
    the sweep's — so the page reads ratios off them. A phase measured at one width
    has no ratio to give and is left out."""
    out: dict[tuple, dict] = {}
    if scaling.empty:
        return out
    for (lane, model, quant, phase), g in scaling.groupby(["lane", "model", "quant", "phase"]):
        g = g[(g.threads > 0) & (g.tps > 0)].sort_values("threads")
        own = _plain(g[f"threads_{'batch' if phase == 'prefill' else 'decode'}"].iloc[0], 0)
        if g.threads.nunique() < 2 or own is None:
            continue
        points = [[int(t), round(float(v), 2)] for t, v in zip(g.threads, g.tps, strict=True)]
        out.setdefault((lane, model, quant), {})[phase] = {"threads": int(own), "points": points}
    return out


def _records(
    df: pd.DataFrame, sweeps: pd.DataFrame, mem: pd.DataFrame, scaling: pd.DataFrame
) -> list[dict]:
    """Everything the page needs to price a task, per (lane, model): the measured
    prefill curve, the decode ladder, the once-per-launch costs, the allocator's
    reservations and, on a CPU lane, the thread ladder (`scaling`).

    Both cost curves are the sweep's measured points, never a fit of them: `curve`
    is cumulative time to first token over depth, `ladder` is (fill, tok/s) pairs,
    both ascending — inside the measured range a price is the measurement,
    interpolated, so a lane's own nonlinearity (thermal sag, bandwidth saturation)
    prices itself. Past either curve's end the page carries the last measured rate
    onward and marks the number as an estimate. `n_ctx_train` is the model's own
    trained context — the one task the page refuses outright, because past the
    *sweep's* depth there is still a curve to carry onward.

    A cell with neither curve is dropped: there is nothing to evaluate."""
    geo = _task_geometry(sweeps)
    ids = _model_ids(df)
    memory = _memory(mem)
    threads = _thread_ladders(scaling)
    records = []
    for r in df.itertuples():
        g = geo.get((r.machine, r.backend, r.provider, r.model, r.quant), {})
        curve = g.get("curve") or []
        ladder = g.get("ladder") or []
        if not curve and not ladder:
            continue
        if r.status == "ok":
            # The job measured a real generation rate at a real fill (its prompt
            # plus half its reply). It joins the ladder as a point: on the slow
            # lanes whose sweep reached one deep fill this is the difference
            # between interpolating and holding an 8 tok/s rate against a job
            # that measured 20.
            ttft, comp = (getattr(r, "ttft_ms_p50", None), getattr(r, "completion_ms_p50", None))
            ptps = getattr(r, "prefill_tps_p50", None)
            dtps = _plain(getattr(r, "decode_tps_p50", None))
            if all(v is not None and v == v for v in (ttft, comp, ptps, dtps)):
                reply = dtps * (comp - ttft) / 1e3
                fill = round(ptps * ttft / 1e3 + reply / 2)
                if fill > 0 and not any(abs(fill - f) < 256 for f, _ in ladder):
                    ladder = sorted([*ladder, [fill, round(float(dtps), 2)]])
                    g = {**g, "kv_max": max(g.get("kv_max") or 0, fill)}
        # What a launch pays before its first prompt: the weights read from a warm
        # page cache and the context allocated. Not the warm pass — the survey tool
        # warms by running prompts through, so that span is mostly inference, on CPU
        # lanes too, where nothing compiles at all. What it does buy on a GPU lane —
        # the driver's pipeline set — is `compile_s`, the once-per-machine cost.
        spans = [
            _plain(getattr(r, f"{span}_ms_p50", None), 1) for span in ("model_load", "context_init")
        ]
        paid = [s for s in spans if s is not None]
        n_ctx_train = _plain(getattr(r, "geo_n_ctx_train", None), 0)
        records.append(
            {
                "lane": r.lane,
                "model": ids[(r.model, r.quant)],
                "curve": curve,
                "pre_max": g.get("pre_max"),
                "ladder": ladder,
                "kv_max": g.get("kv_max"),
                "n_ctx_train": None if n_ctx_train is None else int(n_ctx_train),
                "load_s": round(sum(paid) / 1e3, 2) if paid else None,
                "compile_s": _compile_seconds(r),
                "mem": memory.get((r.lane, r.model, r.quant)),
                "scaling": threads.get((r.lane, r.model, r.quant)),
            }
        )
    return records


def _ceilings(probes: pd.DataFrame) -> dict[str, dict]:
    """Per lane, the two bare probes run with no model loaded: f16 GEMM in TFLOP/s
    and an on-device copy in GB/s. The best of each."""
    out: dict[str, dict] = {}
    if probes.empty:
        return out
    for lane, g in probes[probes.status == "ok"].groupby("lane"):
        out[lane] = {
            "gemm": _plain(g[g.kind == "gemm"].tflops.max(), 2),
            "d2d": _plain(g[g.kind == "d2d"].gbs.max()),
        }
    return out


def _pack(
    df: pd.DataFrame,
    sweeps: pd.DataFrame,
    mem: pd.DataFrame,
    probes: pd.DataFrame,
    scaling: pd.DataFrame,
) -> dict:
    """The one data island the page evaluates: the records and what they hang off.

    Only what has a record is carried — a lane the sweep measured nothing on has no
    place on a page that prices tasks, and neither does a machine with no such lane.
    `lanes` name each device in full (`id`, the label every frame agrees on) and in
    short; `machines` carry the name a reader would call the box, and its class;
    `models` are in size order. `classes` and `families` are the page's vocabulary
    for the two things a reader filters on."""
    records = _records(df, sweeps, mem, scaling)
    measured = {rec["lane"] for rec in records}
    ceilings = _ceilings(probes)
    lanes = []
    for r in df.drop_duplicates("lane").itertuples():
        if r.lane not in measured:
            continue
        threads = _plain(r.threads_decode, 0) if r.family == "cpu" else None
        lanes.append(
            {
                "id": r.lane,
                "short": _lane_short(r.device, r.cpu, r.family),
                "machine": r.machine,
                "cls": r.dev_class,
                "family": r.family,
                "threads": None if threads is None else int(threads),
                **ceilings.get(r.lane, {"gemm": None, "d2d": None}),
            }
        )
    machines = []
    for r in df.drop_duplicates("machine").itertuples():
        own = [lane for lane in lanes if lane["machine"] == r.machine]
        if not own:
            continue
        chip = _brief(_chip(r.cpu))
        cards = [lane["short"] for lane in own if lane["cls"] == "discrete GPU"]
        machines.append(
            {
                "id": r.machine,
                # A PC with a card is known by both: 'Ryzen 9 9950X + RTX 5080'. The
                # short name drops the card's product line where its number says it.
                "name": f"{chip} + {cards[0].replace('Radeon Pro ', '')}" if cards else chip,
                "full": f"{chip} + {cards[0]}" if cards else chip,
                "kind": _machine_class(r.os, r.cpu, {lane["cls"] for lane in own}),
                "cores": int(r.cpu_cores),
                "threads": int(r.cpu_threads),
                "ram": _plain(r.ram_gb, 0),
                "os": r.os,
            }
        )
    ids = _model_ids(df)
    seen = {rec["model"] for rec in records}
    models = []
    for r in df.drop_duplicates(["model", "quant"]).itertuples():
        params = df[(df.model == r.model) & (df.quant == r.quant)]
        n_params, n_bytes = params.geo_n_params.dropna(), params.geo_file_bytes.dropna()
        if ids[(r.model, r.quant)] in seen:
            models.append(
                {
                    "id": ids[(r.model, r.quant)],
                    "params_b": _plain(n_params.iloc[0] / 1e9) if len(n_params) else None,
                    "file_gb": _plain(n_bytes.iloc[0] / 1e9) if len(n_bytes) else None,
                }
            )
    models.sort(key=lambda m: (m["params_b"] is None, m["params_b"], m["id"]))
    held = {lane["family"] for lane in lanes}
    named = dict(FAMILIES)
    families = [{"key": k, "label": label} for k, label in FAMILIES if k in held]
    families += [{"key": k, "label": k} for k in sorted(held - named.keys())]
    return {
        "records": records,
        "lanes": lanes,
        "machines": machines,
        "models": models,
        "classes": [
            {
                "key": k,
                "label": label,
                "share": share,
                "cpus": cpus,
                "cores": TIER_CORES.get(cpus),
            }
            for k, label, share, cpus in CLASSES
        ],
        "families": families,
        "default_model": DEFAULT_MODEL if DEFAULT_MODEL in seen else models[0]["id"],
    }


def build(published: Path, out: Path) -> None:
    df = load_results(published)
    context: dict = {
        "project_url": PROJECT_URL,
        "install_bash": INSTALL_BASH,
        "install_ps": INSTALL_PS,
        "empty": True,
    }
    if not df.empty:
        df = _with_lanes(df)
        # An empty frame has no columns for `_with_lanes` to read, and a shelf whose
        # runs carried no allocator ladder, no probes or no thread ladder is exactly that.
        loaders = (load_sweeps, load_memory, load_probes, load_thread_scaling)
        frames = [load(published) for load in loaders]
        pack = _pack(df, *(f if f.empty else _with_lanes(f) for f in frames))
        # A shelf where no sweep measured a curve has nothing to price.
        if pack["records"]:
            context.update(empty=False, pack=pack)

    env = Environment(loader=PackageLoader("analysis"), autoescape=select_autoescape(default=True))
    for name in ("report.css", "tasks.js", "fleet.js", "report.js"):
        context[name.replace(".", "_")] = (PKG / "assets" / name).read_text()
    context["css"] = context.pop("report_css")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(env.get_template("report.html.j2").render(context))


def main() -> None:
    ap = argparse.ArgumentParser(description="build the static report")
    ap.add_argument("--published", type=Path, default=REPO / "submissions")
    ap.add_argument("--out", type=Path, default=REPO / "site" / "index.html")
    args = ap.parse_args()
    build(args.published, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
