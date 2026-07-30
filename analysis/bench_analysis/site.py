"""Build the report: one self-contained, data-first HTML page.

The page answers "how fast is on-device inference on real machines" one model at
a time. A control row scopes everything below it — model, device class, quant,
backend — and a control is rendered only where the shelf holds more than one
value for it. Under it, three charts (decode tok/s, prefill tok/s, warm init)
share one lane axis: lanes are grouped into bands by what kind of device they are
(discrete GPU / integrated GPU / CPU) and sorted by generation speed inside each
band. The two tok/s columns draw an interval, not a bar: both rates fall as the
context fills, so a lane spans the range the sweep measured across depth, with a
dot on the validation job's own number. Warm init is a bar — seconds to ready has
no depth. Each column's heading sits over the plot area, its numeric scale under
it, and on the two tok/s columns the reading-speed references cross the bands as
gridlines, named under the scale. The sections below read the same controls: what a
task would take on each lane (the same three columns, in seconds, computed in the
page from the measured cost functions — `assets/tasks.js`), what the first launch
costs before any rate applies (pipeline compilation, cold first touch), the cost
curves over depth, then the allocator's reservations against
context size and what the process actually held while the job ran, and — on the
shelves that measured it — what each phase of a CPU lane pays for intra-op thread
width. The page opens on one fold (the
contribute call to action) and closes on one more (the reference tables, as tab
panels) — the charts are the only thing that greets the reader unasked.

Hue means one thing on the page: which lane. A lane wears the same color in its
grid bar as in its curves, so the two halves read as one comparison — the first
eight lanes in a fixed order, gray past that. Device class is structure, not
color: it is the grid's row bands, and it is in the tooltip.

Rendering is escaping-by-construction: jinja autoescape for HTML, `tojson`
for the vega spec islands (escapes `<` as \\u003c, so submission strings can
never terminate a script element). Submission data never lands in a template
`|safe`; only repo-owned assets (css/js/vega libs) do.

    uv run --project analysis python -m bench_analysis.site   # writes the default
    uv run --project analysis python -m bench_analysis.site --out path.html

Vega/vega-lite/vega-embed are fetched once at build time — versions pinned to
the installed altair's constants — and cached under `third_party/vega/`.
Nothing is fetched when the page is *viewed*.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from . import load_memory, load_probes, load_results, load_sweeps, load_thread_scaling

PKG = Path(__file__).parent
REPO = PKG.parents[1]  # …/analysis/bench_analysis → the repo root
PROJECT_URL = "https://github.com/MonsieurTapir/llm-on-device-survey"

INSTALL_BASH = (
    "curl -fsSL https://raw.githubusercontent.com/MonsieurTapir/"
    "llm-on-device-survey/main/run.sh | bash"
)
INSTALL_PS = (
    "irm https://raw.githubusercontent.com/MonsieurTapir/llm-on-device-survey/main/run.ps1 | iex"
)

# Lane identity colors — validated (light + dark) against the six checks;
# assigned to lanes in fixed sorted order, never cycled past the overflow gray.
# Every chart draws its lanes from this list through `_lane_scale`, so one lane
# is one hue across the whole page.
LANE_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
LANE_COLORS_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]
LANE_OVERFLOW = "#898781"

# Device classes: the band order in the grid, fastest kind of silicon first. The
# class is carried by the banding and the tooltip, not by a hue of its own.
CLASSES = ("discrete GPU", "integrated GPU", "CPU")

VEGA_LIBS = (
    ("vega", alt.VEGA_VERSION),
    ("vega-lite", alt.VEGALITE_VERSION),
    ("vega-embed", alt.VEGAEMBED_VERSION),
)

# The spec dialect follows the vega-lite the page actually ships (see VEGA_LIBS).
VL_SCHEMA = f"https://vega.github.io/schema/vega-lite/v{alt.VEGALITE_VERSION.split('.')[0]}.json"

# The measured grid's columns. The two rate columns say their ink spans context
# depth, because a rate means nothing without the depth it belongs to — and the
# interval's ends are exactly that: the sweep's shallowest and deepest measured
# depth.
METRICS = (  # column key, title, subtitle
    ("decode", "generation (tok/s)", "across context depth"),
    ("prefill", "prompt reading (tok/s)", "across context depth"),
    ("init", "init, warm (s)", "lower is better"),
)

# The task grid's three column slots. Their meaning belongs to the selected task —
# TASKS below carries each task's own column titles, and the page re-titles the
# axes on selection — so these keys are routing, not semantics.
TASK_METRICS = ("ttft", "tps", "total")

# The columns whose number moves with how full the context is, and which therefore
# draw the sweep's range across depth instead of one bar. `init` is not one: it is
# paid before the first token, once. One place to edit when the columns change.
RANGE_METRICS = {"decode", "prefill"}

# Human-scale reference speeds for the two tok/s columns, as gridlines on the
# metric axis: a rate a reader already has a feel for, next to the measured bar.
# Rates convert at 1 token ≈ ¾ of a word, from a 250-word page and a 90,000-word
# book: silent reading ~200 wpm, a 60-word paragraph or a 250-word page per second, a
# book a minute. Three per column at most — past that the chart is gridlines. Every
# anchor is drawn, and each is named under the axis on its own line
# (`_anchor_axis`), so names never fight each other for room. The name is the whole
# label and reads as the human rate it stands for — the numeric axis just above
# already says where the line sits in tok/s. `init` has no key: seconds-to-ready
# has no reading-speed equivalent, so that column carries only the numeric scale.
READING_ANCHORS = {
    "decode": ((5, "silent reading"), (130, "1 paragraph/s")),
    "prefill": ((130, "1 paragraph/s"), (400, "1 page/s"), (1700, "1 book/min")),
}

# The task grid's generation column is the same quantity, so it carries the same two
# references.
TASK_ANCHORS = {"tps": ((5, "silent reading"), (130, "1 paragraph/s"))}

# The plot area of one grid column, in pixels.
GRID_WIDTH = 190

# The three tasks, in select order. Each carries its own column titles — a chat
# turn and a background job answer different questions, and the title is where
# that difference lives, not a footnote. Editing the token inputs edits the
# selected task; it stays that task. `mid_unit` says what the middle column is
# (a rate to watch, or seconds of writing); `fields` are the extra inputs the
# task uses (context-before is a chat idea). `summarize` mirrors the validation
# job (~1,550-token prompt, 256 tokens of reply, from empty), so its prediction
# is read against the job's own measurement as a dot.
TASKS = (
    {
        "key": "chat",
        "label": "chat turn",
        "depth": 4096,
        "prompt": 120,
        "out": 200,
        "measured": False,
        "mid_unit": "tps",
        "fields": ["depth"],
        "columns": (
            ("time to first word (s)", None),
            ("generation at this depth (tok/s)", None),
            ("whole turn (s)", "lower is better"),
        ),
        "note": "The earlier turns already sit in the KV cache, so only the new "
        "message is read before the first word.",
    },
    {
        "key": "summarize",
        "label": "read & summarize",
        "depth": 0,
        "prompt": 1550,
        "out": 256,
        "measured": True,
        "mid_unit": "tps",
        "fields": [],
        "columns": (
            ("time to first token (s)", None),
            ("generation over the reply (tok/s)", None),
            ("total (s)", "lower is better"),
        ),
        "note": "The whole prompt is read from empty before the first token. This is "
        "the validation job's own shape — the accuracy section below reads "
        "these predictions against its measurements.",
    },
    {
        "key": "extract",
        "label": "background extraction",
        "depth": 0,
        "prompt": 4096,
        "out": 400,
        "measured": False,
        "mid_unit": "s",
        "fields": [],
        "columns": (
            ("reading the input (s)", None),
            ("writing the output (s)", None),
            ("done after (s)", "lower is better"),
        ),
        "note": "Runs unattended, so only the end matters: done-after adds reading "
        "the weights and allocating the context to the two phases beside it.",
    },
)

# The controls, in row order: signal name, the row field it filters, its label.
# `model` is single-valued (one model at a time); the rest carry an "all" option.
CONTROLS = (
    ("f_model", "model", "model"),
    ("f_class", "dev_class", "device"),
    ("f_quant", "quant", "quant"),
    ("f_backend", "backend", "backend"),
)


def _vega_js(cache: Path) -> str:
    """The three vega libraries, concatenated; fetched once into `cache`."""
    parts = []
    cache.mkdir(parents=True, exist_ok=True)
    for name, version in VEGA_LIBS:
        f = cache / f"{name}@{version}.min.js"
        if not f.exists():
            url = f"https://cdn.jsdelivr.net/npm/{name}@{version}"
            with urllib.request.urlopen(url) as r:  # noqa: S310 — pinned https url
                f.write_bytes(r.read())
        parts.append(f.read_text())
    return "\n".join(parts)


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
    short = _TRADEMARK.sub("", text or "")
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


def _width(family: str, batch, decode) -> str:
    """The intra-op width a CPU lane ran, as a label suffix (' 6t', or ' 6t/18t'
    when the phases differ). CPU rates are per-thread-count results and the count
    is not implied by the chip — llama.cpp asks for every physical core on linux
    but only the top performance cluster on macOS, so an 18-core M-series lane
    runs 6 threads next to an 8-core linux lane running 8. Unlabelled, the two
    read as peers. Empty for GPU lanes, where the pool only serves leftover ops."""
    if family != "cpu":
        return ""
    batch, decode = (int(t) if t == t and t else None for t in (batch, decode))
    if batch is None or decode is None:
        return ""
    return f" {decode}t" if batch == decode else f" {decode}t/{batch}t"


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
        f"{_lane_chip(d, c, f)} · {f}{_width(f, b, k)}"
        for d, c, f, b, k in zip(
            df.device, df.cpu, df.family, df.threads_batch, df.threads_decode, strict=True
        )
    ]
    for qualifier in ("machine", "provider"):
        shared = df.groupby("lane")[qualifier].transform("nunique") > 1
        df.loc[shared, "lane"] = [
            f"{lane} ({q})" for lane, q in zip(df.lane[shared], df[qualifier][shared], strict=True)
        ]
    return df


def _lane_order(df: pd.DataFrame) -> list[str]:
    """Every lane, grouped by device class then named — the color domain, stable
    across models and filters (color follows the lane, never its rank)."""
    seen = df.drop_duplicates("lane")
    return [
        r.lane
        for r in sorted(seen.itertuples(), key=lambda r: (CLASSES.index(r.dev_class), r.lane))
    ]


# ------------------------------------------------------------------ chart specs
def _plain(value, digits: int = 1):
    """A number the islands can carry: pandas hands out NaN for a stat that was
    never measured, and the specs are strict JSON, so an absent number is None."""
    if value is None or value != value:  # NaN-safe
        return None
    return round(float(value), digits)


def _seconds(ms) -> float | None:
    """A millisecond stat in seconds, or None where it was never measured."""
    value = _plain(ms, 3)
    return None if value is None else round(value / 1e3, 2)


def _depth_ranges(sweeps: pd.DataFrame) -> dict[tuple[tuple, str], dict]:
    """What the sweep measured across context depth, per cell and range metric.

    Keyed on `((machine, backend, provider, model, quant), metric)` — the cell as
    the results frame rows it, not the display lane, so two lanes of one machine
    stay two keys. Decode endpoints are the tok/s at the shallowest and the deepest
    measured `kv_fill`; prefill endpoints are the cumulative average rate
    (`tokens / ttft`) at the shallowest and the deepest prompt depth. Published
    sweep points arrive in whatever order the harness wrote them (`[8192, 2048, 0]`
    is real data), so both phases sort before taking their ends."""
    out: dict[tuple[tuple, str], dict] = {}
    if sweeps.empty:
        return out
    frames: dict[str, pd.DataFrame] = {}
    if "tps_p50" in sweeps:
        dec = sweeps[(sweeps.kind == "decode") & sweeps.tps_p50.gt(0)].copy()
        dec["depth"], dec["rate"] = dec.kv_fill, dec.tps_p50
        frames["decode"] = dec
    if "ttft_ms" in sweeps:
        pre = sweeps[(sweeps.kind == "prefill") & sweeps.ttft_ms.gt(0)].copy()
        pre["depth"] = pre.tokens
        pre["rate"] = pre.tokens / (pre.ttft_ms / 1e3)
        frames["prefill"] = pre
    key = ["machine", "backend", "provider", "model", "quant"]
    for metric, frame in frames.items():
        for cell, g in frame.sort_values("depth").groupby(key, sort=False):
            shallow, deep = g.iloc[0], g.iloc[-1]
            out[(cell, metric)] = {
                "d_shallow": int(shallow.depth),
                "d_deep": int(deep.depth),
                "v_shallow": round(float(shallow.rate), 2),
                "v_deep": round(float(deep.rate), 2),
                "n_depths": len(g),
            }
    return out


def _rate(v: float) -> str:
    """A tok/s figure as the grid prints it: whole numbers once double digits, one
    decimal below that — a slow lane's 8.9 and 9.6 are different answers."""
    return f"{v:,.0f}" if v >= 10 else f"{round(v, 1):g}"


def _range_fields(span: dict | None) -> dict:
    """One range metric's interval for one cell: the sweep's endpoints over depth,
    and nothing else's.

    `range_label` prints the two ends at the interval's far edge — a one-point
    sweep collapses to a point and prints one number — and `label_x` is that edge,
    which is also where a status note goes. `sweep_str` is the hover's version,
    with the depths the ends were measured at."""
    fields = {
        "v_shallow": None,
        "v_deep": None,
        "d_shallow": None,
        "d_deep": None,
        "n_depths": None,
        "lo": None,
        "hi": None,
        "label_x": None,
        "range_label": None,
        "sweep_str": None,
    }
    if span:
        fields.update(span)
        lo, hi = sorted((span["v_shallow"], span["v_deep"]))
        fields["lo"], fields["hi"] = lo, hi
        fields["label_x"] = hi
        fields["range_label"] = _rate(hi) if lo == hi else f"{_rate(lo)}–{_rate(hi)}"
        shallow = f"{span['v_shallow']:g} at {_depth_str(span['d_shallow'])} tokens"
        deep = f"{span['v_deep']:g} at {_depth_str(span['d_deep'])}"
        fields["sweep_str"] = (
            shallow if span["d_shallow"] == span["d_deep"] else f"{shallow} → {deep}"
        )
    return fields


def _speed_rank(df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Each lane's generation-speed position within one model, keyed
    `(model, lane)`. Every chart with a lane axis sorts on it, so the reader meets
    the lanes in the same order wherever they appear. A lane that scored no job has
    no position; the charts fall back to the end of the order."""
    rank: dict[tuple[str, str], int] = {}
    for model, g in df[df.status == "ok"].groupby("model"):
        ordered = g.sort_values("decode_tps_p50", ascending=False)
        rank.update({(model, lane): i for i, lane in enumerate(ordered.lane)})
    return rank


def _grid_rows(df: pd.DataFrame, ranges: dict[tuple[tuple, str], dict]) -> list[dict]:
    """Long-format rows for the grid: one per (lane, model, metric). Every cell of
    every measured (lane, model) is emitted — a cell with no number carries the
    reason instead — so the three metric columns keep identical rows, and no lane
    silently drops out of a band.

    The two range columns carry only what the sweep measured (`_range_fields`); the
    validation job's own numbers stay out of this grid — the accuracy section is
    where they are read against the calculator. `value` is the init column's
    number; the range columns' ink is `lo`/`hi`.

    A cell that measured nothing carries the reason — and carries it *once*, on the
    first column, because the reason belongs to the whole (lane, model) and
    printing it in all three columns is the same sentence three times. The other
    columns keep the row and stay blank. Where the sweep left an interval but the
    job produced no number, the reason names the job ("job too slow") — the row
    visibly measured something, and the note must not read as if it had not; that
    column's range label yields to the note, and the range stays on hover. The
    inverse gap is also said out loud: an ok cell whose sweep left no points in a
    range column notes "no sweep data" there instead of a silent blank.

    `rank` is the lane's generation-speed position within its model; the y axis
    sorts on it, so all three columns order their bands the same way."""
    rank = _speed_rank(df)
    rows = []
    for r in df.itertuples():
        # Everything a row carries is drawn or filtered on: the lane is the axis, the
        # class is the band, and the other three are what the control row scopes by.
        base = {
            "lane": r.lane,
            "dev_class": r.dev_class,
            "backend": r.backend,
            "model": r.model,
            "quant": r.quant,
            "rank": rank.get((r.model, r.lane), len(rank)),
        }
        cell = (r.machine, r.backend, r.provider, r.model, r.quant)
        ok = r.status == "ok"
        init = (r.model_load_ms_p50 + r.context_init_ms_p50) / 1e3 if ok else None
        note = None if ok else str(r.status).replace("_", " ")
        if note and any((cell, m) in ranges for m in RANGE_METRICS):
            note = f"job {note}"  # the sweep left an interval; the job is what did not
        told = False  # the status note, once, in the first column
        for metric in ("decode", "prefill", "init"):
            value = None
            if metric == "init" and init is not None and init == init:  # NaN-safe
                value = round(float(init), 2)
            here = note if note and not told else None
            told = told or here is not None
            row = {**base, "metric": metric, "value": value, "note": here}
            if metric in RANGE_METRICS:
                span = ranges.get((cell, metric))
                row.update(_range_fields(span))
                if here:
                    row["range_label"] = None  # the note sits where the label would
                elif ok and span is None:
                    row["note"] = "no sweep data"
            rows.append(row)
    return rows


def _lane_scale(lanes: list[str], colors: list[str]) -> dict:
    """The one lane→color mapping the page has: `lanes` in their fixed order against
    `colors`, so a lane's bar and its curve are the same hue. Past the eighth lane
    the palette is out and the rest share `LANE_OVERFLOW` gray — those lanes are
    told apart by the axis label in the grid, by the legend in the curves."""
    return {
        "domain": lanes,
        "range": [colors[i] if i < len(colors) else LANE_OVERFLOW for i in range(len(lanes))],
    }


def _controls(rows: list[dict], default_model: str) -> list[dict]:
    """The control row: one entry per signal that has something to choose. `model`
    always carries a signal (the page shows one model at a time) but is only
    rendered as a select when the shelf holds more than one."""
    controls = []
    for signal, field, label in CONTROLS:
        values = sorted({r[field] for r in rows if r[field] is not None})
        if signal == "f_model":
            controls.append(
                {
                    "signal": signal,
                    "field": field,
                    "label": label,
                    "options": values,
                    "value": default_model,
                    "render": len(values) > 1,
                }
            )
        elif len(values) > 1:
            controls.append(
                {
                    "signal": signal,
                    "field": field,
                    "label": label,
                    "options": ["all", *values],
                    "value": "all",
                    "render": True,
                }
            )
    return controls


def _filters(controls: list[dict]) -> list[dict]:
    out = []
    for c in controls:
        field, signal = c["field"], c["signal"]
        out.append(
            {
                "filter": f"datum.{field} === {signal}"
                if signal == "f_model"
                else f"{signal} === 'all' || datum.{field} === {signal}"
            }
        )
    return out


def _params(controls: list[dict]) -> list[dict]:
    return [{"name": c["signal"], "value": c["value"]} for c in controls]


def _depth_str(tokens: float) -> str:
    """A token depth as a tooltip can say it: plain under a thousand, '7.2k' above —
    the same shorthand the axis headings use."""
    return f"{round(tokens):,}" if tokens < 1000 else f"{round(tokens / 100) / 10:g}k"


def _heading_axis(title: str, subtitle: str | None) -> dict:
    """A grid column's heading: a top x axis that carries only its title.

    The heading rides an axis rather than the column, because the axis group is
    exactly the plot area — `orient: top` with `titleAnchor: start` puts the text
    over the first bar's left edge, where a concat title would span the lane-label
    gutter too. Everything else the column's scale has to say sits below the plot
    (`_value_axis`, `_anchor_axis`), so this axis draws no labels, ticks or line."""
    return {
        "orient": "top",
        "titleAnchor": "start",
        "title": [title, subtitle] if subtitle else title,
        "titleFontSize": 12,
        "titleLineHeight": 13,
        "labels": False,
        "ticks": False,
        "domain": False,
        "grid": False,
    }


def _value_axis(fmt: str | None = None) -> dict:
    """A grid column's scale: an ordinary numeric axis under the plot — line,
    plain tick labels, no grid. This is what a mark is read against; the exact
    number a mark stands for is its tooltip's job, not a printed label's."""
    axis: dict = {
        "orient": "bottom",
        "title": None,
        "grid": False,
        "domain": True,
        "domainColor": "currentColor",
        "domainOpacity": 0.3,
        "ticks": False,
        "labelFontSize": 9,
        "labelOpacity": 0.7,
        "labelOverlap": "greedy",
    }
    return {**axis, "format": fmt} if fmt else axis


def _anchor_axis(anchors: tuple) -> dict:
    """A column's reading-speed references: dashed gridlines crossing every band,
    each named below the numeric axis. The names stack — anchor *i* renders on
    line *i* of a multi-line tick label — so two anchors close together on the
    scale never fight for the same stretch of text, and every reference that is
    drawn is named.

    The anchors are gridlines, not a rule layer, because a gridline that no longer
    fits the domain simply is not drawn: filtering the page rescales the references
    with the marks (a rule *mark* would stretch the domain to reach itself), and a
    culled gridline takes its name with it."""
    label = "".join(
        f"datum.value === {value} ? "
        + "["
        + ", ".join(f"'{s}'" for s in [""] * i + [name])
        + "] : "
        for i, (value, name) in enumerate(anchors)
    )
    return {
        "orient": "bottom",
        "title": None,
        "grid": True,
        "gridDash": [1, 3],
        "gridOpacity": 0.4,
        "gridColor": "currentColor",
        "values": [v for v, _ in anchors],
        "labelExpr": f"{label}''",
        "labelFontSize": 9,
        "labelOpacity": 0.7,
        "labelLineHeight": 11,
        "labelPadding": 14,
        "labelOverlap": False,
        "ticks": False,
        "domain": False,
    }


def _axis_carriers(y: dict, heading: dict, anchors: tuple | None) -> list[dict]:
    """Invisible layers whose only job is to carry one axis each. A vega-lite layer
    defines at most one x axis, and a grid column needs up to three on one shared
    scale (heading on top, numeric scale below, anchor references below that) — so
    the extra axes ride zero-opacity point marks. The carrier's y is the data
    layers' own y (same sort), or the merged y scale would carry conflicting sorts
    — and a carrier with *no* y breaks the step-sized facet outright (vega-lite
    6.4.1 emits a duplicate `height` signal). It must not touch the y *axis*
    though: an explicit `axis: None` wins the layer merge and strips the lane
    labels off the column that has them. Layers may not resolve the x axis
    `independent` either — same faceted-layer bug — but distinct per-layer axis
    objects on one shared scale coexist without it."""
    carrier = {"mark": {"type": "point", "opacity": 0}}
    layers = [
        {
            **carrier,
            "encoding": {
                "y": y,
                "x": {"field": "value", "type": "quantitative", "axis": heading},
            },
        }
    ]
    if anchors:
        layers.append(
            {
                **carrier,
                "encoding": {
                    "y": y,
                    "x": {"field": "value", "type": "quantitative", "axis": _anchor_axis(anchors)},
                },
            }
        )
    return layers


def _grid_tooltip(metric: str) -> list[dict]:
    """One cell's numbers on hover — and nothing else. Which lane, machine, model and
    device class a row belongs to is already on the screen (the lane is the y-axis
    label, the rest are the control row's selections), so a tooltip carries only what
    the marks cannot: for a range metric, each end of the sweep *with the depth it
    was measured at* — the range is unreadable without them."""
    if metric not in RANGE_METRICS:
        return [{"field": "value", "title": "value"}]
    return [{"field": "sweep_str", "title": "tok/s over depth"}]


def _grid_spec(rows: list[dict], controls: list[dict], lanes: list[str]) -> dict:
    """One hconcat: per metric, a chart banded (by device class) down a shared lane
    axis. A lane's mark wears its lane's color, the one the curves use, so the reader
    carries a lane between the two sections; the band it sits in is its device class.
    The leftmost column carries the band and lane labels for all three; every row
    carries one number, which is also the contrast relief the palette needs.

    The two `RANGE_METRICS` columns draw an interval — the range the sweep measured
    from its shallowest to its deepest context depth — and print its two ends at
    the interval's edge. `init` keeps a plain bar with its value at the end. The
    validation job appears nowhere here: what it measured against what the
    calculator predicts is the accuracy section's story. A cell that measured
    nothing carries its italic status note in one column (`_grid_rows`); a cell
    whose sweep produced points but whose job did not draws the interval and the
    note says it is the job that failed, so nothing in the row reads as a
    measurement the job never made.

    A column's heading sits over the plot area (`_heading_axis`), its scale under it
    (`_value_axis`), and the reading-speed references are gridlines named under that
    scale (`_anchor_axis`) — all on one shared x scale, the extra axes carried by
    invisible layers (`_axis_carriers`)."""
    y = {
        "field": "lane",
        "type": "nominal",
        "title": None,
        "sort": {"field": "rank", "op": "min", "order": "ascending"},
    }
    columns = []
    for i, (metric, title, subtitle) in enumerate(METRICS):
        labelled = i == 0  # band + lane labels once, on the left
        tooltip = _grid_tooltip(metric)
        y_axis = {**y, "axis": {"labelLimit": 200, "labelFontSize": 11} if labelled else None}
        heading = _heading_axis(title, subtitle)
        color = {
            "field": "lane",
            "type": "nominal",
            "legend": None,
            "scale": _lane_scale(lanes, LANE_COLORS),
        }
        note = {
            "type": "text",
            "align": "left",
            "fontSize": 10,
            "fontStyle": "italic",
            "opacity": 0.65,
        }
        text = {"type": "text", "align": "left", "fontSize": 10}
        if metric in RANGE_METRICS:
            layer = [
                # A rule carries no zero the way a bar does; ask for it, or a lane
                # whose whole range sits high would read as if it started there.
                {
                    "transform": [{"filter": "datum.hi !== null"}],
                    "mark": {
                        "type": "rule",
                        "strokeWidth": 7,
                        "strokeCap": "round",
                        "opacity": 0.4,
                    },
                    "encoding": {
                        "y": y_axis,
                        "x": {
                            "field": "lo",
                            "type": "quantitative",
                            "scale": {"zero": True},
                            "axis": _value_axis(),
                        },
                        "x2": {"field": "hi"},
                        "color": color,
                        "tooltip": tooltip,
                    },
                },
                # The interval's two ends, printed at its edge — cleared of the
                # rule's ~4px round cap by dx.
                {
                    "transform": [{"filter": "datum.range_label !== null"}],
                    "mark": {**text, "dx": 9},
                    "encoding": {
                        "y": y,
                        "x": {"field": "label_x", "type": "quantitative", "axis": None},
                        "text": {"field": "range_label"},
                    },
                },
                {
                    "transform": [{"calculate": "datum.label_x * 1.16", "as": "headroom"}],
                    "mark": {"type": "point", "opacity": 0},
                    "encoding": {
                        "y": y,
                        "x": {"field": "headroom", "type": "quantitative", "axis": None},
                    },
                },
                {
                    "transform": [
                        {"filter": "datum.note !== null"},
                        {"calculate": "datum.label_x === null ? 0 : datum.label_x", "as": "note_x"},
                    ],
                    "mark": {**note, "dx": 9},
                    "encoding": {
                        "y": y,
                        "x": {"field": "note_x", "type": "quantitative", "axis": None},
                        "text": {"field": "note"},
                    },
                },
            ]
        else:
            layer = [
                {
                    "mark": {"type": "bar", "height": 9, "cornerRadiusEnd": 3},
                    "encoding": {
                        "y": y_axis,
                        "x": {"field": "value", "type": "quantitative", "axis": _value_axis()},
                        "color": color,
                        "tooltip": tooltip,
                    },
                },
                {
                    "transform": [{"filter": "datum.value !== null"}],
                    "mark": {**text, "dx": 4},
                    "encoding": {
                        "y": y,
                        "x": {"field": "value", "type": "quantitative", "axis": None},
                        "text": {"field": "value", "format": ".1f"},
                    },
                },
                {
                    "transform": [{"calculate": "datum.value * 1.16", "as": "headroom"}],
                    "mark": {"type": "point", "opacity": 0},
                    "encoding": {
                        "y": y,
                        "x": {"field": "headroom", "type": "quantitative", "axis": None},
                    },
                },
                {
                    "transform": [{"filter": "datum.note !== null"}],
                    "mark": {**note, "dx": 4},
                    "encoding": {
                        "y": y,
                        "x": {"datum": 0, "type": "quantitative", "axis": None},
                        "text": {"field": "note"},
                    },
                },
            ]
        layer += _axis_carriers(y, heading, READING_ANCHORS.get(metric))
        columns.append(
            {
                "transform": [{"filter": f"datum.metric === '{metric}'"}, *_filters(controls)],
                "facet": {
                    "row": {
                        "field": "dev_class",
                        "title": None,
                        "sort": list(CLASSES),
                        "header": {
                            "labels": labelled,
                            "labelAngle": 0,
                            "labelAlign": "left",
                            "labelFontWeight": "bold",
                            "labelPadding": 2,
                            "labelLimit": 120,
                        },
                    }
                },
                "spacing": 6,
                "resolve": {"scale": {"y": "independent"}},  # a band shows only its lanes
                "spec": {"width": GRID_WIDTH, "height": {"step": 21}, "layer": layer},
            }
        )
    return {
        "$schema": VL_SCHEMA,
        "data": {"values": rows},
        "params": _params(controls),
        "hconcat": columns,
        "spacing": 26,
        "config": {"view": {"stroke": None}, "axis": {"grid": False}},
    }


# ------------------------------------------------------------------------ tasks
# A token is about ¾ of a word, which is the only unit a reader has a feel for: 4,096
# tokens is a document, not a number. The page states the ratio once, in the controls
# row, and `assets/tasks.js` carries the same two functions for the live hint beside
# each input — keep the two sides in step.
WORDS_PER_TOKEN = 0.75


def _words(tokens: int) -> str:
    """A token count in words, rounded to a step that reads as the estimate it is:
    25 words under 500, 50 under 2,000, 250 above."""
    words = tokens * WORDS_PER_TOKEN
    step = 25 if words < 500 else 50 if words < 2000 else 250
    return f"{round(words / step) * step:,}"


def _workload(p: dict) -> str:
    """One task's default configuration as the work it stands for, with the token
    counts the arithmetic actually uses beside it. The task's own note then says how
    that work is charged — together they make a column of seconds mean something."""
    parts = []
    if p["depth"]:
        parts.append(
            f"~{_words(p['depth'])} words of conversation already in the "
            f"context ({p['depth']:,} tokens)"
        )
    if p["prompt"]:
        parts.append(f"a ~{_words(p['prompt'])}-word prompt ({p['prompt']:,} tokens)")
    if p["out"]:
        parts.append(f"a ~{_words(p['out'])}-word reply ({p['out']:,} tokens)")
    if not parts:
        return ""
    head = ", ".join(parts[:-1])
    sentence = f"{head} and {parts[-1]}" if head else parts[-1]
    return f"{sentence[0].upper()}{sentence[1:]}."


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


def _task_pack(df: pd.DataFrame, sweeps: pd.DataFrame) -> dict:
    """Everything the page needs to price a task, per (lane, model): the measured
    prefill curve, the decode ladder, the once-per-launch costs, and the validation
    job's own numbers to check the arithmetic against.

    This is a data pack, not a set of chart rows — the page evaluates it, so a preset
    and a hand-typed configuration go through exactly one code path (`assets/tasks.js`)
    and no reload. Both cost curves are the sweep's measured points, never a fit of
    them: `curve` is cumulative time to first token over depth, `ladder` is (fill,
    tok/s) pairs, both ascending — inside the measured range a prediction is the
    measurement, interpolated, so a lane's own nonlinearity (thermal sag, bandwidth
    saturation) prices itself. Past either curve's end the page carries the last
    measured rate onward and marks the number as an estimate. `n_ctx_train` is the
    model's own trained context — the one configuration the page refuses outright,
    because past the *sweep's* depth there is still a curve to carry onward.

    A cell with neither curve is dropped: there is nothing to evaluate, and the
    Results grid above already carries the reason it measured nothing."""
    geo = _task_geometry(sweeps)
    rank = _speed_rank(df)
    records = []
    for r in df.itertuples():
        cell = (r.machine, r.backend, r.provider, r.model, r.quant)
        g = geo.get(cell, {})
        curve = g.get("curve") or []
        ladder = g.get("ladder") or []
        if not curve and not ladder:
            continue
        measured = None
        if r.status == "ok":
            measured = {
                "ttft": _seconds(getattr(r, "ttft_ms_p50", None)),
                "tps": _plain(getattr(r, "decode_tps_p50", None)),
                "total": _seconds(getattr(r, "completion_ms_p50", None)),
            }
            # The job measured a real generation rate at a real fill (its prompt
            # plus half its reply). It joins the ladder as a point: on the slow
            # lanes whose sweep reached one deep fill this is the difference
            # between interpolating and holding an 8 tok/s rate against a job
            # that measured 20. It is also why the accuracy section never grades
            # generation alone: on such lanes the generation point *is* the job.
            ttft, comp = (getattr(r, "ttft_ms_p50", None), getattr(r, "completion_ms_p50", None))
            ptps, dtps = (getattr(r, "prefill_tps_p50", None), measured["tps"])
            if all(v is not None and v == v for v in (ttft, comp, ptps, dtps)):
                reply = dtps * (comp - ttft) / 1e3
                fill = round(ptps * ttft / 1e3 + reply / 2)
                if fill > 0 and not any(abs(fill - f) < 256 for f, _ in ladder):
                    ladder = sorted([*ladder, [fill, round(float(dtps), 2)]])
                    g = {**g, "kv_max": max(g.get("kv_max") or 0, fill)}
        # What a batch pays before its first document: the weights read from a warm
        # page cache and the context allocated. Not the warm pass — the harness warms
        # by running the task's own prompts through (plus a synthetic width walk), so
        # that span is ~1.85x the job's own TTFT on every lane measured, CPU lanes
        # included, where nothing compiles at all. Charging it here would bill a
        # reader for prefilling the prompt roughly three times. What it does buy on a
        # GPU lane — the driver's pipeline set — the first-launch chart reports on
        # its own, as the once-per-machine cost it is.
        spans = [
            _plain(getattr(r, f"{span}_ms_p50", None), 1) for span in ("model_load", "context_init")
        ]
        paid = [s for s in spans if s is not None]
        n_ctx_train = _plain(getattr(r, "geo_n_ctx_train", None), 0)
        records.append(
            {
                "lane": r.lane,
                "dev_class": r.dev_class,
                "machine": r.machine,
                "model": r.model,
                "quant": r.quant,
                "backend": r.backend,
                "rank": rank.get((r.model, r.lane), len(rank)),
                "curve": curve,
                "pre_max": g.get("pre_max"),
                "ladder": ladder,
                "kv_max": g.get("kv_max"),
                # The model's own trained context: the one limit the page refuses at,
                # because it is a fact about the model rather than about our sweep budget.
                "n_ctx_train": None if n_ctx_train is None else int(n_ctx_train),
                "load_s": round(sum(paid) / 1e3, 2) if paid else None,
                "cold_s": _seconds(getattr(r, "cold_start_ms_p50", None)),
                "first_launch_s": _compile_seconds(r),
                "measured": measured,
            }
        )
    # Each task's note opens with the work its defaults stand for, in words as well
    # as tokens; the controls row carries the live word count as the reader types.
    tasks = [{**t, "note": f"{_workload(t)} {t['note']}"} for t in TASKS]
    return {"records": records, "tasks": tasks, "metrics": list(TASK_METRICS)}


def _task_tooltip() -> list[dict]:
    """One predicted cell on hover: its number, and the one-line reason it is marked
    as an estimate or has no number at all. The configuration is already in the
    controls row and the lane is the axis label, so neither is repeated. The reason
    is a label, never a sentence — what a half-lit bar means is explained once in
    the section's copy, where a reader can actually read it, and there is exactly
    one reason row because a column either has a number resting on something or has
    no number at all."""
    return [
        {"field": "value_label", "title": "value"},
        {"field": "why", "title": "note"},
    ]


def _task_spec(controls: list[dict], lanes: list[str]) -> dict:
    """The task grid: the same lane axis and bands as the measured grid, three
    columns of seconds-for-a-task, and no data of its own — `assets/tasks.js` fills
    the named `tasks` set from the pack whenever a control moves.

    Layout idioms come from the measured grid (heading over the plot area, numeric
    scale and anchor references below it, band and lane labels on the leftmost
    column only, value printed at the bar's end, italic note at the axis where
    there is no number). What is different is what this chart is: every bar is
    computed, not measured. So a bar whose prediction rests on something thin — a
    single measured fill, a loose prefill fit, a depth past the one the sweep
    reached — is drawn at half ink and its label wears a `~`. A row draws no bar
    only where there is nothing to evaluate: no cost function on the lane, or a
    task past the model's own trained context. Hue is still the lane's, which is
    why the estimate marking is opacity: a value condition carries no scale, so it
    adds no second legend to read. How these predictions compare to a measurement
    is the accuracy section's story (`_accuracy_spec`), not a mark here.

    Column titles here are the first task's; the page swaps them (and the middle
    column's reading-speed gridlines, meaningless when that column is seconds) when
    the reader picks another task, by re-embedding this spec with the picked task's
    `columns` — the titles live in TASKS, not here."""
    y = {
        "field": "lane",
        "type": "nominal",
        "title": None,
        "sort": {"field": "rank", "op": "min", "order": "ascending"},
    }
    tooltip = _task_tooltip()
    columns = []
    for i, (metric, (title, subtitle)) in enumerate(
        zip(TASK_METRICS, TASKS[0]["columns"], strict=True)
    ):
        labelled = i == 0
        y_axis = {**y, "axis": {"labelLimit": 200, "labelFontSize": 11} if labelled else None}
        # Generation carries reading-speed anchors (see TASK_ANCHORS);
        # seconds-for-a-task has no reading-speed equivalent. The trimmed tick format
        # is for the moment before the page's first insert, when the domain is empty
        # and vega would otherwise derive six decimals of precision for a lone zero.
        color = {
            "field": "lane",
            "type": "nominal",
            "legend": None,
            "scale": _lane_scale(lanes, LANE_COLORS),
        }
        columns.append(
            {
                "transform": [{"filter": f"datum.metric === '{metric}'"}, *_filters(controls)],
                "facet": {
                    "row": {
                        "field": "dev_class",
                        "title": None,
                        "sort": list(CLASSES),
                        "header": {
                            "labels": labelled,
                            "labelAngle": 0,
                            "labelAlign": "left",
                            "labelFontWeight": "bold",
                            "labelPadding": 2,
                            "labelLimit": 120,
                        },
                    }
                },
                "spacing": 6,
                "resolve": {"scale": {"y": "independent"}},
                "spec": {
                    "width": GRID_WIDTH,
                    "height": {"step": 21},
                    "layer": [
                        {
                            "mark": {"type": "bar", "height": 9, "cornerRadiusEnd": 3},
                            "encoding": {
                                "y": y_axis,
                                "x": {
                                    "field": "value",
                                    "type": "quantitative",
                                    "scale": {"zero": True},
                                    "axis": _value_axis("~r"),
                                },
                                "color": color,
                                "opacity": {
                                    "condition": {"test": "datum.est", "value": 0.45},
                                    "value": 1,
                                },
                                "tooltip": tooltip,
                            },
                        },
                        # The value at the bar's end: a string the page's arithmetic
                        # writes ("2 min 14 s"), with its ~ when it is an estimate.
                        {
                            "transform": [{"filter": "datum.label !== null"}],
                            "mark": {"type": "text", "align": "left", "dx": 4, "fontSize": 10},
                            "encoding": {
                                "y": y,
                                "x": {"field": "value", "type": "quantitative", "axis": None},
                                "text": {"field": "label"},
                            },
                        },
                        {
                            "transform": [{"calculate": "datum.value * 1.16", "as": "headroom"}],
                            "mark": {"type": "point", "opacity": 0},
                            "encoding": {
                                "y": y,
                                "x": {"field": "headroom", "type": "quantitative", "axis": None},
                            },
                        },
                        {
                            "transform": [{"filter": "datum.note !== null"}],
                            "mark": {
                                "type": "text",
                                "align": "left",
                                "dx": 4,
                                "fontSize": 10,
                                "fontStyle": "italic",
                                "opacity": 0.65,
                            },
                            "encoding": {
                                "y": y,
                                "x": {"datum": 0, "type": "quantitative", "axis": None},
                                "text": {"field": "note"},
                            },
                        },
                        *_axis_carriers(
                            y, _heading_axis(title, subtitle), TASK_ANCHORS.get(metric)
                        ),
                    ],
                },
            }
        )
    return {
        "$schema": VL_SCHEMA,
        "data": {"name": "tasks"},  # filled by assets/tasks.js, never at build time
        "params": _params(controls),
        "hconcat": columns,
        "spacing": 26,
        "config": {"view": {"stroke": None}, "axis": {"grid": False}},
    }


# The two quantities the calculator is graded on, in chart order (the group order
# and the opacity domain, so the solid bar is always time to first token).
# Generation alone is deliberately not one of them: the job's own rate joins a
# lane's decode ladder where the sweep ran thin (`_task_pack`), so scoring it
# would compare the job to itself. `assets/tasks.js` builds the rows with these
# exact phase strings — keep the two sides in step.
ACCURACY_PHASES = ("time to first token", "whole task")


def _accuracy_spec(controls: list[dict], lanes: list[str]) -> dict:
    """How far the calculator lands from the validation job, per lane: signed
    percent error, one bar per graded phase, computed in the page (the same
    `assets/tasks.js` arithmetic that fills the task grid prices the job's own
    shape and reads it against the job's measurements — `accuracyRows`).

    Zero — prediction equals measurement — is a rule down the middle; bars grow
    right where the prediction is slower than the machine really was, left where
    it is faster. The label at each bar's end is the signed error; hover carries
    the two numbers it was computed from. Unfaceted like the launch chart: the
    lanes here are the ones whose job scored, and the two phases already double
    every row."""
    y = {
        "field": "lane",
        "type": "nominal",
        "title": None,
        "sort": {"field": "rank", "op": "min", "order": "ascending"},
        # the grid's lane-label idiom: names only, no domain spine or ticks
        "axis": {"labelLimit": 200, "labelFontSize": 11, "domain": False, "ticks": False},
    }
    offset = {"field": "phase", "type": "nominal", "scale": {"domain": list(ACCURACY_PHASES)}}
    color = {
        "field": "lane",
        "type": "nominal",
        "legend": None,
        "scale": _lane_scale(lanes, LANE_COLORS),
    }
    # `opacity` carries the phase and its bottom legend is the key — the same
    # idiom (and the same vega-lite 6.4.1 fillOpacity caveat) as `_launch_spec`.
    opacity = {
        "field": "phase",
        "type": "nominal",
        "scale": {"domain": list(ACCURACY_PHASES), "range": [1, 0.55]},
        "legend": {
            "orient": "bottom",
            "title": None,
            "symbolType": "square",
            "symbolFillColor": "currentColor",
            "symbolStrokeWidth": 0,
        },
    }
    tooltip = [
        {"field": "pred_label", "title": "predicted"},
        {"field": "meas_label", "title": "measured"},
    ]
    x = {
        "field": "err_pct",
        "type": "quantitative",
        "title": None,
        "scale": {"zero": True},
        "axis": _value_axis("~r"),  # the grids' numeric-scale idiom
    }
    bare_x = {k: v for k, v in x.items() if k != "axis"}  # only one layer may define it
    return {
        "$schema": VL_SCHEMA,
        "title": {
            "text": "prediction error (%)",
            "anchor": "start",
            "subtitle": "right of zero: predicted slower than the job measured",
            "fontSize": 12,
            "subtitleFontSize": 10,
        },
        "data": {"name": "accuracy"},  # filled by assets/tasks.js, never at build time
        "params": _params(controls),
        "transform": _filters(controls),
        "width": 400,
        "height": {"step": 30},
        "layer": [
            {
                "mark": {"type": "bar", "height": 11, "cornerRadiusEnd": 3},
                "encoding": {
                    "y": y,
                    "yOffset": offset,
                    "x": x,
                    "color": color,
                    "opacity": opacity,
                    "tooltip": tooltip,
                },
            },
            # One row of its own data, or the rule would draw once per accuracy
            # row and the overdraw would read as a solid spine.
            {
                "data": {"values": [{}]},
                "mark": {"type": "rule", "color": "currentColor", "opacity": 0.3},
                "encoding": {"x": {"datum": 0, "type": "quantitative"}},
            },
            # The signed error at the bar's end, on whichever side of zero the bar
            # grew — alignment and clearance flip with the sign.
            {
                "mark": {
                    "type": "text",
                    "fontSize": 10,
                    "align": {"expr": "datum.err_pct < 0 ? 'right' : 'left'"},
                    "dx": {"expr": "datum.err_pct < 0 ? -4 : 4"},
                },
                "encoding": {
                    "y": y,
                    "yOffset": offset,
                    "x": bare_x,
                    "text": {"field": "err_label"},
                },
            },
            # Room past both ends, so a label never clips at the domain edge.
            {
                "transform": [{"calculate": "datum.err_pct * 1.35", "as": "headroom"}],
                "mark": {"type": "point", "opacity": 0},
                "encoding": {"y": y, "x": {"field": "headroom", "type": "quantitative"}},
            },
        ],
        "config": {"view": {"stroke": None}, "axis": {"grid": False}},
    }


# --------------------------------------------------------------- first launch
# The two things a lane pays once, before any rate applies. Order is the chart's
# group order and its opacity domain, so the solid bar is always the compile.
LAUNCH_PHASES = ("pipeline compilation", "cold first touch")

# What a lane says where there is no compile number. Only one state supports the
# claim that nothing was compiled: the harness pinned an empty cache and the run
# left it empty. Where the cache was not pinnable we know nothing about what was
# built beforehand, so the span is reported and marked rather than explained away.
LAUNCH_NOTES = {
    "nothing": "nothing compiled",
    "no_fit": "no cost function to net the warm pass out of",
    "absent": "not measured",
}

# Marks a first-launch span measured against a cache we could not pin, on the bar
# and in the subtitle that explains it.
UNVERIFIED_MARK = "*"


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

    Where the harness pinned the cache (`shader_cache == "redirected"`) an empty
    cache afterwards means the run genuinely compiled nothing — the one state in
    which that can be said. Where it could not (macOS, windows) the machine may have
    handed the driver pipelines it built earlier, so what is left after the
    subtraction is a floor of unknown tightness. `_launch_rows` marks that case;
    refusing to report it would only replace an uncertain number with a wrong
    claim."""
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


def _launch_label(seconds: float | None, pinned: bool) -> str | None:
    """The number printed at the end of a bar, marked where the span was measured
    against a cache the harness could not pin. Carried as text rather than formatted
    in the spec, because only the row knows whether it earned the mark."""
    if seconds is None:
        return None
    return f"{seconds:.1f}" if pinned else f"{seconds:.1f}{UNVERIFIED_MARK}"


def _launch_rows(df: pd.DataFrame) -> list[dict]:
    """Rows for the first-launch chart: up to two phases per (lane, model).

    *pipeline compilation* is the sweep's warmup span with its own prefill netted
    out (`_compile_seconds`) — the span is a width walk, so most of it is inference
    on most lanes, and the raw span rides along in the tooltip beside what was
    subtracted rather than disappearing. It reads as a cold from-scratch compile
    only where the harness pinned the cache (`shader_cache == 'redirected'`); there,
    an empty cache afterwards means nothing was compiled and the row says so instead
    of printing seconds. Where the cache could not be pinned the estimate is
    reported and marked (see `_launch_label`) — that platform tells us nothing about
    what it had built already, which is a reason to qualify the number, not to
    withhold it. CPU lanes emit no compile row at all: the GPU backend leaves a
    fixed, model-independent pipeline set behind at registry init, so their non-zero
    `shader_bytes` is that artifact and none of their warmup is compilation.

    *cold first touch* is the job's `cold_start_ms`, which the harness measures once
    per machine and model file and attributes to the first cell that scored it — so
    it is emitted only where it exists, never spread across the lanes that later
    read the same warm file.

    A (lane, model) with no number at all is dropped: a row of two notes and no ink
    is not a chart row."""
    rows: list[dict] = []
    if df.empty:
        return rows
    rank = _speed_rank(df)
    for r in df.itertuples():
        gpu = r.family != "cpu"
        shader_bytes = _plain(getattr(r, "shader_bytes", None), 0)
        base = {
            "lane": r.lane,
            "dev_class": r.dev_class,
            "machine": r.machine,
            "model": r.model,
            "quant": r.quant,
            "backend": r.backend,
            "rank": rank.get((r.model, r.lane), len(rank)),
        }
        cell = []
        if gpu:
            pinned = r.shader_cache == "redirected"
            span = _seconds(getattr(r, "shader_warmup_ms", None))
            netted = _seconds(_warm_pass_prefill_ms(r))
            value = _compile_seconds(r)
            if value is not None:
                note = None
            elif pinned and not shader_bytes:
                note = LAUNCH_NOTES["nothing"]
            elif span is not None and netted is None:
                note = LAUNCH_NOTES["no_fit"]
            else:
                note = LAUNCH_NOTES["absent"]
            # How much was compiled, out of which cache, and what the estimate was
            # cut from describe *this* phase and only this one — a cold read of the
            # weights compiles nothing and nets nothing out.
            cell.append(
                {
                    **base,
                    "phase": LAUNCH_PHASES[0],
                    "seconds": value,
                    "note": note,
                    "cache": r.shader_cache,
                    "label": _launch_label(value, pinned),
                    "span": span,
                    "netted": netted,
                    "mb": _plain(shader_bytes / 1e6) if shader_bytes else None,
                }
            )
        cold = _seconds(getattr(r, "cold_start_ms_p50", None))
        if cold is not None:
            cell.append(
                {
                    **base,
                    "phase": LAUNCH_PHASES[1],
                    "seconds": cold,
                    "note": None,
                    "cache": None,
                    "mb": None,
                    "span": None,
                    "netted": None,
                    "label": _launch_label(cold, True),
                }
            )
        if any(row["seconds"] is not None for row in cell):
            rows += cell
    return rows


def _launch_spec(rows: list[dict], controls: list[dict], lanes: list[str]) -> dict:
    """One-time costs per lane: two grouped bars, compilation solid and cold first
    touch half-lit, sharing the lane's own color and the lane order of the grid.

    Not faceted by device class — the lanes that appear here are the ones that
    compile or that loaded a file cold, which is a shorter list than the grid's, and
    every bar is in the same unit. Opacity carries the phase, and its legend at the
    bottom is the key; the color legend stays off, because the lane is already the
    axis label. A lane with nothing to show for a phase carries the reason at the
    axis, in the phase's own row.

    The bars are one mark drawn in two phase-filtered layers, because the two phases
    do not answer the same questions: how much was compiled and out of which cache
    belong to the compile, and a cold read of the weights has no such fields to show.
    The axis rides the compile layer, the one that anchors the scale — two layers on
    one scale each defining an axis would merge into a doubled heading."""
    y = {
        "field": "lane",
        "type": "nominal",
        "title": None,
        "sort": {"field": "rank", "op": "min", "order": "ascending"},
        "axis": {"labelLimit": 200, "labelFontSize": 11},
    }
    offset = {"field": "phase", "type": "nominal", "scale": {"domain": list(LAUNCH_PHASES)}}
    # A CPU-only selection leaves this chart with nothing to draw, which collapses
    # the domain to [0, 0] — and on a zero-width domain vega derives six decimals of
    # tick precision, so the empty chart would read "0.000000". An explicit trimmed
    # format keeps that lone tick at "0" without padding the populated ticks either.
    x = {
        "field": "seconds",
        "type": "quantitative",
        "title": None,
        "scale": {"zero": True},
        "axis": {"format": "~r"},
    }
    color = {
        "field": "lane",
        "type": "nominal",
        "legend": None,
        "scale": _lane_scale(lanes, LANE_COLORS),
    }
    # The phase rides `opacity`, not `fillOpacity`: vega-lite 6.4.1 compiles no
    # legend for `fillOpacity` (verified against the shipped build), and the legend
    # *is* the phase key. On a bar with no stroke the two channels draw the same
    # thing. The symbols take the surface's own ink, so the pair reads as one square
    # at two strengths in either theme.
    opacity = {
        "field": "phase",
        "type": "nominal",
        "scale": {"domain": list(LAUNCH_PHASES), "range": [1, 0.5]},
        "legend": {
            "orient": "bottom",
            "title": None,
            "symbolType": "square",
            "symbolFillColor": "currentColor",
            "symbolStrokeWidth": 0,
        },
    }
    # The lane is the axis label and the model is a selection: what a bar cannot say
    # is which of the two phases it is and how long that phase took.
    common = [{"field": "phase"}, {"field": "seconds", "title": "seconds"}]
    tooltips = {
        LAUNCH_PHASES[0]: [
            *common,
            {"field": "mb", "title": "compiled (MB)"},
            {"field": "cache", "title": "shader cache"},
            # The estimate's own arithmetic, so a reader can see what
            # it was cut from instead of taking the difference on
            # trust: seconds == span - netted.
            {"field": "span", "title": "warm pass (s)"},
            {"field": "netted", "title": "its prefill (s)"},
        ],
        LAUNCH_PHASES[1]: common,
    }
    bare_x = {k: v for k, v in x.items() if k != "axis"}  # only one layer may define it
    bars = [
        {
            "transform": [{"filter": f"datum.seconds !== null && datum.phase === '{phase}'"}],
            "mark": {"type": "bar", "height": 11, "cornerRadiusEnd": 3},
            "encoding": {
                "y": y,
                "yOffset": offset,
                "x": x if i == 0 else bare_x,
                "color": color,
                "opacity": opacity,
                "tooltip": tip,
            },
        }
        for i, (phase, tip) in enumerate(tooltips.items())
    ]
    return {
        "$schema": VL_SCHEMA,
        "title": {
            "text": "one-time first launch (s)",
            "anchor": "start",
            # Two lines, because one runs past the container and clips.
            "subtitle": [
                "compilation estimated from an empty shader cache: "
                "the warm pass, less what its width walk costs as "
                "prefill",
                f"{UNVERIFIED_MARK} no cache to empty here (macOS, "
                f"windows), so the estimate is a floor",
            ],
            "fontSize": 12,
            "subtitleFontSize": 10,
        },
        "data": {"values": rows},
        "params": _params(controls),
        "transform": _filters(controls),
        "width": 400,
        "height": {"step": 30},
        "layer": [
            *bars,
            {
                "transform": [{"filter": "datum.seconds !== null"}],
                "mark": {"type": "text", "align": "left", "dx": 4, "fontSize": 10},
                "encoding": {
                    "y": y,
                    "yOffset": offset,
                    "x": {"field": "seconds", "type": "quantitative"},
                    "text": {"field": "label"},
                },
            },
            {
                "transform": [{"calculate": "datum.seconds * 1.16", "as": "headroom"}],
                "mark": {"type": "point", "opacity": 0},
                "encoding": {"y": y, "x": {"field": "headroom", "type": "quantitative"}},
            },
            {
                "transform": [{"filter": "datum.note !== null"}],
                "mark": {
                    "type": "text",
                    "align": "left",
                    "dx": 4,
                    "fontSize": 10,
                    "fontStyle": "italic",
                    "opacity": 0.65,
                },
                "encoding": {
                    "y": y,
                    "yOffset": offset,
                    "x": {"datum": 0, "type": "quantitative"},
                    "text": {"field": "note"},
                },
            },
        ],
        "config": {"view": {"stroke": None}, "axis": {"grid": False}},
    }


def _single_depth(df: pd.DataFrame) -> pd.Series:
    """Which rows belong to a (lane, model) the sweep only reached once. Its budget
    stops at the first depth on a slow lane, so that lane has a measurement but no
    slope — one point drawn as a line looks like a broken chart, so it is drawn as
    a point and said out loud."""
    return df.groupby(["lane", "model"]).lane.transform("size") < 2


def _curve_spec(
    rows: list[dict],
    lanes: list[str],
    controls: list[dict],
    *,
    x: str,
    y: str,
    x_title: str,
    y_title: str,
    log_y: bool = False,
) -> dict:
    """A curve per lane, plus the lanes that have one depth instead of a curve —
    hollow marks, so a single measurement never masquerades as a trend."""
    encoding = {
        "x": {"field": x, "type": "quantitative", "title": x_title},
        "y": {
            "field": y,
            "type": "quantitative",
            "title": y_title,
            "scale": {"type": "log"} if log_y else {},
        },
        "color": {
            "field": "lane",
            "scale": _lane_scale(lanes, LANE_COLORS),
            "legend": {"orient": "bottom", "columns": 2, "title": None},
        },
        # The point's two coordinates, read off the axes it sits between — which lane's
        # curve it belongs to is the legend's job and the color's.
        "tooltip": [{"field": x, "title": x_title}, {"field": y, "title": y_title}],
    }
    return {
        "$schema": VL_SCHEMA,
        "data": {"values": rows},
        "params": _params(controls),
        "transform": _filters(controls),
        "width": 400,
        "height": 220,
        "layer": [
            {
                "transform": [{"filter": "!datum.single"}],
                "mark": {"type": "line", "point": {"size": 30}, "strokeWidth": 2},
                "encoding": encoding,
            },
            {
                "transform": [{"filter": "datum.single"}],
                "mark": {"type": "point", "size": 70, "filled": False, "strokeWidth": 2},
                "encoding": encoding,
            },
        ],
        "config": {"view": {"stroke": None}},
    }


# ------------------------------------------------------------------------ memory
# What a lane says when it reports no per-process device memory. Null is not zero:
# on unified memory there is one pool and the RSS already contains it; on a lane
# with no per-process accounting the platform simply cannot answer.
VRAM_NOTES = {"unified": "unified memory: in RSS", "n/a": "no per-process VRAM here"}


def _memory_rows(mem: pd.DataFrame) -> list[dict]:
    """Rows for the allocator curve: one per (lane, model, quant, backend, n_ctx),
    with the three pools the allocator reserved and their total. These are the
    allocator's own numbers at each context size the sweep sized a context for, so
    they are exact — there is nothing to average and no spread to draw."""
    if mem.empty:
        return []
    rows = []
    for r in mem.itertuples():
        total = r.weights_mb + r.kv_mb + r.compute_mb
        rows.append(
            {
                "lane": r.lane,
                "dev_class": r.dev_class,
                "model": r.model,
                "quant": r.quant,
                "backend": r.backend,
                "n_ctx": int(r.n_ctx),
                "weights_mb": _plain(r.weights_mb),
                "kv_mb": _plain(r.kv_mb),
                "compute_mb": _plain(r.compute_mb),
                "total_mb": _plain(total),
            }
        )
    return rows


def _footprint_spec(rows: list[dict], lanes: list[str], controls: list[dict]) -> dict:
    """The allocator's reservations against context size, one lane per color.

    Two lines per lane: the solid one is everything the allocator reserved, the
    dashed one is the weights, which do not move with context. The vertical gap
    between them is what the context itself costs — KV cache plus compute buffers —
    at the sizes the sweep measured. Zero-based, because the question is how much of
    a machine's memory the whole thing takes, not how the total varies."""
    x = {"field": "n_ctx", "type": "quantitative", "title": "context size (tokens)"}
    color = {
        "field": "lane",
        "scale": _lane_scale(lanes, LANE_COLORS),
        "legend": {"orient": "bottom", "columns": 2, "title": None},
    }
    # The three pools behind the two lines, and their sum: the chart draws the total
    # and the weights floor, so the split between KV and compute is what it cannot.
    tooltip = [
        {"field": "n_ctx", "title": "context size (tokens)"},
        {"field": "weights_mb", "title": "weights (MB)"},
        {"field": "kv_mb", "title": "KV cache (MB)"},
        {"field": "compute_mb", "title": "compute buffers (MB)"},
        {"field": "total_mb", "title": "total reserved (MB)"},
    ]

    def y_of(field: str) -> dict:
        return {
            "field": field,
            "type": "quantitative",
            "title": "MB reserved",
            "scale": {"zero": True},
        }

    return {
        "$schema": VL_SCHEMA,
        "data": {"values": rows},
        "params": _params(controls),
        "transform": _filters(controls),
        "width": 400,
        "height": 220,
        "layer": [
            {
                "mark": {"type": "line", "point": {"size": 30}, "strokeWidth": 2},
                "encoding": {"x": x, "y": y_of("total_mb"), "color": color, "tooltip": tooltip},
            },
            {
                "mark": {"type": "line", "strokeDash": [4, 3], "strokeWidth": 2, "opacity": 0.45},
                "encoding": {"x": x, "y": y_of("weights_mb"), "color": color, "tooltip": tooltip},
            },
        ],
        "config": {"view": {"stroke": None}},
    }


def _job_memory_rows(ok: pd.DataFrame, mem: pd.DataFrame) -> list[dict]:
    """Rows for the process-level chart: one per (lane, model, quant, backend) that
    scored a job, carrying what the process held and the allocator total to read it
    against.

    Peak resident memory is the whole process; the device-local pool is a *separate*
    number wherever the platform reports one per process (a Vulkan lane on DRM holds
    ~1.5 GB of host RSS and ~1.5 GB of VRAM at the same time), and null wherever it
    does not — never zero, so a lane that cannot answer says so. The allocator
    reference is the deepest context the ladder sized, which is why it usually sits
    to the right of a job that ran a shallower one."""
    if ok.empty:
        return []
    keys = ["lane", "model", "quant", "backend"]
    frame = ok
    if not mem.empty:
        deepest = mem.loc[mem.groupby(keys, sort=False).n_ctx.idxmax()].copy()
        deepest["alloc_total"] = (deepest.weights_mb + deepest.kv_mb + deepest.compute_mb).round(1)
        frame = ok.merge(
            deepest.rename(columns={"n_ctx": "alloc_ctx"})[[*keys, "alloc_total", "alloc_ctx"]],
            on=keys,
            how="left",
        )
    rows = []
    for r in frame.itertuples():
        rss = _plain(getattr(r, "decode_rss_peak_mb_p50", None))
        if rss is None:
            continue  # nothing to draw a row around
        vram = _plain(getattr(r, "decode_vram_peak_mb_p50", None))
        alloc_ctx = _plain(getattr(r, "alloc_ctx", None), 0)
        rows.append(
            {
                "lane": r.lane,
                "dev_class": r.dev_class,
                "machine": r.machine,
                "model": r.model,
                "quant": r.quant,
                "backend": r.backend,
                "rss_peak": rss,
                "rss_sustained": _plain(getattr(r, "decode_rss_sustained_mb_p50", None)),
                "prefill_rss_peak": _plain(getattr(r, "prefill_rss_peak_mb_p50", None)),
                "vram_peak": vram,
                "vram_method": r.vram_method,
                "vram_note": None
                if vram is not None
                else VRAM_NOTES.get(r.vram_method, "not measured"),
                "alloc_total": _plain(getattr(r, "alloc_total", None)),
                "alloc_ctx": int(alloc_ctx) if alloc_ctx is not None else None,
                # The label rides past the far end of the two measured marks, so it
                # never prints over the device-pool dot when the two pools are alike.
                "label_x": max(v for v in (rss, vram) if v is not None),
            }
        )
    return rows


def _job_memory_spec(rows: list[dict], lanes: list[str], controls: list[dict]) -> dict:
    """What the process held, per lane: a bar for peak resident memory, a dot for the
    device-local pool where the platform reports one, and a thin dashed marker at the
    allocator's total as the reference. A lane with no device-pool number carries the
    reason under its bar instead of a dot.

    The dashed marker is a rotated `stroke` symbol rather than a tick mark: vega
    compiles a tick to a filled rect, whose stroke dash would never show."""
    y = {
        "field": "lane",
        "type": "nominal",
        "title": None,
        "sort": lanes,
        "axis": {"labelLimit": 200, "labelFontSize": 11},
    }
    color = {
        "field": "lane",
        "type": "nominal",
        "legend": None,
        "scale": _lane_scale(lanes, LANE_COLORS),
    }
    # Each mark answers for itself: the bar is the process, the dashed marker is the
    # allocator's reference, the dot is the device pool. Splitting them keeps a lane
    # that reports no pool from printing an empty pool row — its bar never had one.
    resident = [
        {"field": "rss_peak", "title": "peak resident (MB)"},
        {"field": "rss_sustained", "title": "sustained resident (MB)"},
        {"field": "prefill_rss_peak", "title": "peak resident, prompt (MB)"},
    ]
    reserved = [
        {"field": "alloc_total", "title": "allocator total (MB)"},
        {"field": "alloc_ctx", "title": "allocator total at context"},
    ]
    pool = [
        {"field": "vram_peak", "title": "peak device-local (MB)"},
        {"field": "vram_method", "title": "device-local source"},
    ]
    return {
        "$schema": VL_SCHEMA,
        "data": {"values": rows},
        "params": _params(controls),
        "transform": _filters(controls),
        "width": 400,
        "height": {"step": 27},
        "layer": [
            {
                "mark": {"type": "bar", "height": 9, "cornerRadiusEnd": 3},
                "encoding": {
                    "y": y,
                    "x": {"field": "rss_peak", "type": "quantitative", "title": "MB"},
                    "color": color,
                    "tooltip": resident,
                },
            },
            {
                "transform": [{"filter": "datum.alloc_total !== null"}],
                "mark": {
                    "type": "point",
                    "shape": "stroke",
                    "angle": 90,
                    "size": 200,
                    "strokeWidth": 1.2,
                    "strokeDash": [2, 2],
                    "color": "currentColor",
                    "opacity": 0.8,
                },
                "encoding": {
                    "y": y,
                    "x": {"field": "alloc_total", "type": "quantitative"},
                    "tooltip": reserved,
                },
            },
            # The pool dot rides just above its bar: the two numbers are often within
            # a few MB of each other (one Vulkan lane holds 1,563 MB of RSS and
            # 1,569 MB of VRAM), and a dot at the bar's own height would read as a
            # rounded end rather than a second measurement.
            {
                "transform": [{"filter": "datum.vram_peak !== null"}],
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 46,
                    "yOffset": -9,
                    "stroke": "currentColor",
                    "strokeWidth": 1,
                },
                "encoding": {
                    "y": y,
                    "x": {"field": "vram_peak", "type": "quantitative"},
                    "color": color,
                    "tooltip": pool,
                },
            },
            {
                "mark": {"type": "text", "align": "left", "dx": 9, "fontSize": 10},
                "encoding": {
                    "y": y,
                    "x": {"field": "label_x", "type": "quantitative"},
                    "text": {"field": "rss_peak", "format": ".0f"},
                },
            },
            {
                "transform": [{"calculate": "datum.label_x * 1.16", "as": "headroom"}],
                "mark": {"type": "point", "opacity": 0},
                "encoding": {"y": y, "x": {"field": "headroom", "type": "quantitative"}},
            },
            # The note sits under its own bar: a lane label is already on the left and
            # the row's right edge belongs to the allocator reference.
            {
                "transform": [{"filter": "datum.vram_note !== null"}],
                "mark": {
                    "type": "text",
                    "align": "left",
                    "dx": 2,
                    "dy": 10,
                    "fontSize": 9,
                    "fontStyle": "italic",
                    "opacity": 0.65,
                },
                "encoding": {
                    "y": y,
                    "x": {"datum": 0, "type": "quantitative"},
                    "text": {"field": "vram_note"},
                },
            },
        ],
        "config": {"view": {"stroke": None}, "axis": {"grid": False}},
    }


# ---------------------------------------------------------------- thread scaling
# The ladder's two phases, in section order, each in the unit its fit was taken in
# — the milliseconds of one fixed prefill chunk, the tok/s of one fixed decode
# burst. Both work units are small and deliberately not the main sweep's, so a
# number here means something only against the same lane's other widths. `y_title`
# names the unit with the work unit the rows carry; `y_plain` is what it says when a
# selection somehow mixes two work units.
THREAD_PHASES = (
    {
        "phase": "prefill",
        "title": "prompt reading, by thread width",
        "subtitle": "one chunk from an empty cache per width — lower is better; "
        "compare widths, not charts",
        "y_title": "ms per {n}-token chunk",
        "y_plain": "ms per chunk",
        "value_title": "chunk (ms)",
    },
    {
        "phase": "decode",
        "title": "generation, by thread width",
        "subtitle": "one burst per width, at the same already-primed fill; "
        "compare widths, not charts",
        "y_title": "tok/s of a {n}-token burst",
        "y_plain": "tok/s per burst",
        "value_title": "burst (tok/s)",
    },
)


# How many points a fitted curve is drawn from, between one thread and the widest
# width that lane measured. Enough that a hyperbola reads as a curve; the fit is
# never sampled past its evidence.
def _thread_rows(thr: pd.DataFrame) -> list[dict]:
    """The ladder's measured points: one row per (lane, model, phase, width).

    `value` is the phase's own unit — the chunk's milliseconds for prefill, the
    burst's tok/s for decode. `at_width` marks the width the lane actually runs: the
    one point in the chart that is also an operating point. A prefill point has no
    `kv_fill` (its chunk starts from an empty cache), and that reads as None, never
    NaN — the island is strict JSON."""
    rows: list[dict] = []
    if thr.empty:
        return rows
    for r in thr.itertuples():
        value = _plain(r.ms, 2) if r.phase == "prefill" else _plain(r.tps, 2)
        if value is None:
            continue
        width = _plain(r.threads_batch if r.phase == "prefill" else r.threads_decode, 0)
        fill = _plain(r.kv_fill, 0)
        rows.append(
            {
                "lane": r.lane,
                "dev_class": r.dev_class,
                "machine": r.machine,
                "model": r.model,
                "quant": r.quant,
                "backend": r.backend,
                "phase": r.phase,
                "kind": "point",
                "threads": int(r.threads),
                "value": value,
                "tokens": int(r.tokens),
                "kv_fill": None if fill is None else int(fill),
                "at_width": bool(width is not None and int(width) == int(r.threads)),
                "label": None,
            }
        )
    return rows


def _thread_efficiency(phase: str, lo: dict, hi: dict) -> float:
    """What one step up the ladder bought, against what it would have bought if the
    added threads were free and perfectly divided. Prefill is a time, so its speedup
    is the ratio the other way up; decode is a rate."""
    ideal = hi["threads"] / lo["threads"]
    gained = lo["value"] / hi["value"] if phase == "prefill" else hi["value"] / lo["value"]
    return gained / ideal


def _thread_steps(points: list[dict]) -> list[dict]:
    """One row per adjacent pair of measured widths: what the step actually bought.

    This is the section's only derived number, and it is a ratio between two things
    that were measured — no curve is fitted through the ladder and nothing is
    extrapolated past it. Drawn at the midpoint of the step it describes, so it reads
    along the curve rather than at an axis."""
    by_lane: dict[tuple[str, str, str], list[dict]] = {}
    for row in points:
        by_lane.setdefault((row["lane"], row["model"], row["phase"]), []).append(row)
    rows: list[dict] = []
    for (_lane, _model, phase), group in sorted(by_lane.items()):
        ladder = sorted(group, key=lambda r: r["threads"])
        for lo, hi in zip(ladder, ladder[1:], strict=False):
            if not (lo["value"] > 0 and hi["value"] > 0):
                continue
            efficiency = _thread_efficiency(phase, lo, hi)
            rows.append(
                {
                    **{
                        k: lo[k]
                        for k in (
                            "lane",
                            "dev_class",
                            "machine",
                            "model",
                            "quant",
                            "backend",
                            "phase",
                            "tokens",
                            "kv_fill",
                        )
                    },
                    "kind": "step",
                    "at_width": False,
                    "threads": round((lo["threads"] + hi["threads"]) / 2, 3),
                    "value": round((lo["value"] + hi["value"]) / 2, 2),
                    "label": f"{lo['threads']}→{hi['threads']} threads: {efficiency:.0%} of ideal",
                }
            )
    return rows


def _thread_scalar_rows(points: list[dict], steps: list[dict]) -> list[dict]:
    """The headline numbers behind the two charts, one row per (lane, model) — all of
    them read off measured points.

    The widths run are the ladder itself, so a reader can see how much of the machine
    it covered; the top step is what the last doubling actually bought. There is no
    ceiling column and no 90%-of-peak column: both were fitted asymptotes, and this
    ladder stops at the widest width the lane was willing to run, which is where an
    asymptote is least constrained. What a wider width would buy is not in evidence
    here, so it is not claimed here."""
    keyed: dict[tuple[str, str], dict] = {}
    for row in points:
        rec = keyed.setdefault((row["lane"], row["model"]), {"widths": {}, "run": {}, "top": {}})
        rec["widths"].setdefault(row["phase"], set()).add(row["threads"])
        if row["at_width"]:
            rec["run"][row["phase"]] = row["threads"]
    for row in steps:  # the last step of each ladder is the informative one
        rec = keyed.get((row["lane"], row["model"]))
        if rec is not None:
            best = rec["top"].get(row["phase"])
            if best is None or row["threads"] > best["threads"]:
                rec["top"][row["phase"]] = row

    rows = []
    for (lane, model), rec in sorted(keyed.items()):
        run = sorted(set(rec["run"].values()))
        widths = sorted({w for s in rec["widths"].values() for w in s})
        step = {p: (rec["top"].get(p) or {}).get("label", "—") for p in ("prefill", "decode")}
        rows.append(
            {
                "lane": lane,
                "model": model,
                "width": ", ".join(str(w) for w in run) or "—",
                "measured": ", ".join(str(w) for w in widths) or "—",
                "prefill_step": step["prefill"],
                "decode_step": step["decode"],
                "note": "two widths only" if len(widths) < 3 else "—",
            }
        )
    return rows


def _thread_tooltip(value_title: str, *, fill: bool) -> list[dict]:
    """One ladder point on hover: the width, its number in the phase's own unit, and
    the fill it was primed at where a phase has one. The lane is the color and the
    legend, the work unit is in the y-axis title, and the widths are the dots
    themselves — none of that is repeated here. A prefill chunk starts from an empty
    cache, so that chart has no fill row at all rather than an empty one. No fit
    quality either: nothing is fitted through these points any more."""
    return [
        {"field": "threads", "title": "intra-op threads"},
        {"field": "value", "title": value_title},
        *([{"field": "kv_fill", "title": "primed fill (tokens)"}] if fill else []),
    ]


def _thread_spec(
    rows: list[dict],
    lanes: list[str],
    controls: list[dict],
    *,
    title: str,
    subtitle: str,
    y_title: str,
    value_title: str,
) -> dict:
    """One phase's thread ladder: the measured widths as dots, joined so the shape
    reads. The line is interpolation between measurements and nothing more — no curve
    is fitted through them and nothing is drawn past the widest width run.

    The hollow ring is the width the lane actually runs — every other dot is a width
    the ladder visited to find the shape.

    On hover, that lane's steps label themselves with what each one bought against a
    perfect division of the work. One lane at a time: every lane's at once is more
    annotation than chart.

    Same frame as the cost curves (400×220, lane hue, legend at the bottom) and the
    same controls, because it is the same lanes seen from a different axis. Unlike
    the task grid this chart's data is built here: the domains are fixed by the
    measurement, so nothing has to be inserted at view time."""
    x = {
        "field": "threads",
        "type": "quantitative",
        "title": "intra-op threads",
        # Threads are counted, so the axis steps in whole ones — and it starts at
        # one thread rather than zero, which is not a width anything can run.
        "scale": {"zero": False, "nice": False},
        "axis": {"tickMinStep": 1},
    }
    y = {"field": "value", "type": "quantitative", "title": y_title, "scale": {"zero": True}}
    color = {
        "field": "lane",
        "scale": _lane_scale(lanes, LANE_COLORS),
        "legend": {"orient": "bottom", "columns": 2, "title": None},
    }
    tooltip = _thread_tooltip(value_title, fill=any(r.get("kv_fill") is not None for r in rows))
    # A step label sits on the segment it describes, lifted clear of the line.
    label = {
        "type": "text",
        "align": "center",
        "baseline": "bottom",
        "dy": -7,
        "fontSize": 9,
    }
    # A step belongs to one lane, and every lane's at once buried the curves the chart
    # is about. They are asked for instead: point at a curve and that lane's own steps
    # label themselves.
    #
    # The target is a fat invisible copy of the line, because a 2px path is not
    # something a pointer can be asked to land on. It sits *under* the dots so their
    # tooltips still win the pointer — which is also why `nearest` is not used here:
    # it overlays a voronoi whose datum is the mark item rather than the row, and
    # every tooltip field on the layer it is declared on comes out `undefined`.
    #
    # Nothing clears the selection. Crossing a dot would otherwise drop the lane's
    # labels for as long as the pointer sat on it, and a reader who has stopped
    # pointing is still reading the lane they stopped on.
    hover = {
        "name": "lane_hover",
        "select": {
            "type": "point",
            "fields": ["lane"],
            "on": "pointerover",
            "clear": False,
        },
    }
    only_hovered = {"filter": {"param": "lane_hover", "empty": False}}
    return {
        "$schema": VL_SCHEMA,
        "title": {
            "text": title,
            "anchor": "start",
            "subtitle": subtitle,
            "fontSize": 12,
            "subtitleFontSize": 10,
        },
        "data": {"values": rows},
        "params": _params(controls),
        "transform": _filters(controls),
        "width": 400,
        "height": 220,
        "layer": [
            # The measured widths, joined. Interpolation for legibility, not a fit:
            # it stops at the widest width run, because nothing was measured past it.
            {
                "transform": [{"filter": "datum.kind === 'point'"}],
                "mark": {"type": "line", "strokeWidth": 2},
                "encoding": {"x": x, "y": y, "color": color},
            },
            # The hover target: that line again, fat and invisible, one path per lane.
            # Under the dots, so a dot's own tooltip still wins the pointer.
            {
                "transform": [{"filter": "datum.kind === 'point'"}],
                "params": [hover],
                "mark": {"type": "line", "strokeWidth": 16, "opacity": 0},
                "encoding": {"x": x, "y": y, "detail": {"field": "lane"}},
            },
            {
                "transform": [{"filter": "datum.kind === 'point'"}],
                "mark": {"type": "point", "filled": True, "size": 55},
                "encoding": {"x": x, "y": y, "color": color, "tooltip": tooltip},
            },
            # The operating point, ringed: the width this lane's runs are measured at
            # everywhere else on the page.
            {
                "transform": [{"filter": "datum.kind === 'point' && datum.at_width"}],
                "mark": {"type": "point", "filled": False, "size": 110, "strokeWidth": 2},
                "encoding": {"x": x, "y": y, "color": color, "tooltip": tooltip},
            },
            # What each step of the hovered lane's ladder bought, on the step itself.
            {
                "transform": [{"filter": "datum.kind === 'step'"}, only_hovered],
                "mark": label,
                "encoding": {"x": x, "y": y, "color": color, "text": {"field": "label"}},
            },
        ],
        "config": {"view": {"stroke": None}},
    }


def build(published: Path, out: Path, vega_cache: Path | None = None) -> None:
    df = load_results(published)
    empty = df.empty
    specs: dict[str, dict] = {}
    context: dict = {
        "built": date.today().isoformat(),
        "project_url": PROJECT_URL,
        "install_bash": INSTALL_BASH,
        "install_ps": INSTALL_PS,
        "empty": empty,
        "specs": specs,
        "lane_dark_map": dict(zip(LANE_COLORS, LANE_COLORS_DARK, strict=True)),
    }

    if not empty:
        df = _with_lanes(df)
        ok = df[df.status == "ok"]
        lanes = _lane_order(df)
        # Open on the model the most lanes could measure — the widest comparison.
        default_model = (
            ok.groupby("model").lane.nunique().sort_values(ascending=False).index[0]
            if len(ok)
            else sorted(df.model.dropna().unique())[0]
        )
        sweeps = _with_lanes(load_sweeps(published))
        grid_rows = _grid_rows(df, _depth_ranges(sweeps))
        controls = _controls(grid_rows, default_model)
        context["controls"] = [c for c in controls if c["render"]]
        specs["grid"] = _grid_spec(grid_rows, controls, lanes)

        pack = _task_pack(df, sweeps)
        if pack["records"]:
            specs["tasks"] = _task_spec(controls, lanes)
            context["tasks"] = True
            context["task_pack"] = pack
            context["task_list"] = pack["tasks"]
            # The accuracy chart needs a measurement to grade against: at least
            # one lane whose validation job scored.
            if any(rec["measured"] for rec in pack["records"]):
                specs["accuracy"] = _accuracy_spec(controls, lanes)
                context["accuracy"] = True

        launch_rows = _launch_rows(df)
        if launch_rows:
            specs["launch"] = _launch_spec(launch_rows, controls, lanes)
            context["launch"] = True

        pre = sweeps[(sweeps.kind == "prefill") & sweeps.ttft_ms.notna()].copy()
        dec = sweeps[(sweeps.kind == "decode") & sweeps.tps_p50.gt(0)].copy()
        keep = ["lane", "dev_class", "model", "quant", "backend", "single"]
        if len(pre):
            pre["single"] = _single_depth(pre)
            specs["curve-ttft"] = _curve_spec(
                pre[[*keep, "tokens", "ttft_ms"]].to_dict("records"),
                lanes,
                controls,
                x="tokens",
                y="ttft_ms",
                x_title="prompt tokens",
                y_title="time to first token (ms)",
            )
        if len(dec):
            dec["single"] = _single_depth(dec)
            specs["curve-decode"] = _curve_spec(
                dec[[*keep, "kv_fill", "tps_p50"]].to_dict("records"),
                lanes,
                controls,
                x="kv_fill",
                y="tps_p50",
                log_y=True,
                x_title="context already used (tokens)",
                y_title="tok/s",
            )
        context["curves"] = [sid for sid in ("curve-ttft", "curve-decode") if sid in specs]

        # An empty frame has no columns for `_with_lanes` to read, and a shelf whose
        # runs carried no allocator ladder is exactly that.
        mem = load_memory(published)
        if not mem.empty:
            mem = _with_lanes(mem)
        footprint_rows = _memory_rows(mem)
        if footprint_rows:
            specs["memory-footprint"] = _footprint_spec(footprint_rows, lanes, controls)
        job_memory_rows = _job_memory_rows(ok, mem)
        if job_memory_rows:
            specs["memory-job"] = _job_memory_spec(job_memory_rows, lanes, controls)
        context["memory"] = [sid for sid in ("memory-footprint", "memory-job") if sid in specs]

        # The thread ladder is optional measurement: CPU lanes only, and only from
        # submissions new enough to carry it. An empty frame has no columns for
        # `_with_lanes` to read, so the guard comes first.
        thr = load_thread_scaling(published)
        thread_ids: list[str] = []
        if not thr.empty:
            points = _thread_rows(_with_lanes(thr))
            steps = _thread_steps(points)
            rows = points + steps
            for phase in THREAD_PHASES:
                drawn = [r for r in rows if r["phase"] == phase["phase"]]
                if not any(r["kind"] == "point" for r in drawn):
                    continue  # a phase the ladder never reached draws nothing
                units = {r["tokens"] for r in drawn}
                y_title = (
                    phase["y_title"].format(n=units.pop()) if len(units) == 1 else phase["y_plain"]
                )
                sid = f"thread-{phase['phase']}"
                specs[sid] = _thread_spec(
                    drawn,
                    lanes,
                    controls,
                    title=phase["title"],
                    subtitle=phase["subtitle"],
                    y_title=y_title,
                    value_title=phase["value_title"],
                )
                thread_ids.append(sid)
            context["thread_scalars"] = _thread_scalar_rows(points, steps)
        context["threads"] = thread_ids

        probes = _with_lanes(load_probes(published))
        ceilings = []
        for lane_label, g in probes[probes.status == "ok"].groupby("lane"):
            gemm = g[g.kind == "gemm"].tflops.max()
            d2d = g[g.kind == "d2d"].gbs.max()
            ceilings.append(
                {
                    "lane": lane_label,
                    "gemm": f"{gemm:.1f}" if gemm == gemm else "—",
                    "d2d": f"{d2d:.0f}" if d2d == d2d else "—",
                }
            )
        context["ceilings"] = sorted(ceilings, key=lambda c: c["lane"])

        # The GPU column is what the lanes actually ran on: an iGPU is invisible to
        # the machine block's NVML/system_profiler probe, but its lane reports it.
        gpus = {
            m: ", ".join(sorted({d for d, f in zip(g.device, g.family, strict=True) if f != "cpu"}))
            for m, g in df.groupby("machine")
        }
        context["machines"] = [
            {
                "machine": r.machine,
                "cpu": r.cpu,
                "gpu": gpus.get(r.machine) or "—",
                "ram": f"{r.ram_gb:g} GB" if r.ram_gb == r.ram_gb else "?",
            }
            for r in df.drop_duplicates("machine").itertuples()
        ]
        bad = df[df.status != "ok"]
        context["unusable"] = [
            {
                "lane": r.lane,
                "machine": r.machine,
                "model": r.model,
                "quant": r.quant,
                "status": str(r.status).replace("_", " "),
            }
            for r in bad.itertuples()
        ]
        with_text = ok[ok.sample_completion.notna()].groupby("model", observed=True).head(1)
        context["completions"] = [
            {
                "who": f"{r.machine} · {r.model} {r.quant} · {r.lane}",
                "text": str(r.sample_completion).strip(),
            }
            for r in with_text.itertuples()
        ]
        # One fold, four panels: the reference tables the charts stand on. Only the
        # ones this shelf can fill are offered; machines is always among them.
        context["tabs"] = [
            {
                "id": "machines",
                "label": f"Machines ({len(context['machines'])})",
                "hint": "machines",
                "about": "What each submission ran on. The GPU column is what the lanes "
                "reported, which is the only place an iGPU shows up.",
            },
            *(
                [
                    {
                        "id": "ceilings",
                        "label": "Device ceilings",
                        "hint": "ceilings",
                        "about": "Bare probes with no model loaded: f16 GEMM and on-device "
                        "copies, the roof the inference numbers sit under.",
                    }
                ]
                if context["ceilings"]
                else []
            ),
            *(
                [
                    {
                        "id": "unusable",
                        "label": f"Unmeasurable cells ({len(context['unusable'])})",
                        "hint": "unmeasurable cells",
                        "about": "Attempted and produced no timing: killed by the backstop, "
                        "below the usable tok/s floor, or failed the brain-check.",
                    }
                ]
                if context["unusable"]
                else []
            ),
            *(
                [
                    {
                        "id": "samples",
                        "label": "Output samples",
                        "hint": "output samples",
                        "about": "One completion per model, decoded during the measured run "
                        "— the numbers came from a model that was working.",
                    }
                ]
                if context["completions"]
                else []
            ),
        ]
        context["stats"] = [
            {"v": f"{df.machine.nunique()}", "k": "machines"},
            {"v": f"{df.model.nunique()}", "k": "models"},
            {"v": f"{ok.lane.nunique()}", "k": "lanes"},
        ]

    env = Environment(
        loader=PackageLoader("bench_analysis"), autoescape=select_autoescape(default=True)
    )
    context["css"] = (PKG / "assets" / "report.css").read_text()
    context["tasks_js"] = (PKG / "assets" / "tasks.js").read_text()
    context["report_js"] = (PKG / "assets" / "report.js").read_text()
    context["vega_js"] = _vega_js(vega_cache or REPO / "third_party" / "vega")
    out.write_text(env.get_template("report.html.j2").render(context))


def main() -> None:
    ap = argparse.ArgumentParser(description="build the static report")
    ap.add_argument("--published", type=Path, default=REPO / "results" / "published")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "published" / "report.html")
    args = ap.parse_args()
    build(args.published, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
