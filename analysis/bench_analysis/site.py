"""Build the report: one self-contained, data-first HTML page.

The page answers "how fast is on-device inference on real machines" one model at
a time. A control row scopes everything below it — model, device class, quant,
backend — and a control is rendered only where the shelf holds more than one
value for it. Under it, three bar charts (decode tok/s, prefill tok/s, warm init)
share one lane axis: lanes are grouped into bands by what kind of device they are
(discrete GPU / integrated GPU / CPU) and sorted by generation speed inside each
band. The cost curves below read the same controls.

Two encodings, two jobs: in the grid, identity is the lane label on the axis, so
hue is free to carry the device class (three classes, one validated triple, never
exhausted). In the curves, identity *is* the line, so hue carries the lane — the
first eight in a fixed order, gray past that.

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

from . import load_probes, load_results, load_sweeps

PKG = Path(__file__).parent
REPO = PKG.parents[1]  # …/analysis/bench_analysis → the repo root
PROJECT_URL = "https://github.com/MonsieurTapir/llm-on-device-survey"

INSTALL_BASH = (
    "curl -fsSL https://raw.githubusercontent.com/MonsieurTapir/"
    "llm-on-device-survey/main/run.sh | bash"
)
INSTALL_PS = (
    "irm https://raw.githubusercontent.com/MonsieurTapir/"
    "llm-on-device-survey/main/run.ps1 | iex"
)

# Lane identity colors — validated (light + dark) against the six checks;
# assigned to lanes in fixed sorted order, never cycled past the overflow gray.
LANE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LANE_COLORS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"]
LANE_OVERFLOW = "#898781"

# Device classes: the band order in the grid (fastest kind of silicon first) and
# the hue each one wears. Three slots, so the palette can never run out — the
# triple is validated in both modes (the light green sits under 3:1 against the
# surface, which the per-bar value labels relieve).
CLASSES = ("discrete GPU", "integrated GPU", "CPU")
CLASS_COLORS = {"discrete GPU": LANE_COLORS[1],
                "integrated GPU": LANE_COLORS[2],
                "CPU": LANE_COLORS[0]}

VEGA_LIBS = (
    ("vega", alt.VEGA_VERSION),
    ("vega-lite", alt.VEGALITE_VERSION),
    ("vega-embed", alt.VEGAEMBED_VERSION),
)

# The spec dialect follows the vega-lite the page actually ships (see VEGA_LIBS).
VL_SCHEMA = f"https://vega.github.io/schema/vega-lite/v{alt.VEGALITE_VERSION.split('.')[0]}.json"

METRICS = (  # column key, title, value-label format (vega d3-format), subtitle
    ("decode", "generation (tok/s)", ".0f", None),
    ("prefill", "prompt reading (tok/s)", ".0f", None),
    ("init", "init, warm (s)", ".1f", "lower is better"),
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
    r"\s*\((?:RADV|ANV|NVK|LLVM|MESA|SWIFTSHADER|LAVAPIPE|GFX)[^)]*\)", re.IGNORECASE)
_VENDOR = re.compile(r"^(?:NVIDIA|AMD|Intel|Advanced Micro Devices,?(?: Inc\.?)?)\s+",
                     re.IGNORECASE)
# What a CPU brand string pads its model with — including the iGPU it mentions
# ("Ryzen 7 255 w/ Radeon 780M Graphics"), which belongs to the GPU lane, not here.
_CPU_TAIL = re.compile(r"\s*(?:\bw/\s.*|\d+-Core Processor|Processor|CPU @.*)$",
                       re.IGNORECASE)
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


def _dev_class(device: str, cpu: str, family: str) -> str:
    if family == "cpu":
        return "CPU"
    return "integrated GPU" if _integrated(device, cpu) else "discrete GPU"


def _family(provider: str) -> str:
    return provider.split(":")[0]


def _with_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """Add `family`, `dev_class` and the display `lane` ('RTX 5080 · vulkan'). A
    label two machines (or two lanes of one machine) would share is qualified until
    it is unique — rows are never silently pooled."""
    df = df.copy()
    df["family"] = [_family(p) for p in df.provider]
    df["dev_class"] = [_dev_class(d, c, f)
                       for d, c, f in zip(df.device, df.cpu, df.family, strict=True)]
    df["lane"] = [f"{_lane_chip(d, c, f)} · {f}"
                  for d, c, f in zip(df.device, df.cpu, df.family, strict=True)]
    for qualifier in ("machine", "provider"):
        shared = df.groupby("lane")[qualifier].transform("nunique") > 1
        df.loc[shared, "lane"] = [f"{lane} ({q})" for lane, q
                                  in zip(df.lane[shared], df[qualifier][shared], strict=True)]
    return df


def _lane_order(df: pd.DataFrame) -> list[str]:
    """Every lane, grouped by device class then named — the color domain, stable
    across models and filters (color follows the lane, never its rank)."""
    seen = df.drop_duplicates("lane")
    return [r.lane for r in sorted(
        seen.itertuples(), key=lambda r: (CLASSES.index(r.dev_class), r.lane))]


# ------------------------------------------------------------------ chart specs
def _grid_rows(df: pd.DataFrame) -> list[dict]:
    """Long-format rows for the grid: one per (lane, model, metric). Every cell of
    every measured (lane, model) is emitted — a cell with no number carries the
    reason instead — so the three metric columns keep identical rows, and no lane
    silently drops out of a band.

    `rank` is the lane's generation-speed position within its model; the y axis
    sorts on it, so all three columns order their bands the same way."""
    rank: dict[tuple[str, str], int] = {}
    for model, g in df[df.status == "ok"].groupby("model"):
        ordered = g.sort_values("decode_tps_p50", ascending=False)
        rank.update({(model, lane): i for i, lane in enumerate(ordered.lane)})

    rows = []
    for r in df.itertuples():
        cold = getattr(r, "cold_start_ms_p50", None)
        base = {
            "lane": r.lane, "dev_class": r.dev_class, "machine": r.machine,
            "backend": r.backend, "model": r.model, "quant": r.quant,
            "device": r.device, "rank": rank.get((r.model, r.lane), len(rank)),
            "cold_s": round(cold / 1e3, 1) if cold == cold and cold else None,
        }
        if r.status == "ok":
            values = {"decode": r.decode_tps_p50, "prefill": r.prefill_tps_p50,
                      "init": (r.model_load_ms_p50 + r.context_init_ms_p50) / 1e3}
            note = None
        else:  # measured nothing: the band keeps the row and states why
            values = dict.fromkeys(("decode", "prefill", "init"))
            note = str(r.status).replace("_", " ")
        for metric, value in values.items():
            usable = value is not None and value == value  # NaN-safe
            rows.append({**base, "metric": metric,
                         "value": round(float(value), 2) if usable else None,
                         "note": None if usable else note})
    return rows


def _lane_scale(lanes: list[str], colors: list[str]) -> dict:
    return {"domain": lanes,
            "range": [colors[i] if i < len(colors) else LANE_OVERFLOW
                      for i in range(len(lanes))]}


def _controls(rows: list[dict], default_model: str) -> list[dict]:
    """The control row: one entry per signal that has something to choose. `model`
    always carries a signal (the page shows one model at a time) but is only
    rendered as a select when the shelf holds more than one."""
    controls = []
    for signal, field, label in CONTROLS:
        values = sorted({r[field] for r in rows if r[field] is not None})
        if signal == "f_model":
            controls.append({"signal": signal, "field": field, "label": label,
                             "options": values, "value": default_model,
                             "render": len(values) > 1})
        elif len(values) > 1:
            controls.append({"signal": signal, "field": field, "label": label,
                             "options": ["all", *values], "value": "all",
                             "render": True})
    return controls


def _filters(controls: list[dict]) -> list[dict]:
    out = []
    for c in controls:
        field, signal = c["field"], c["signal"]
        out.append({"filter": f"datum.{field} === {signal}" if signal == "f_model"
                    else f"{signal} === 'all' || datum.{field} === {signal}"})
    return out


def _params(controls: list[dict]) -> list[dict]:
    return [{"name": c["signal"], "value": c["value"]} for c in controls]


def _grid_spec(rows: list[dict], controls: list[dict]) -> dict:
    """One hconcat: per metric, a bar chart banded (by device class) down a shared
    lane axis. The leftmost column carries the band and lane labels for all three;
    every bar carries its value, which is also the contrast relief the palette needs."""
    tooltip = [
        {"field": "lane", "title": "lane"},
        {"field": "machine", "title": "machine"},
        {"field": "device", "title": "device"},
        {"field": "dev_class", "title": "class"},
        {"field": "model"}, {"field": "quant"}, {"field": "backend"},
        {"field": "value", "title": "value"},
        {"field": "cold_s", "title": "cold first-touch (s)"},
    ]
    y = {"field": "lane", "type": "nominal", "title": None,
         "sort": {"field": "rank", "op": "min", "order": "ascending"}}
    columns = []
    for i, (metric, title, fmt, subtitle) in enumerate(METRICS):
        labelled = i == 0  # band + lane labels once, on the left
        columns.append({
            "transform": [{"filter": f"datum.metric === '{metric}'"}, *_filters(controls)],
            "facet": {"row": {"field": "dev_class", "title": None, "sort": list(CLASSES),
                              "header": {"labels": labelled, "labelAngle": 0,
                                         "labelAlign": "left", "labelFontWeight": "bold",
                                         "labelPadding": 2, "labelLimit": 120}}},
            "spacing": 6,
            "resolve": {"scale": {"y": "independent"}},  # a band shows only its lanes
            "spec": {
                "width": 190, "height": {"step": 21},
                "layer": [
                    {"mark": {"type": "bar", "height": 9, "cornerRadiusEnd": 3},
                     "encoding": {
                         "y": {**y, "axis": {"labelLimit": 200, "labelFontSize": 11}
                               if labelled else None},
                         "x": {"field": "value", "type": "quantitative", "title": None},
                         "color": {"field": "dev_class", "type": "nominal", "legend": None,
                                   "scale": {"domain": list(CLASSES),
                                             "range": [CLASS_COLORS[c] for c in CLASSES]}},
                         "tooltip": tooltip,
                     }},
                    {"transform": [{"filter": "datum.value !== null"}],
                     "mark": {"type": "text", "align": "left", "dx": 4, "fontSize": 10},
                     "encoding": {"y": y, "x": {"field": "value", "type": "quantitative"},
                                  "text": {"field": "value", "format": fmt}}},
                    {"transform": [{"calculate": "datum.value * 1.16", "as": "headroom"}],
                     "mark": {"type": "point", "opacity": 0},
                     "encoding": {"y": y, "x": {"field": "headroom",
                                                "type": "quantitative"}}},
                    {"transform": [{"filter": "datum.note !== null"}],
                     "mark": {"type": "text", "align": "left", "dx": 4, "fontSize": 10,
                              "fontStyle": "italic", "opacity": 0.65},
                     "encoding": {"y": y, "x": {"datum": 0, "type": "quantitative"},
                                  "text": {"field": "note"}}},
                ],
            },
            "title": {"text": title, "anchor": "start", "fontSize": 12,
                      **({"subtitle": subtitle, "subtitleFontSize": 10} if subtitle else {})},
        })
    return {
        "$schema": VL_SCHEMA,
        "data": {"values": rows},
        "params": _params(controls),
        "hconcat": columns,
        "spacing": 26,
        "config": {"view": {"stroke": None}, "axis": {"grid": False}},
    }


def _single_depth(df: pd.DataFrame) -> pd.Series:
    """Which rows belong to a (lane, model) the sweep only reached once. Its budget
    stops at the first depth on a slow lane, so that lane has a measurement but no
    slope — one point drawn as a line looks like a broken chart, so it is drawn as
    a point and said out loud."""
    return df.groupby(["lane", "model"]).lane.transform("size") < 2


def _curve_spec(rows: list[dict], lanes: list[str], controls: list[dict], *,
                x: str, y: str, x_title: str, y_title: str,
                log_y: bool = False) -> dict:
    """A curve per lane, plus the lanes that have one depth instead of a curve —
    hollow marks, so a single measurement never masquerades as a trend."""
    encoding = {
        "x": {"field": x, "type": "quantitative", "title": x_title},
        "y": {"field": y, "type": "quantitative", "title": y_title,
              "scale": {"type": "log"} if log_y else {}},
        "color": {"field": "lane", "scale": _lane_scale(lanes, LANE_COLORS),
                  "legend": {"orient": "bottom", "columns": 2, "title": None}},
        "tooltip": [{"field": "lane"}, {"field": x}, {"field": y}],
    }
    return {
        "$schema": VL_SCHEMA,
        "data": {"values": rows},
        "params": _params(controls),
        "transform": _filters(controls),
        "width": 400, "height": 220,
        "layer": [
            {"transform": [{"filter": "!datum.single"}],
             "mark": {"type": "line", "point": {"size": 30}, "strokeWidth": 2},
             "encoding": encoding},
            {"transform": [{"filter": "datum.single"}],
             "mark": {"type": "point", "size": 70, "filled": False, "strokeWidth": 2},
             "encoding": encoding},
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
        default_model = (ok.groupby("model").lane.nunique()
                         .sort_values(ascending=False).index[0] if len(ok)
                         else sorted(df.model.dropna().unique())[0])
        grid_rows = _grid_rows(df)
        controls = _controls(grid_rows, default_model)
        context["controls"] = [c for c in controls if c["render"]]
        specs["grid"] = _grid_spec(grid_rows, controls)

        sweeps = _with_lanes(load_sweeps(published))
        pre = sweeps[(sweeps.kind == "prefill") & sweeps.ttft_ms.notna()].copy()
        dec = sweeps[(sweeps.kind == "decode") & sweeps.tps_p50.gt(0)].copy()
        keep = ["lane", "dev_class", "model", "quant", "backend", "single"]
        if len(pre):
            pre["single"] = _single_depth(pre)
            specs["curve-ttft"] = _curve_spec(
                pre[[*keep, "tokens", "ttft_ms"]].to_dict("records"), lanes, controls,
                x="tokens", y="ttft_ms",
                x_title="prompt tokens", y_title="time to first token (ms)")
        if len(dec):
            dec["single"] = _single_depth(dec)
            specs["curve-decode"] = _curve_spec(
                dec[[*keep, "kv_fill", "tps_p50"]].to_dict("records"), lanes, controls,
                x="kv_fill", y="tps_p50", log_y=True,
                x_title="context already used (tokens)", y_title="tok/s")
        context["curves"] = [sid for sid in ("curve-ttft", "curve-decode") if sid in specs]

        probes = _with_lanes(load_probes(published))
        ceilings = []
        for lane_label, g in probes[probes.status == "ok"].groupby("lane"):
            gemm = g[g.kind == "gemm"].tflops.max()
            d2d = g[g.kind == "d2d"].gbs.max()
            ceilings.append({"lane": lane_label,
                             "gemm": f"{gemm:.1f}" if gemm == gemm else "—",
                             "d2d": f"{d2d:.0f}" if d2d == d2d else "—"})
        context["ceilings"] = sorted(ceilings, key=lambda c: c["lane"])

        # The GPU column is what the lanes actually ran on: an iGPU is invisible to
        # the machine block's NVML/system_profiler probe, but its lane reports it.
        gpus = {m: ", ".join(sorted({d for d, f in zip(g.device, g.family, strict=True)
                                     if f != "cpu"}))
                for m, g in df.groupby("machine")}
        context["machines"] = [
            {"machine": r.machine, "cpu": r.cpu, "gpu": gpus.get(r.machine) or "—",
             "ram": f"{r.ram_gb:g} GB" if r.ram_gb == r.ram_gb else "?"}
            for r in df.drop_duplicates("machine").itertuples()
        ]
        bad = df[df.status != "ok"]
        context["unusable"] = [
            {"lane": r.lane, "machine": r.machine, "model": r.model,
             "quant": r.quant, "status": str(r.status).replace("_", " ")}
            for r in bad.itertuples()
        ]
        with_text = (ok[ok.sample_completion.notna()]
                     .groupby("model", observed=True).head(1))
        context["completions"] = [
            {"who": f"{r.machine} · {r.model} {r.quant} · {r.lane}",
             "text": str(r.sample_completion).strip()}
            for r in with_text.itertuples()
        ]
        context["stats"] = [
            {"v": f"{df.machine.nunique()}", "k": "machines"},
            {"v": f"{df.model.nunique()}", "k": "models"},
            {"v": f"{ok.lane.nunique()}", "k": "lanes"},
        ]

    env = Environment(loader=PackageLoader("bench_analysis"),
                      autoescape=select_autoescape(default=True))
    context["css"] = (PKG / "assets" / "report.css").read_text()
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
