"""Build the report: one self-contained, data-first HTML page.

One faceted grid answers "how fast is on-device inference on real machines":
decode tok/s, prefill tok/s, and init time per (machine, lane), faceted by
model, filterable by backend / quant / lane family. Sweep cost curves sit in
per-model expanders; machines and sample completions in appendix tables. The
page opens with the one-command install lines — every reader is a potential
contributor.

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

VEGA_LIBS = (
    ("vega", alt.VEGA_VERSION),
    ("vega-lite", alt.VEGALITE_VERSION),
    ("vega-embed", alt.VEGAEMBED_VERSION),
)

METRICS = (  # column key, title, value-label format (vega d3-format)
    ("decode", "generation (tok/s)", ".0f"),
    ("prefill", "prompt reading (tok/s)", ".0f"),
    ("init", "init, warm (s)", ".1f"),
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


_PARENS = re.compile(r"\(.*?\)")
_NOISE = re.compile(
    r"\b(NVIDIA|AMD|Intel|GeForce|Graphics|Processor|CPU|\d+-Core)\b", re.IGNORECASE
)


def _chip(device: str) -> str:
    """A short silicon label from the exe-reported device string:
    'AMD Ryzen 9 9950X 16-Core Processor' → 'Ryzen 9 9950X'."""
    short = _NOISE.sub(" ", _PARENS.sub(" ", device or ""))
    return " ".join(short.split()) or (device or "?")


def _family(provider: str) -> str:
    return provider.split(":")[0]


def _with_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """Add `family` and the display `lane` ('RTX 5080 · vulkan'). Two devices
    of one family with the same chip name keep their lane index to stay
    distinct rows, never silently pooled."""
    df = df.copy()
    df["family"] = [_family(p) for p in df.provider]
    df["lane"] = [f"{_chip(d)} · {f}" for d, f in zip(df.device, df.family, strict=True)]
    dupes = df.groupby("lane").provider.transform("nunique") > 1
    df.loc[dupes, "lane"] = [
        f"{_chip(d)} · {p}" for d, p in zip(df.device[dupes], df.provider[dupes], strict=True)
    ]
    return df


def _grid_rows(ok: pd.DataFrame) -> list[dict]:
    """Long-format rows for the results grid: one per (lane, model, metric)."""
    rows = []
    for r in ok.itertuples():
        init_s = (r.model_load_ms_p50 + r.context_init_ms_p50) / 1e3
        cold = getattr(r, "cold_start_ms_p50", None)
        base = {
            "lane": r.lane, "family": r.family, "machine": r.machine,
            "backend": r.backend, "model": r.model, "quant": r.quant,
            "device": r.device,
            "cold_s": round(cold / 1e3, 1) if cold == cold and cold else None,
        }
        for metric, value in (("decode", r.decode_tps_p50),
                              ("prefill", r.prefill_tps_p50),
                              ("init", init_s)):
            if value == value and value is not None:  # NaN-safe
                rows.append({**base, "metric": metric, "value": round(float(value), 2)})
    return rows


def _lane_scale(lanes: list[str], colors: list[str]) -> dict:
    return {"domain": lanes,
            "range": [colors[i] if i < len(colors) else LANE_OVERFLOW
                      for i in range(len(lanes))]}


def _filter_params(rows: list[dict]) -> list[dict]:
    def options(key):
        return ["all", *sorted({r[key] for r in rows})]

    return [
        {"name": f"f_{key}", "value": "all",
         "bind": {"input": "select", "options": options(key), "name": f"{label} "}}
        for key, label in (("backend", "backend"), ("quant", "quant"),
                           ("family", "lane"))
    ]


def _grid_spec(rows: list[dict], lanes: list[str]) -> dict:
    """One hconcat: per metric, a row-faceted (by model) horizontal bar chart.
    Shared filter params; per-metric shared x so models compare within a
    column; value labels on every bar (the palette's contrast relief)."""
    filters = [{"filter": f"f_{k} == 'all' || datum.{k} == f_{k}"}
               for k in ("backend", "quant", "family")]
    tooltip = [
        {"field": "lane", "title": "lane"},
        {"field": "machine", "title": "machine"},
        {"field": "device", "title": "device"},
        {"field": "model"}, {"field": "quant"}, {"field": "backend"},
        {"field": "value", "title": "value"},
        {"field": "cold_s", "title": "cold first-touch (s)"},
    ]
    columns = []
    for metric, title, fmt in METRICS:
        columns.append({
            "transform": [{"filter": f"datum.metric == '{metric}'"}, *filters],
            "facet": {"row": {"field": "model", "title": None,
                              "header": {"labelAngle": 0, "labelAlign": "left",
                                         "labelFontWeight": "bold"}}},
            "spec": {
                "width": 220, "height": {"step": 22},
                "layer": [
                    {"mark": {"type": "bar", "height": 10, "cornerRadiusEnd": 4},
                     "encoding": {
                         "y": {"field": "lane", "type": "nominal", "sort": lanes,
                               "title": None},
                         "x": {"field": "value", "type": "quantitative",
                               "title": None},
                         "color": {"field": "lane", "scale": _lane_scale(
                             lanes, LANE_COLORS), "legend": None},
                         "tooltip": tooltip,
                     }},
                    {"mark": {"type": "text", "align": "left", "dx": 4,
                              "fontSize": 10},
                     "encoding": {
                         "y": {"field": "lane", "type": "nominal", "sort": lanes},
                         "x": {"field": "value", "type": "quantitative"},
                         "text": {"field": "value", "format": fmt},
                     }},
                ],
            },
            "title": {"text": title, "anchor": "start", "fontSize": 12},
        })
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": rows},
        "params": _filter_params(rows),
        "hconcat": columns,
        "resolve": {"scale": {"color": "shared"}},
        "config": {"view": {"stroke": None}, "axis": {"grid": False}},
    }


def _curve_spec(rows: pd.DataFrame, lanes: list[str], *, x: str, y: str,
                x_title: str, y_title: str, log: bool = False) -> dict:
    scale = {"type": "log"} if log else {}
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": rows.to_dict("records")},
        "width": 420, "height": 220,
        "mark": {"type": "line", "point": {"size": 30}, "strokeWidth": 2},
        "encoding": {
            "x": {"field": x, "type": "quantitative", "title": x_title, "scale": scale},
            "y": {"field": y, "type": "quantitative", "title": y_title, "scale": scale},
            "color": {"field": "lane", "scale": _lane_scale(lanes, LANE_COLORS),
                      "legend": {"orient": "bottom", "columns": 2, "title": None}},
            "tooltip": [{"field": "lane"}, {"field": x}, {"field": y}],
        },
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
        ok = df[df.status == "ok"].copy()
        lanes = sorted(ok.lane.unique(), key=lambda s: (s.rsplit(" · ")[-1], s))
        grid_rows = _grid_rows(ok)
        specs["grid"] = _grid_spec(grid_rows, lanes)

        sweeps = _with_lanes(load_sweeps(published))
        context["curve_models"] = []
        for model in sorted(sweeps.model.dropna().unique()):
            g = sweeps[sweeps.model == model]
            pre = g[(g.kind == "prefill") & g.ttft_ms.notna()]
            dec = g[(g.kind == "decode") & g.tps_p50.gt(0)].copy()
            ids = []
            if len(pre):
                sid = f"pre-{model}"
                specs[sid] = _curve_spec(
                    pre[["lane", "tokens", "ttft_ms"]], lanes,
                    x="tokens", y="ttft_ms",
                    x_title="prompt tokens", y_title="time to first token (ms)")
                ids.append(sid)
            if len(dec):
                dec["kv_fill"] = dec.kv_fill.clip(lower=64)
                sid = f"dec-{model}"
                specs[sid] = _curve_spec(
                    dec[["lane", "kv_fill", "tps_p50"]], lanes,
                    x="kv_fill", y="tps_p50", log=True,
                    x_title="context already used (tokens)", y_title="tok/s")
                ids.append(sid)
            if ids:
                context["curve_models"].append({"model": model, "specs": ids})

        probes = _with_lanes(load_probes(published).rename(
            columns={"machine": "submission"}).assign(machine=lambda d: d.submission))
        ceilings = []
        for lane_label, g in probes[probes.status == "ok"].groupby("lane"):
            gemm = g[g.kind == "gemm"].tflops.max()
            d2d = g[g.kind == "d2d"].gbs.max()
            ceilings.append({"lane": lane_label,
                             "gemm": f"{gemm:.1f}" if gemm == gemm else "—",
                             "d2d": f"{d2d:.0f}" if d2d == d2d else "—"})
        context["ceilings"] = sorted(ceilings, key=lambda c: c["lane"])

        context["machines"] = [
            {"machine": r.machine, "cpu": r.cpu, "gpu": r.gpu,
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
            {"v": f"{df[df.status == 'ok'].lane.nunique()}", "k": "lanes"},
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
