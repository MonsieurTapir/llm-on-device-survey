"""The site builder renders one self-contained, data-first page — and hostile
submission strings can never reach an executable position: HTML context is
autoescaped, spec islands go through `tojson` (`<` → \\u003c), so a payload
that tries to terminate a script element survives only as inert text."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from bench_analysis import (
    load_memory,
    load_results,
    load_sweeps,
    load_thread_scaling,
    site,
)

FIXTURES = Path(__file__).parent / "fixtures"
PAYLOAD = "</script><script>alert(1)</script>"


def _build(tmp_path, published=FIXTURES):
    cache = tmp_path / "vega"
    cache.mkdir(exist_ok=True)
    for name, version in site.VEGA_LIBS:  # stubs: the test never fetches
        (cache / f"{name}@{version}.min.js").write_text("/* stub */")
    out = tmp_path / "report.html"
    site.build(published, out, vega_cache=cache)
    return out.read_text()


def test_build_renders_grid_curves_and_strict_json(tmp_path):
    h = _build(tmp_path)
    assert "{{" not in h  # no unrendered template
    for anchor in (
        'data-spec="grid"',
        "Cost curves",
        "Machines",
        'data-spec="launch"',
        "First launch",
        'data-spec="memory-footprint"',
        'data-spec="memory-job"',
        "cmd-bash",
        "cmd-ps",
        'button class="copy"',
    ):
        assert anchor in h
    islands = re.findall(r'<script type="application/json"[^>]*>(.*?)</script>', h, re.S)
    assert islands, "no spec islands rendered"
    for body in islands:
        json.loads(body)  # strict — raises on NaN/Infinity


def _color_encodings(node) -> list[dict]:
    """Every scaled `color` encoding anywhere in a vega-lite spec."""
    if isinstance(node, list):
        return [c for item in node for c in _color_encodings(item)]
    if not isinstance(node, dict):
        return []
    found = [c for value in node.values() for c in _color_encodings(value)]
    color = node.get("color")
    if isinstance(color, dict) and "scale" in color:
        found.append(color)
    return found


def test_one_hue_per_lane_across_every_chart(tmp_path):
    """Hue means lane, and only lane: every chart on the page colors on `lane`
    through the same domain→range mapping, so a bar and its curve match."""
    h = _build(tmp_path)
    islands = {
        m.group(1): json.loads(m.group(2))
        for m in re.finditer(
            r'<script type="application/json" data-island="([^"]+)">(.*?)</script>', h, re.S
        )
    }
    assert {"grid", "launch", "curve-ttft", "curve-decode"} <= islands.keys()

    scales = []
    for name, spec in islands.items():
        encodings = _color_encodings(spec)
        assert encodings, f"{name} colors on nothing"
        for color in encodings:
            assert color["field"] == "lane", f"{name} colors on {color['field']}"
            scales.append(color["scale"])

    assert all(scale == scales[0] for scale in scales)
    assert len(scales[0]["domain"]) == len(set(scales[0]["domain"]))


def _tooltips(node) -> list[list[dict]]:
    """Every tooltip encoding anywhere in a vega-lite spec."""
    if isinstance(node, list):
        return [t for item in node for t in _tooltips(item)]
    if not isinstance(node, dict):
        return []
    found = [t for value in node.values() for t in _tooltips(value)]
    tip = node.get("tooltip")
    if isinstance(tip, list):
        found.append(tip)
    return found


# What is already on the screen when a tooltip opens: the lane is the y-axis label or
# the legend entry, and the rest is what the control row is currently scoped to.
IDENTITY = {"lane", "machine", "device", "dev_class", "model", "quant", "backend"}


def test_no_tooltip_repeats_what_the_page_already_shows(tmp_path):
    """A tooltip carries only what its mark cannot: numbers, and why a number is
    marked or missing. Identity is never among them — printing the lane, the machine
    and the model on hover buries the one line the reader came for."""
    h = _build(tmp_path)
    islands = {
        m.group(1): json.loads(m.group(2))
        for m in re.finditer(
            r'<script type="application/json" data-island="([^"]+)">(.*?)</script>', h, re.S
        )
    }
    assert len(islands) >= 6
    for name, spec in islands.items():
        tips = _tooltips(spec)
        assert tips, f"{name} has no tooltip at all"
        for tip in tips:
            fields = {entry["field"] for entry in tip}
            assert not fields & IDENTITY, f"{name} hover repeats {fields & IDENTITY}"
            # The longest list on the page is the measured grid's range: each end of
            # the sweep with its depth, and the job's dot with its own.
            assert len(fields) <= 6, f"{name} hover carries {len(fields)} fields"


def _grid_columns(html: str) -> list[dict]:
    island = re.search(
        r'<script type="application/json" data-island="grid">(.*?)</script>', html, re.S
    )
    assert island, "no grid island rendered"
    return json.loads(island.group(1))["hconcat"]


def _anchor_layer(column: dict) -> dict:
    """The one layer of a grid column that carries the x-axis object. Vega-lite
    merges the axes of layers sharing a scale, so a second one breaks the heading."""
    carriers = [layer for layer in column["spec"]["layer"] if "axis" in layer["encoding"]["x"]]
    assert len(carriers) == 1, "exactly one layer may define the metric axis"
    return carriers[0]


def test_metric_heading_rides_the_x_axis_over_the_bars(tmp_path):
    """The heading is the shared x axis's title, drawn from the top over the plot
    area — never a concat title, which would span the lane-label gutter as well. The
    two rate columns add the depth their dot was measured at, read off the rows: the
    section below prices the same two quantities, so neither heading may float free
    of the depth it belongs to."""
    columns = _grid_columns(_build(tmp_path))
    assert len(columns) == len(site.METRICS)
    for column, (metric, title, _, _) in zip(columns, site.METRICS, strict=True):
        assert "title" not in column
        anchor = _anchor_layer(column)
        # Whichever mark anchors the scale carries the axis: the interval's rule on
        # a range metric, the bar where there is still a bar.
        assert anchor["mark"]["type"] == ("rule" if metric in site.RANGE_METRICS else "bar")
        assert anchor is column["spec"]["layer"][0]
        axis = anchor["encoding"]["x"]["axis"]
        assert axis["orient"] == "top"
        assert axis["titleAnchor"] == "start"
        text = axis["title"]
        assert (text[0] if isinstance(text, list) else text) == title
        if metric in site.RANGE_METRICS:
            # the fixtures' jobs sit at ~1,555 tokens, to the nearest half-thousand
            assert text[1] == "dot: the ~1.5k-token job"
        else:
            assert text[1] == "lower is better"


def test_a_shelf_whose_jobs_measured_no_depth_drops_the_depth_scope():
    """The depth in the heading is derived, not assumed: with no job to read a prompt
    depth off, the two rate columns say nothing about depth rather than a made-up
    number, and the heading stays a bare title."""
    assert site._job_depth([{"job_at": None}, {}]) is None
    assert site._job_depth([{"job_at": 1557}, {"job_at": 1553}]) == "~1.5k"
    assert site._job_depth([{"job_at": 4096}]) == "~4k"

    rows = [
        {
            "lane": "l",
            "dev_class": "CPU",
            "metric": m,
            "value": 1.0,
            "note": None,
            "rank": 0,
            "model": "m",
            "quant": "q",
            "backend": "b",
            "machine": "x",
            "device": "d",
            "cold_s": None,
            "job_at": None,
        }
        for m, *_ in site.METRICS
    ]
    columns = site._grid_spec(rows, [], ["l"])["hconcat"]
    decode = columns[0]["spec"]["layer"][0]["encoding"]["x"]["axis"]
    assert decode["title"] == "generation (tok/s)"  # no subtitle line at all


def test_reading_anchors_are_at_most_three_ascending_gridlines(tmp_path):
    """Reference rules are gridlines at named reading speeds — a short, ordered set on
    the tok/s columns, and nothing at all on warm init. Every anchor is drawn; the
    ones that have room say what they are on the line itself."""
    html = _build(tmp_path)
    columns = {
        metric: column
        for (metric, *_), column in zip(site.METRICS, _grid_columns(html), strict=True)
    }
    for metric, column in columns.items():
        axis = _anchor_layer(column)["encoding"]["x"]["axis"]
        values = axis.get("values")
        if metric == "init":
            assert values is None and axis["grid"] is False
            assert "labelExpr" not in axis  # plain numeric ticks, as everywhere else
            continue
        assert axis["grid"] is True
        assert 0 < len(values) <= 3
        assert values == sorted(values)
        assert values == [v for v, _ in site.READING_ANCHORS[metric]]
        # the first anchor of every column always fits, so a reader always has a
        # named reference on the chart itself
        first, name = site.READING_ANCHORS[metric][0]
        assert f"datum.value === {first} ? '{first:,} · {name}'" in axis["labelExpr"]
    # ...and the copy above the charts is the full key, for the lines that had no room
    copy = html.split("<h2>Results</h2>")[1].split("</section>")[0]
    for value, _ in {a for anchors in site.READING_ANCHORS.values() for a in anchors}:
        assert f"({value:,}" in copy, value


def test_an_anchor_name_is_dropped_rather_than_printed_over_its_neighbour():
    """Vega culls nothing on an axis with explicit `values`, so which anchors can wear
    their names is arithmetic about pixels: wide-open columns crowd the low anchors
    together, and a name that will not fit falls back to the bare rate, then to
    nothing. A narrow column has room for every name it can still draw."""
    pre = site.READING_ANCHORS["prefill"]
    # 1,040 tok/s of ink — this shelf's fastest prompt lane: 130 and 400 land 39
    # pixels apart, so only the first is named and 1,700 is off the scale entirely.
    wide = site._anchor_labels(pre, 1040 * 1.16 * 1.1)
    assert wide == [(130, "130 · paragraph/s"), (400, "")]
    # a CPU-only selection: one line, named
    assert site._anchor_labels(pre, 185 * 1.16 * 1.1) == [(130, "130 · paragraph/s")]
    # no domain to fit against (the task grid computes its numbers in the page): the
    # two anchors there are 26× apart, so both are named
    assert site._anchor_labels(site.TASK_ANCHORS["tps"], None) == [
        (5, "5 · silent reading"),
        (130, "130 · paragraph/s"),
    ]


def test_the_fitted_span_is_the_widest_a_column_can_get():
    """Names are fitted against the unfiltered ink, because every control-row state
    only removes rows — so a name that fits at build time fits in all of them."""
    rows = [
        {"metric": "prefill", "label_x": 1040.0, "value": 900.0},
        {"metric": "prefill", "label_x": 185.0, "value": 185.0},
        {"metric": "decode", "label_x": None, "value": None},
        {"metric": "init", "value": 1.4},
    ]
    spans = site._metric_spans(rows)
    assert spans["prefill"] == pytest.approx(1040 * 1.16 * 1.1)
    assert spans["init"] == pytest.approx(1.4 * 1.16 * 1.1)
    assert "decode" not in spans  # nothing measured, nothing to fit against


def _fixture_grid_rows() -> dict[tuple[str, str, str], dict]:
    """The fixture shelf's grid rows, keyed (lane, model, metric)."""
    df = site._with_lanes(load_results(FIXTURES))
    ranges = site._depth_ranges(site._with_lanes(load_sweeps(FIXTURES)))
    return {(r["lane"], r["model"], r["metric"]): r for r in site._grid_rows(df, ranges)}


def test_range_metrics_carry_the_sweep_interval_around_the_job_dot():
    """A range column draws the sweep's endpoints over depth with the job's own
    number inside them: the m1-max ladder measured two fills (80 tok/s empty, 75 at
    2,048) and the job scored 40, so the interval has to reach down to the dot."""
    rows = _fixture_grid_rows()
    decode = rows[("Apple M1 Max · mtl", "gemma4-E2B", "decode")]
    assert (decode["d_shallow"], decode["v_shallow"]) == (0, 80.0)
    assert (decode["d_deep"], decode["v_deep"]) == (2048, 75.0)
    assert decode["n_depths"] == 2
    assert decode["value"] == 40.0  # the job's point, untouched
    assert (decode["lo"], decode["hi"]) == (40.0, 80.0)  # union, so the dot is on it
    # the label rides the dot (`value`); `label_x` is only the row's ink edge, which
    # is what the invisible headroom mark reserves room past
    assert decode["label_x"] == 80.0 and "label_v" not in decode

    # 18 prefill chunks fan out to 9 cumulative depths; the rate falls with depth.
    prefill = rows[("Apple M1 Max · mtl", "gemma4-E2B", "prefill")]
    assert prefill["n_depths"] == 9
    assert prefill["d_shallow"] == 512 and prefill["d_deep"] == 4096
    assert prefill["v_shallow"] > prefill["v_deep"]
    assert (prefill["lo"], prefill["hi"]) == (prefill["v_deep"], prefill["v_shallow"])
    # That run's job measured no prompt rate at all: the sweep's interval is still
    # drawn, and nothing is labelled — the number a range column prints is the job's,
    # and it prints it on the job's dot or not at all.
    assert prefill["value"] is None

    # The init column has no depth story and no interval fields at all.
    init = rows[("Apple M1 Max · mtl", "gemma4-E2B", "init")]
    assert not {"lo", "hi", "v_deep", "label_x"} & init.keys()


def test_a_status_note_is_told_once_per_lane_not_once_per_column():
    """ "too slow" is a fact about the (lane, model), not about a column: the first
    column that would have shown a number carries it and the other two stay blank,
    so the row reads as one sentence instead of three."""
    rows = _fixture_grid_rows()
    notes = {
        metric: rows[("RTX 3090 · cuda", "gemma4-E4B", metric)]["note"]
        for metric, *_ in site.METRICS
    }
    assert notes == {"decode": "too slow", "prefill": None, "init": None}
    # ...and a cell that measured everything carries no note in any column
    assert not any(
        rows[("RTX 3090 · cuda", "qwen3-4B", metric)]["note"] for metric, *_ in site.METRICS
    )


def test_one_measured_depth_collapses_and_keeps_its_status_note():
    """The 3090's gemma4-E4B was killed by the backstop after a single prefill
    chunk: one depth is one point, not a fabricated span, and the cell still says
    why it has no job number."""
    rows = _fixture_grid_rows()
    prefill = rows[("RTX 3090 · cuda", "gemma4-E4B", "prefill")]
    assert prefill["n_depths"] == 1
    assert prefill["d_shallow"] == prefill["d_deep"] == 512
    assert prefill["v_shallow"] == prefill["v_deep"] == prefill["lo"] == prefill["hi"]
    assert prefill["value"] is None  # no job number to put a dot on, and so no label
    assert prefill["label_x"] == prefill["v_deep"]  # the interval is still the ink

    # Same cell, decode: the sweep never got there, so there is no interval either —
    # and this is the column that tells the reader why (once, for the whole row).
    decode = rows[("RTX 3090 · cuda", "gemma4-E4B", "decode")]
    assert decode["note"] == "too slow"
    for field in ("lo", "hi", "v_shallow", "v_deep", "n_depths", "label_x"):
        assert decode[field] is None, field  # None, never NaN — the island is strict


def test_a_cell_with_no_sweep_at_all_keeps_only_its_note():
    """The unhealthy (model, provider) never ran: no interval, no dot, no label."""
    row = _fixture_grid_rows()[("Apple M1 Max · mtl", "qwen3-4B", "decode")]
    assert row["note"] == "unhealthy"
    assert row["value"] is None and row["lo"] is None and row["label_x"] is None


def test_the_job_dot_sits_at_a_prompt_depth_read_off_its_own_numbers():
    """`job_at` is derived — rate × time to first token — so the tooltip can say
    which depth the dot belongs to without the report hardcoding the task's prompt.
    A lane whose sweep only reached one deep fill still reads as one interval."""
    df = pd.DataFrame(
        [
            {
                "provider": "cpu:0",
                "machine": "box",
                "backend": "llamacpp",
                "device": "AMD Ryzen 7 255 w/ Radeon 780M Graphics",
                "cpu": "AMD Ryzen 7 255 w/ Radeon 780M Graphics",
                "threads_batch": 8,
                "threads_decode": 8,
                "model": "Ministral3-3B",
                "quant": "q4",
                "status": "ok",
                "decode_tps_p50": 20.19,
                "prefill_tps_p50": 160.39,
                "ttft_ms_p50": 9695.08,
                "model_load_ms_p50": 500.0,
                "context_init_ms_p50": 100.0,
            }
        ]
    )
    ranges = {
        (("box", "llamacpp", "cpu:0", "Ministral3-3B", "q4"), "decode"): {
            "d_shallow": 7168,
            "d_deep": 7168,
            "v_shallow": 8.9,
            "v_deep": 8.9,
            "n_depths": 1,
        }
    }
    rows = {r["metric"]: r for r in site._grid_rows(site._with_lanes(df), ranges)}
    assert rows["decode"]["job_at"] == 1555  # ≈ the summarize-large prompt
    assert (rows["decode"]["lo"], rows["decode"]["hi"]) == (8.9, 20.19)
    assert rows["decode"]["value"] == 20.19  # what the label prints, where the dot is
    assert rows["prefill"]["job_at"] == 1555
    assert rows["prefill"]["lo"] is None  # no prefill sweep points for this cell


def test_range_columns_draw_a_rule_a_dot_and_one_label(tmp_path):
    """Per range column: the interval, the job's filled dot, one value label riding
    that dot, and a zero-based x scale (a rule, unlike a bar, does not bring its own
    baseline)."""
    columns = {
        metric: column
        for (metric, *_), column in zip(site.METRICS, _grid_columns(_build(tmp_path)), strict=True)
    }
    for metric, column in columns.items():
        marks = [layer["mark"]["type"] for layer in column["spec"]["layer"]]
        if metric not in site.RANGE_METRICS:
            assert marks[0] == "bar"
            continue
        assert marks == ["rule", "point", "text", "point", "text"]
        rule, dot, label = column["spec"]["layer"][:3]
        assert rule["encoding"]["x"]["field"] == "lo"
        assert rule["encoding"]["x2"]["field"] == "hi"
        assert rule["encoding"]["x"]["scale"] == {"zero": True}
        assert dot["mark"]["filled"] is True
        assert dot["encoding"]["x"]["field"] == "value"  # the job's own number
        # The label rides the mark it describes: same field as the dot, cleared by
        # dx so the text never prints over the dot's rim, and drawn only where the
        # job produced a number at all.
        assert label["encoding"]["x"]["field"] == "value"
        assert label["encoding"]["text"]["field"] == "value"
        assert label["mark"]["dx"] == 9
        assert label["transform"] == [{"filter": "datum.value !== null"}]
        # the headroom mark reserves room past the row's ink, which is a different
        # thing from where the label sits: the interval can reach past the dot
        assert "datum.label_x" in column["spec"]["layer"][3]["transform"][0]["calculate"]
        # Hover describes the ink and nothing else: the sweep's points as one
        # prebuilt string (a one-point sweep reads as one point, never a "17–17"
        # range), and the job's value at its own prompt depth as another.
        assert [t["field"] for t in rule["encoding"]["tooltip"]] == ["sweep_str", "job_str"]


def _island(html: str, name: str) -> dict:
    found = re.search(
        f'<script type="application/json" data-island="{name}">(.*?)</script>', html, re.S
    )
    assert found, f"no {name} island rendered"
    return json.loads(found.group(1))


def _fixture_launch_rows() -> dict[tuple[str, str, str], dict]:
    """The fixture shelf's first-launch rows, keyed (lane, model, phase)."""
    rows = site._launch_rows(site._with_lanes(load_results(FIXTURES)))
    return {(r["lane"], r["model"], r["phase"]): r for r in rows}


def test_compilation_is_claimed_only_where_it_was_measured_from_empty():
    """The 3090's cuda lane compiled 5.5 MB of pipelines into a cache the harness
    pinned, so its warmup span is a real from-scratch compile and the number stands
    unqualified. The M1's Metal lane exposes no cache path at all: that launch still
    cost what it cost, so the span is reported and marked — what we don't know is
    how much the machine had already built, which qualifies the number rather than
    licensing the claim that nothing compiled."""
    rows = _fixture_launch_rows()

    cuda = rows[("RTX 3090 · cuda", "qwen3-4B", "pipeline compilation")]
    assert cuda["seconds"] == 1.56 and cuda["note"] is None
    assert cuda["label"] == "1.6"
    # The estimate carries its own arithmetic: a 1.83 s warm pass, 0.27 s of which
    # its width walk spends on plain prefill, priced from this lane's own fit.
    assert (cuda["span"], cuda["netted"]) == (1.83, 0.27)
    assert cuda["mb"] == 5.5 and cuda["cache"] == "redirected"
    # How much was compiled, out of which cache, and what was netted out belong to
    # the compile alone: a cold read of the weights compiles nothing, so it carries
    # none of them.
    cold = rows[("RTX 3090 · cuda", "qwen3-4B", "cold first touch")]
    assert cold["seconds"] == 5.0
    assert cold["mb"] is None and cold["cache"] is None
    assert cold["span"] is None and cold["netted"] is None

    mac = rows[("Apple M1 Max · mtl", "gemma4-E2B", "pipeline compilation")]
    assert mac["seconds"] == 0.61 and mac["note"] is None
    assert mac["label"] == f"0.6{site.UNVERIFIED_MARK}"
    assert mac["cache"] == "unavailable"
    assert mac["mb"] is None  # nothing to report a size for either
    # the lane stays on the chart because its cold first touch is a number
    assert rows[("Apple M1 Max · mtl", "gemma4-E2B", "cold first touch")]["seconds"] == 5.0

    # No compile, no cold load: the cell would be two notes and no ink, so it is
    # gone. E4B's sweep reached one chunk, so it has no fit — and with no fit there
    # is no way to net the width walk out of its 0.24 s warm pass, which leaves the
    # compile unestimable rather than equal to the span.
    assert not [key for key in rows if key[1] == "gemma4-E4B" and "cold" in key[2]]
    assert ("RTX 3090 · cuda", "gemma4-E4B", "pipeline compilation") not in rows
    assert ("RTX 3090 · cuda", "gemma4-E2B", "pipeline compilation") not in rows
    assert ("Apple M1 Max · mtl", "qwen3-4B", "pipeline compilation") not in rows
    json.dumps(list(rows.values()), allow_nan=False)  # strict — the island is


def test_cpu_lanes_report_no_compilation_and_leave_the_chart():
    """A CPU lane's non-zero `shader_bytes` is the GPU backend's registry init, not
    this run's compilation, so it never claims a compile bar — and with the cold load
    attributed to whichever lane read the file first, a CPU lane usually has nothing
    to draw at all."""
    df = pd.DataFrame(
        [
            {
                "provider": "cpu:0",
                "device": "AMD Ryzen 7 255 w/ Radeon 780M Graphics",
                "shader_bytes": 2621456,
                "cold_start_ms_p50": None,
            },
            {
                "provider": "vulkan:0",
                "device": "AMD Radeon Graphics (RADV PHOENIX)",
                "shader_bytes": 2711058,
                "cold_start_ms_p50": 526.48,
            },
        ]
    ).assign(
        machine="box",
        backend="llamacpp",
        model="Ministral3-3B",
        quant="q4",
        cpu="AMD Ryzen 7 255 w/ Radeon 780M Graphics",
        threads_batch=8,
        threads_decode=8,
        status="ok",
        decode_tps_p50=20.0,
        shader_cache="redirected",
        shader_warmup_ms=2990.76,
        fit_width=512,
        fit_intercept_ms=100.0,
        fit_slope_ms_per_1k=0.0,
    )
    rows = site._launch_rows(site._with_lanes(df))
    assert {(r["lane"], r["phase"]) for r in rows} == {
        ("Ryzen 7 255 iGPU · vulkan", "pipeline compilation"),
        ("Ryzen 7 255 iGPU · vulkan", "cold first touch"),
    }
    # 2.99 s of warm pass less the 0.26 s its 512+512+256+32-token walk costs at a
    # flat 100 ms per full chunk.
    assert [r["seconds"] for r in rows] == [2.73, 0.53]


def test_a_lane_with_no_fit_cannot_estimate_its_compile():
    """The warm pass is compilation and prefill together, and only the lane's own
    cost function can say how much of it is prefill. Without one the compile is
    unestimable — which is not the same as the span, and not the same as zero."""
    df = pd.DataFrame(
        [
            {
                "provider": "vulkan:0",
                "device": "AMD Radeon Graphics (RADV PHOENIX)",
                "shader_bytes": 2711058,
                "cold_start_ms_p50": 526.48,
                "fit_width": None,
                "fit_intercept_ms": None,
                "fit_slope_ms_per_1k": None,
            },
        ]
    ).assign(
        machine="box",
        backend="llamacpp",
        model="Ministral3-3B",
        quant="q4",
        cpu="AMD Ryzen 7 255 w/ Radeon 780M Graphics",
        threads_batch=8,
        threads_decode=8,
        status="ok",
        decode_tps_p50=20.0,
        shader_cache="redirected",
        shader_warmup_ms=2990.76,
    )
    (compile_row,) = [
        r for r in site._launch_rows(site._with_lanes(df)) if r["phase"] == "pipeline compilation"
    ]
    assert compile_row["seconds"] is None
    assert compile_row["note"] == "no cost function to net the warm pass out of"
    # The cold first touch keeps the cell on the chart, so the note is seen.
    assert compile_row["span"] == 2.99 and compile_row["netted"] is None


def test_launch_chart_groups_two_phases_on_one_lane_row(tmp_path):
    """Two bars per lane, offset within the lane's row, the phase carried by opacity
    with its legend as the key — and the compilation caveat in the subtitle, where a
    reader meets it before the numbers."""
    spec = _island(_build(tmp_path), "launch")
    assert spec["title"]["text"] == "one-time first launch (s)"
    subtitle = " ".join(spec["title"]["subtitle"])  # split in two so it cannot clip
    assert "empty shader cache" in subtitle

    compile_bar, cold_bar, label, headroom, note = spec["layer"]
    bar = compile_bar
    for phase, layer in zip(site.LAUNCH_PHASES, (compile_bar, cold_bar), strict=True):
        assert layer["mark"]["type"] == "bar"
        assert layer["transform"] == [
            {"filter": f"datum.seconds !== null && datum.phase === '{phase}'"}
        ]
    # Only the compile layer may define the axis — two layers on one scale would
    # merge their axis objects into a doubled heading.
    assert "axis" in compile_bar["encoding"]["x"]
    assert "axis" not in cold_bar["encoding"]["x"]
    # A cold read of the weights compiled nothing, so its tooltip says nothing about
    # compilation: no "compiled (MB)", no shader cache, and no dash standing in. And
    # neither phase repeats the lane, which is the row's own axis label.
    cold_fields = {t["field"] for t in cold_bar["encoding"]["tooltip"]}
    assert cold_fields == {"phase", "seconds"}
    assert {t["field"] for t in compile_bar["encoding"]["tooltip"]} == (
        cold_fields | {"mb", "cache", "span", "netted"}
    )
    assert bar["encoding"]["x"]["field"] == "seconds"
    assert bar["encoding"]["x"]["scale"] == {"zero": True}
    assert bar["encoding"]["y"]["sort"] == {"field": "rank", "op": "min", "order": "ascending"}
    # `opacity`, not `fillOpacity`: vega-lite 6.4.1 compiles no legend for the
    # latter, and this legend is the only thing naming the two phases.
    assert "fillOpacity" not in bar["encoding"]
    opacity = bar["encoding"]["opacity"]
    assert opacity["field"] == "phase"
    assert opacity["scale"] == {"domain": list(site.LAUNCH_PHASES), "range": [1, 0.5]}
    assert opacity["legend"]["orient"] == "bottom"
    for layer in (bar, label, note):  # every mark sits in its own phase row
        assert layer["encoding"]["yOffset"]["field"] == "phase"
    # The row formats its own label: only it knows whether the span earned the mark
    # that says its shader cache could not be pinned.
    assert label["encoding"]["text"] == {"field": "label"}
    assert site.UNVERIFIED_MARK in subtitle
    assert "datum.seconds" in headroom["transform"][0]["calculate"]
    assert note["transform"] == [{"filter": "datum.note !== null"}]
    assert note["mark"]["fontStyle"] == "italic"
    # the note sits at the axis, where a missing bar leaves room for it
    assert note["encoding"]["x"] == {"datum": 0, "type": "quantitative"}


def _fixture_pack() -> dict:
    df = site._with_lanes(load_results(FIXTURES))
    return site._task_pack(df, site._with_lanes(load_sweeps(FIXTURES)))


def test_task_pack_carries_the_cost_function_its_ladder_and_its_envelope():
    """The page prices a task from this pack alone, so it has to hold the whole cost
    function (both parameters, the dispatch width they belong to, and the fit
    quality), the ladder as measured pairs, how far each phase was measured, what a
    launch costs once, and the job's own numbers to check the arithmetic against."""
    pack = _fixture_pack()
    records = {(r["lane"], r["model"]): r for r in pack["records"]}

    cuda = records[("RTX 3090 · cuda", "qwen3-4B")]
    assert cuda["fit"] == {"w": 512, "b": 99.998, "m": 12.0005, "r2": 1.0, "resid": 0.0}
    assert cuda["pre_max"] == 4096  # nine chunks deep, and not a token further
    assert cuda["ladder"] == [[0, 80.0], [2048, 75.0]] and cuda["kv_max"] == 2048

    # The job's own measured generation rate joins the ladder at its fill (prompt +
    # half the reply): on a lane whose sweep reached one deep fill this is what
    # keeps a shallow task from being priced at the deep-fill rate — the bug that
    # predicted ~9 tok/s against a job that measured 20. A lane whose job left no
    # prefill rate (cuda and mtl above) has no fill to place it at, and a point
    # within a chunk of an existing rung stays out.
    cpu = records[("Apple M1 Max · cpu 8t", "gemma4-E2B")]
    assert [1682, 21.0] in cpu["ladder"]  # ptps·ttft + reply/2, at the job's tps
    assert cpu["ladder"] == sorted(cpu["ladder"])
    # Model load + context init, and not the warm pass: that span is the harness
    # forcing its dispatch widths, which no deployment pays for.
    assert cuda["load_s"] == 1.0
    assert cuda["cold_s"] == 5.0 and cuda["first_launch_s"] == 1.56
    assert cuda["measured"] == {"ttft": 0.7, "tps": 80.0, "total": 4.0}
    # The model's own trained context: the page evaluates its fits past the sweep's
    # depths, so this is the one number that can rule a configuration out entirely.
    assert cuda["n_ctx_train"] == 4096

    # Metal exposes no pinnable shader cache, so this first launch is a floor of
    # unknown tightness rather than a cold compile — still a cost a batch pays, and
    # reported as one (same rule as the first-launch chart).
    mac = records[("Apple M1 Max · mtl", "gemma4-E2B")]
    assert mac["first_launch_s"] == 0.61
    assert mac["fit"]["w"] == 512 and mac["kv_max"] == 2048

    # Nothing measured a cost function: no bar could be drawn from it, and the
    # Results grid above already says why the cell is empty.
    assert ("RTX 3090 · cuda", "gemma4-E4B") not in records  # one chunk, no fit
    assert ("RTX 3090 · cuda", "gemma4-E2B") not in records  # errored, nothing at all
    assert ("Apple M1 Max · mtl", "qwen3-4B") not in records
    json.dumps(pack, allow_nan=False)  # strict — it ships as a JSON island


def test_each_task_owns_its_columns_and_its_note():
    """A task is a select option, a note, and its own three column titles — the title
    is where a chat turn and a background job stop being the same three numbers.
    Editing the inputs edits the selected task; nothing is derived."""
    pack = _fixture_pack()
    tasks = {t["key"]: t for t in pack["tasks"]}
    assert set(tasks) == {"chat", "summarize", "extract"}
    for t in tasks.values():
        assert t["note"]
        assert len(t["columns"]) == len(site.TASK_METRICS)
    assert tasks["chat"]["depth"] > 0  # a turn appended to a live cache
    assert tasks["chat"]["fields"] == ["depth"]  # the one task with a depth input
    assert tasks["summarize"]["measured"] is True  # the one task with dots
    assert tasks["extract"]["mid_unit"] == "s"  # phases in seconds, nobody watching
    assert [t["key"] for t in pack["tasks"]][0] == "chat"  # the page opens on it


def test_task_grid_draws_three_computed_columns_the_page_fills(tmp_path):
    """Same layout as the measured grid — heading on the x axis over the plot area,
    one axis carrier per column, labels then headroom — over data the page inserts:
    the island ships a named, empty set, never numbers computed at build time."""
    spec = _island(_build(tmp_path), "tasks")
    assert spec["data"] == {"name": "tasks"}
    columns = spec["hconcat"]
    assert len(columns) == len(site.TASK_METRICS)

    for column, metric, (title, _) in zip(
        columns, site.TASK_METRICS, site.TASKS[0]["columns"], strict=True
    ):
        assert "title" not in column
        bar, dot, label, headroom, note = column["spec"]["layer"]
        anchor = _anchor_layer(column)  # exactly one, or the heading breaks
        assert anchor is bar and bar["mark"]["type"] == "bar"
        axis = bar["encoding"]["x"]["axis"]
        assert axis["orient"] == "top" and axis["titleAnchor"] == "start"
        text = axis["title"]
        assert (text[0] if isinstance(text, list) else text) == title
        # Generation carries the same named reading-speed anchors as the measured
        # grid; seconds have none.
        if metric == "tps":
            assert axis["values"] == [v for v, _ in site.TASK_ANCHORS["tps"]]
            for value, name in site.TASK_ANCHORS["tps"]:
                assert f"'{value:,} · {name}'" in axis["labelExpr"]
        else:
            assert axis.get("values") is None and axis["grid"] is False
        # An estimate is thinner ink, not a second hue — and a value condition
        # carries no scale, so it adds no legend.
        assert bar["encoding"]["opacity"]["condition"]["test"] == "datum.est"
        assert dot["transform"] == [{"filter": "datum.measured !== null"}]
        assert dot["encoding"]["x"]["field"] == "measured"
        assert label["encoding"]["text"]["field"] == "label"  # a string, not a format
        assert label["encoding"]["x"]["field"] == "label_x"
        assert "datum.label_x" in headroom["transform"][0]["calculate"]
        assert note["transform"] == [{"filter": "datum.note !== null"}]
        assert note["mark"]["fontStyle"] == "italic"


def test_task_columns_say_what_they_price_not_what_the_grid_measured(tmp_path):
    """Each task owns its column titles, so a chat turn's first word and a background
    job's done-after never share a heading — and none of them reuses the measured
    grid's own titles, or the same words would appear twice with different numbers.
    The built spec carries the first task's titles; the page swaps them on
    selection."""
    grid_titles = {t for _, t, *_ in site.METRICS}
    for task in site.TASKS:
        titles = [t for t, _ in task["columns"]]
        assert len(set(titles)) == 3
        assert not grid_titles & set(titles)
    axes = [
        column["spec"]["layer"][0]["encoding"]["x"]["axis"]
        for column in _island(_build(tmp_path), "tasks")["hconcat"]
    ]
    first = site.TASKS[0]["columns"]
    assert [a["title"] for a in axes] == [first[0][0], first[1][0], [first[2][0], first[2][1]]]


def test_preset_notes_name_the_work_in_words_as_well_as_tokens():
    """A token count is not a unit anyone has a feel for. Each preset's note opens
    with the work it stands for in words, with the tokens the arithmetic uses beside
    them; `custom` has no note because its numbers are the reader's own."""
    assert site._words(4096) == "3,000"  # 3,072 words, to the nearest 250
    assert site._words(300) == "225"  # under 500 words: to the nearest 25
    assert site._words(1550) == "1,150"  # under 2,000: to the nearest 50
    assert site._words(120) == "100"

    tasks = {t["key"]: t for t in _fixture_pack()["tasks"]}
    assert tasks["extract"]["note"].startswith(
        "A ~3,000-word prompt (4,096 tokens) and a ~300-word reply (400 tokens)."
    )
    assert tasks["chat"]["note"].startswith(
        "~3,000 words of conversation already in the context (4,096 tokens), "
        "a ~100-word prompt (120 tokens) and a ~150-word reply (200 tokens)."
    )
    assert "1,550 tokens" in tasks["summarize"]["note"]
    # the task's own sentence still follows the workload it opens with
    for key, task in tasks.items():
        assert task["note"].endswith(dict((t["key"], t["note"]) for t in site.TASKS)[key])


def test_the_token_inputs_carry_a_live_word_hint(tmp_path):
    """Each token field has a place for its word count next to it, the page fills it
    from the same ratio the notes are written in, and the ratio is stated once."""
    h = _build(tmp_path)
    for field in ("t-depth", "t-prompt", "t-out"):
        assert f'<span class="words" data-for="{field}"></span>' in h
    assert "about ¾ of a word each" in h
    assert "taskMath.words" in h  # report.js fills the hints from tasks.js
    assert "words: words" in h  # ...and tasks.js exports it


def test_the_task_intro_is_short_and_folds_the_rest(tmp_path):
    """Nine lines of preamble is a wall. The section says what it is and where its one
    hard limit is; how to read the marks sits behind a fold."""
    h = _build(tmp_path)
    intro = " ".join(h.split("<h2>What a task costs</h2>")[1].split("<details")[0].split())
    assert intro.count(".") <= 4  # three sentences, one abbreviation of slack
    # The rule as it now stands: evaluated past the evidence, marked when it is, and
    # stopped only by the model's own trained context.
    assert "marked as an estimate" in intro
    assert "the model itself was trained for" in intro
    assert "extrapolated" not in intro  # the old promise is gone, not softened
    fold = h.split("<h2>What a task costs</h2>")[1].split("</details>")[0]
    assert '<details class="fine">' in fold
    assert "<summary>How to read these bars</summary>" in fold
    assert "half-lit" in fold and "<code>~</code>" in fold
    assert "deeper than the sweep went" in fold  # what the ~ now also covers
    assert "trained context" in fold  # ...and the one thing that draws no bar
    assert "details.fine > summary" in h  # and the fold has a style of its own


def _node(tmp_path, body: str) -> list[str]:
    """`assets/tasks.js` run the way the page runs it: one line of JSON out per
    `console.log` the body writes."""
    script = tmp_path / "run.js"
    script.write_text(
        "var window = {};\n" + (site.PKG / "assets" / "tasks.js").read_text() + "\n" + body
    )
    out = subprocess.run(
        ["node", str(script)],  # noqa: S603 — a file this test wrote
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_refuses_only_what_the_model_cannot_hold(tmp_path):
    """The calculator's one refusal: a task longer than the context the model was
    trained for. That is a fact about the model, not about our sweep's budget, and it
    belongs to the lane rather than to a column — so the first column carries the text
    and the rest of the row is blank. And the page's word counts are the ones the
    preset notes were built with."""
    rows_json, words_json = _node(
        tmp_path,
        f"var pack = {json.dumps(_fixture_pack())};\n"
        "var p = {task: 'summarize', depth: 0, prompt: 200000, out: 300};\n"
        "console.log(JSON.stringify(window.taskMath.taskRows(pack, p)));\n"
        "console.log(JSON.stringify([4096, 300, 1550, 120].map(window.taskMath.words)));\n",
    )

    rows = json.loads(rows_json)
    assert rows, "no lane priced"
    metrics = list(site.TASK_METRICS)
    for lane in {r["lane"] for r in rows}:
        mine = [r for r in rows if r["lane"] == lane]
        notes = {r["metric"]: r["note"] for r in mine}
        assert notes[metrics[0]] == "beyond this model's context"
        assert [notes[m] for m in metrics[1:]] == [None, None]
        assert all(r["value"] is None for r in mine)
        # ...printed once, but hover answers in whichever column was asked
        assert {r["why"] for r in mine} == {"beyond this model's context"}

    assert json.loads(words_json) == [3000, 225, 1150, 100]
    assert [site._words(n) for n in (4096, 300, 1550, 120)] == [
        f"{w:,}" for w in json.loads(words_json)
    ]


# One lane's pack record, shaped like the shelf's slowest CPU lanes: a fit taken to
# 7,168 tokens and a decode ladder that stops at the same fill. A 8,192-token prompt
# is past both and well inside the model's 262,144-token context — the case the
# calculator used to refuse.
PAST_MEASURED = {
    "records": [
        {
            "lane": "Ryzen 7 255 · cpu 8t",
            "dev_class": "CPU",
            "model": "Ministral3-3B",
            "quant": "q4",
            "backend": "llamacpp",
            "rank": 0,
            "fit": {"w": 512, "b": 2451.831, "m": 666.999, "r2": 0.99, "resid": 1.0},
            "pre_max": 7168,
            "ladder": [[0, 31.0], [2048, 28.0], [7168, 22.0]],
            "kv_max": 7168,
            "n_ctx_train": 262144,
            "load_s": 1.0,
            "cold_s": None,
            "first_launch_s": None,
            "measured": None,
        }
    ],
    "presets": [],
    "mode_notes": {},
    "metrics": ["ttft", "tps", "total"],
}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_past_the_measured_depth_the_fit_is_evaluated_and_marked(tmp_path):
    """Past the sweep's deepest point there is still a cost function to evaluate: the
    prefill fit is per-dispatch cost plus a depth-linear term, so evaluating it further
    out is a prediction, not a fiction. Every column keeps its number, wears a `~`, is
    drawn at half ink, and names on hover — in a label, not a paragraph — how far past
    the evidence it went. The generation rate is held at the deepest fill measured,
    which is the slowest observed."""
    (rows_json, depths_json) = _node(
        tmp_path,
        f"var pack = {json.dumps(PAST_MEASURED)};\n"
        "var p = {mode: 'stream', depth: 0, prompt: 8192, out: 300, docs: 1};\n"
        "console.log(JSON.stringify(window.taskMath.taskRows(pack, p)));\n"
        "console.log(JSON.stringify([512, 3584, 7168, 8192]"
        ".map(window.taskMath.depthLabel)));\n",
    )

    rows = {r["metric"]: r for r in json.loads(rows_json)}
    assert set(rows) == {"ttft", "tps", "total"}
    for row in rows.values():
        assert row["value"] is not None and row["note"] is None
        assert row["est"] is True and row["label"].startswith("~")

    past = "past the measured 7k tokens"
    held = "rate held at the deepest fill measured"
    assert rows["ttft"]["why"] == past
    assert rows["tps"]["why"] == held
    assert rows["total"]["why"] == f"{past}; {held}"
    # labels, not sentences, so the hover card stays a card and not a paragraph
    for row in rows.values():
        for clause in row["why"].split("; "):
            assert len(clause) <= 40, clause
            assert "." not in clause and "—" not in clause
    # 16 full chunks at 2,451.8 ms, plus 667.0 ms per 1k of depth already read
    assert rows["ttft"]["value"] == pytest.approx(80.2, abs=0.1)
    assert rows["tps"]["value"] == pytest.approx(22.0, abs=0.1)  # the held rate
    # a depth reads as a reader would say it, to the nearest half-thousand
    assert json.loads(depths_json) == ["512", "3.5k", "7k", "8k"]


def test_the_task_section_ships_its_controls_its_pack_and_its_arithmetic(tmp_path):
    """The controls, the pack and `tasks.js` all have to be on the page for any of it
    to work, and the pack is a JSON island like every other data on the page — never
    a `|safe` string."""
    h = _build(tmp_path)
    assert 'data-spec="tasks"' in h and "What a task costs" in h
    assert '<script type="application/json" id="task-pack">' in h
    assert "window.taskMath" in h  # assets/tasks.js inlined
    for element in (
        'id="t-preset"',
        'id="t-depth"',
        'id="t-prompt"',
        'id="t-out"',
        'id="task-note"',
        'data-field="depth"',
    ):
        assert element in h
    assert 'id="t-docs"' not in h  # one task is one task — no document multiplier
    for task in site.TASKS:
        assert f'value="{task["key"]}"' in h
        assert task["label"].replace("&", "&amp;") in h  # jinja escapes the option text


def _memory_frames():
    """The fixture shelf's two memory inputs, laned like the page lanes them."""
    df = site._with_lanes(load_results(FIXTURES))
    mem = site._with_lanes(load_memory(FIXTURES))
    return df[df.status == "ok"], mem


def test_footprint_draws_the_total_over_a_dashed_weights_floor(tmp_path):
    """The allocator curve is two lines per lane: the total, and the weights floor
    that does not move with context. The gap between them is the context's cost."""
    spec = _island(_build(tmp_path), "memory-footprint")
    total, weights = spec["layer"]
    assert total["encoding"]["y"]["field"] == "total_mb"
    assert "strokeDash" not in total["mark"]
    assert weights["encoding"]["y"]["field"] == "weights_mb"
    assert weights["mark"]["strokeDash"] == [4, 3]
    for layer in (total, weights):
        assert layer["encoding"]["x"]["field"] == "n_ctx"
        assert layer["encoding"]["y"]["scale"] == {"zero": True}
        assert layer["encoding"]["y"]["title"] == "MB reserved"
        titles = [t.get("title", t["field"]) for t in layer["encoding"]["tooltip"]]
        assert "KV cache (MB)" in titles and "compute buffers (MB)" in titles

    rows = spec["data"]["values"]
    box = [r for r in rows if r["lane"] == "RTX 3090 · cuda"]
    assert [r["n_ctx"] for r in box] == [512, 2048, 8192]
    assert {r["weights_mb"] for r in box} == {600.0}  # the floor is flat
    assert [r["total_mb"] for r in box] == [687.5, 800.0, 1250.0]


def test_job_memory_rows_keep_both_pools_and_name_the_missing_one():
    """RSS and device-local memory are separate numbers: the 3090's job held both at
    once. A unified-memory lane reports no second pool — that reads as a note, never
    as a zero — and the allocator reference comes from the deepest ladder point."""
    ok, mem = _memory_frames()
    rows = {(r["lane"], r["model"]): r for r in site._job_memory_rows(ok, mem)}

    cuda = rows[("RTX 3090 · cuda", "qwen3-4B")]
    assert cuda["rss_peak"] == 1300.0 and cuda["vram_peak"] == 1700.0
    assert cuda["rss_sustained"] == 1200.0 and cuda["prefill_rss_peak"] == 1500.0
    assert cuda["vram_note"] is None  # a real number needs no excuse
    assert (cuda["alloc_total"], cuda["alloc_ctx"]) == (1250.0, 8192)
    assert cuda["label_x"] == 1700.0  # past both marks, so the label clears the dot

    mac = rows[("Apple M1 Max · mtl", "gemma4-E2B")]
    assert mac["vram_peak"] is None
    assert mac["vram_note"] == "unified memory: in RSS"
    # no allocator ladder on that run: absent, not zero, and still strict JSON
    assert mac["alloc_total"] is None and mac["alloc_ctx"] is None
    json.dumps(site._job_memory_rows(ok, mem), allow_nan=False)


def test_job_memory_marks_the_process_bar_the_pool_and_the_reference(tmp_path):
    spec = _island(_build(tmp_path), "memory-job")
    bar, alloc, vram, label, headroom, note = spec["layer"]
    assert bar["mark"]["type"] == "bar"
    assert bar["encoding"]["x"]["field"] == "rss_peak"
    assert bar["encoding"]["y"]["sort"] == site._lane_order(
        site._with_lanes(load_results(FIXTURES))
    )
    # The allocator reference is dashed and drawn in the surface's own ink; a lane
    # with no ladder draws none of it.
    assert alloc["mark"]["strokeDash"] == [2, 2]
    assert alloc["mark"]["color"] == "currentColor"
    assert alloc["transform"] == [{"filter": "datum.alloc_total !== null"}]
    assert vram["transform"] == [{"filter": "datum.vram_peak !== null"}]
    assert vram["encoding"]["x"]["field"] == "vram_peak"
    assert label["encoding"]["x"]["field"] == "label_x"
    assert "datum.label_x" in headroom["transform"][0]["calculate"]
    assert note["transform"] == [{"filter": "datum.vram_note !== null"}]
    assert note["mark"]["fontStyle"] == "italic" and note["mark"]["opacity"] == 0.65


def test_a_shelf_with_no_allocator_ladder_still_builds(tmp_path):
    """The m1-max submission carries no `memory_points`: the footprint chart has
    nothing to draw and is not rendered, and the process chart still is."""
    only_mac = tmp_path / "published"
    shutil.copytree(FIXTURES / "m1-max", only_mac / "m1-max")
    h = _build(tmp_path, published=only_mac)
    assert 'data-spec="memory-footprint"' not in h
    assert 'data-spec="memory-job"' in h
    assert "unified memory: in RSS" in h


def _fixture_threads():
    """The fixture shelf's thread ladder, as the page assembles it: the fits, the
    measured points, and how far each (lane, model, phase) was measured."""
    df = site._with_lanes(load_results(FIXTURES))
    fits = site._thread_fits(df)
    points = site._thread_rows(site._with_lanes(load_thread_scaling(FIXTURES)), fits)
    return fits, points, site._thread_spans(points)


def test_thread_points_carry_each_phase_in_its_own_unit():
    """A prefill point is the chunk's milliseconds, a decode point the burst's tok/s —
    the units the two fits were taken in, so both fitted parameters can be drawn as
    asymptotes. The ring marks the width the lane actually runs."""
    _, points, spans = _fixture_threads()
    lane = "Apple M1 Max · cpu 8t"
    assert {r["lane"] for r in points} == {lane}  # CPU lanes only

    pre = {r["threads"]: r for r in points if r["phase"] == "prefill"}
    assert sorted(pre) == [2, 4, 8]
    assert [pre[n]["value"] for n in (8, 4, 2)] == [212.4, 378.6, 719.5]
    assert [pre[n]["at_width"] for n in (8, 4, 2)] == [True, False, False]
    assert pre[8]["tokens"] == 128
    assert all(r["kv_fill"] is None for r in pre.values())  # None, never NaN
    assert {r["n_widths"] for r in pre.values()} == {3}
    assert pre[8]["r2"] == 0.99996  # the fit the dots belong to, in the tooltip

    dec = {r["threads"]: r for r in points if r["phase"] == "decode"}
    assert [dec[n]["value"] for n in (8, 4, 2)] == [22.1, 20.4, 15.2]
    assert {r["kv_fill"] for r in dec.values()} == {2048}
    assert dec[8]["tokens"] == 16

    assert spans[(lane, "gemma4-E2B", "prefill")] == {
        "widest": 8,
        "n": 3,
        "tokens": 128,
        "kv_fill": None,
    }
    json.dumps(points, allow_nan=False)  # strict — the island is


def test_the_fit_is_drawn_over_its_evidence_and_no_further():
    """The ladder walks down from the lane's own width, so the curve stops there: past
    it nothing was measured. Both fitted parameters draw as asymptotes, and the
    90%-of-peak width draws as a rule because it falls inside the measured widths."""
    fits, points, spans = _fixture_threads()
    rows = site._thread_fit_rows(fits, spans)
    lane, model = "Apple M1 Max · cpu 8t", "gemma4-E2B"

    for phase in ("prefill", "decode"):
        curve = [r for r in rows if r["kind"] == "fit" and r["phase"] == phase]
        assert len(curve) == site.THREAD_FIT_SAMPLES + 1
        threads = [r["threads"] for r in curve]
        assert threads == sorted(threads)
        assert min(threads) == 1 and max(threads) == spans[(lane, model, phase)]["widest"]

    floor = [r for r in rows if r["kind"] == "asymptote" and r["phase"] == "prefill"]
    assert [r["value"] for r in floor] == [41.95]  # positive, so it is a floor
    assert "floor" in floor[0]["label"]
    ceiling = [r for r in rows if r["kind"] == "asymptote" and r["phase"] == "decode"]
    assert [r["value"] for r in ceiling] == [22.44]
    assert ceiling[0]["threads"] is None  # a horizontal rule has no width of its own

    p90 = [r for r in rows if r["kind"] == "p90"]
    assert len(p90) == 1 and p90[0]["phase"] == "decode"
    assert p90[0]["in_domain"] is True
    assert p90[0]["threads"] == fits[(lane, model)]["decode"]["p90"] == 4.01
    assert p90[0]["label"] == "90% of peak at 4.01 threads"
    json.dumps(rows + points, allow_nan=False)


def test_a_short_fit_claims_neither_a_floor_nor_a_width_it_never_reached():
    """Two widths fit two parameters with one residual left over. A floor that comes
    out negative is not a floor and draws no line; a 90%-of-peak width above every
    width measured draws no rule either — it would stretch the axis to reach itself —
    so it becomes a note, and the table says both out loud."""
    lane, model = "Ryzen 5 PRO 230 · cpu 4t", "Ministral3-3B"
    fits = {
        (lane, model): {
            "lane": lane,
            "dev_class": "CPU",
            "machine": "mini",
            "model": model,
            "quant": "q4",
            "backend": "llamacpp",
            "prefill": {
                "floor_ms": -12.0,
                "scaled_ms": 900.0,
                "floor_pct": -1.35,
                "r2": 0.9,
                "width": 4,
            },
            "decode": {
                "rate_max_tps": 30.0,
                "threads_scale": 6.0,
                "p90": 13.82,
                "r2": 0.95,
                "width": 4,
            },
        }
    }
    spans = {
        (lane, model, "prefill"): {"widest": 4, "n": 2, "tokens": 128, "kv_fill": None},
        (lane, model, "decode"): {"widest": 4, "n": 2, "tokens": 16, "kv_fill": 2048},
    }
    rows = site._thread_fit_rows(fits, spans)
    assert not [r for r in rows if r["kind"] == "asymptote" and r["phase"] == "prefill"]
    p90 = next(r for r in rows if r["kind"] == "p90")
    assert p90["in_domain"] is False and p90["threads"] is None
    assert p90["label"] == "90% of peak above 4 threads"

    scalar = site._thread_scalar_rows(fits, spans)[0]
    assert scalar["floor"] == "no measurable floor"
    assert scalar["p90"] == "13.82 threads (above the 4 measured)"
    assert scalar["note"] == "two widths only"


def test_thread_scalars_headline_the_widths_the_charts_show():
    fits, _, spans = _fixture_threads()
    rows = site._thread_scalar_rows(fits, spans)
    assert len(rows) == 1
    row = rows[0]
    assert row["lane"] == "Apple M1 Max · cpu 8t" and row["model"] == "gemma4-E2B"
    assert row["width"] == "8"  # both phases run the same width
    assert row["p90"] == "4.01 threads"  # half the width buys 90% of decode
    assert row["ceiling"] == "22.4 tok/s"
    assert row["floor"] == "3.01%"  # prefill keeps rewarding cores
    assert row["r2"] == "1.0000 / 0.9971"
    assert row["note"] == "—"  # three widths: nothing to caveat


def test_thread_islands_draw_the_dots_the_fit_and_its_limits(tmp_path):
    """One island per phase, each in its own unit: the measured widths as dots with the
    operating width ringed, the harness's fit as the line through them, and the fitted
    limit as a dashed line. The decode island also marks where the fit reaches 90% of
    its ceiling."""
    h = _build(tmp_path)
    assert "Thread width" in h
    units = {"prefill": "ms per 128-token chunk", "decode": "tok/s of a 16-token burst"}
    titled = {p["phase"]: p for p in site.THREAD_PHASES}
    for phase, y_title in units.items():
        spec = _island(h, f"thread-{phase}")
        assert spec["width"] == 400 and spec["height"] == 220
        assert spec["title"]["text"] == titled[phase]["title"]
        # the caveat rides the chart: these work units compare across widths only
        assert "compare widths" in spec["title"]["subtitle"]
        kinds = [layer["transform"][0]["filter"] for layer in spec["layer"]]
        assert "datum.kind === 'fit'" in kinds
        assert "datum.kind === 'point'" in kinds
        assert "datum.kind === 'point' && datum.at_width" in kinds
        assert "datum.kind === 'asymptote'" in kinds

        by_kind = dict(zip(kinds, spec["layer"], strict=True))
        fit = by_kind["datum.kind === 'fit'"]
        assert fit["mark"] == {"type": "line", "strokeWidth": 2}
        assert fit["encoding"]["x"]["field"] == "threads"
        assert fit["encoding"]["x"]["axis"]["tickMinStep"] == 1
        # one thread is the fit's own baseline; zero threads is not a width
        assert fit["encoding"]["x"]["scale"] == {"zero": False, "nice": False}
        assert fit["encoding"]["y"]["title"] == y_title
        assert fit["encoding"]["y"]["scale"] == {"zero": True}
        dot = by_kind["datum.kind === 'point'"]
        assert dot["mark"]["filled"] is True and dot["mark"]["size"] == 55
        ring = by_kind["datum.kind === 'point' && datum.at_width"]
        assert ring["mark"]["filled"] is False and ring["mark"]["size"] == 110
        asymptote = by_kind["datum.kind === 'asymptote'"]
        assert asymptote["mark"]["strokeDash"] == [4, 3]
        assert "x" not in asymptote["encoding"]  # a limit spans the whole width
        # A dot hands over its own measurement and the fit through it; the dashed
        # lines hand over what they are, because a fitted parameter is not a width the
        # ladder ever ran. The work unit is in the y-axis title and the widths are the
        # dots, so neither is repeated on hover.
        titles = [t.get("title", t["field"]) for t in dot["encoding"]["tooltip"]]
        assert titles[:2] == ["intra-op threads", titled[phase]["value_title"]]
        assert "fit r²" in titles
        assert "work unit (tokens)" not in titles and "line" not in titles
        assert ("primed fill (tokens)" in titles) is (phase == "decode")
        assert [t["field"] for t in asymptote["encoding"]["tooltip"]] == ["label", "value"]

    decode = _island(h, "thread-decode")
    p90 = [
        layer
        for layer in decode["layer"]
        if "datum.kind === 'p90'" in layer["transform"][0]["filter"]
    ]
    rule, label, note = p90
    assert rule["mark"]["type"] == "rule" and rule["mark"]["strokeDash"] == [2, 3]
    assert rule["encoding"]["x"]["field"] == "threads"
    assert label["mark"]["type"] == "text" and label["encoding"]["text"]["field"] == "label"
    # the out-of-domain case says so in text instead of drawing a rule off the axis
    assert note["transform"][0]["filter"].endswith("!datum.in_domain")
    assert note["encoding"]["x"] == {"datum": 1, "type": "quantitative"}

    # the table under the charts carries the derived numbers, in the page's own ink
    for cell in ("Apple M1 Max · cpu 8t", "4.01 threads", "22.4 tok/s", "3.01%"):
        assert cell in h


def test_a_shelf_with_no_cpu_lane_renders_no_thread_section(tmp_path):
    """No published machine carries a thread ladder yet, so the section has to be
    absent rather than empty — and building without one must not raise."""
    only_box = tmp_path / "published"
    shutil.copytree(FIXTURES / "3090-box", only_box / "3090-box")
    h = _build(tmp_path, published=only_box)
    assert 'data-spec="thread' not in h
    assert "Thread width" not in h
    assert 'data-spec="grid"' in h  # the rest of the page is unaffected


def test_wide_content_scrolls_inside_its_own_box(tmp_path):
    """The page never scrolls sideways as a whole: a chart wider than the window
    scrolls inside its island, and a wide table inside its own wrapper. The island rule
    has to restate `display` against the class vega-embed stamps on it at mount time —
    its injected `inline-block` would size the box to the chart and push the page."""
    h = _build(tmp_path)
    css = h.split("<style>")[1].split("</style>")[0]
    island = next(line for line in css.splitlines() if line.startswith(".island,"))
    assert ".island.vega-embed" in island
    assert "display: block" in island and "overflow-x: auto" in island
    assert ".scroll-x { overflow-x: auto; }" in css
    # every table on the page opens inside one of those wrappers
    before = re.findall(r"(.{0,40})<table>", h, re.S)
    assert before, "no table rendered"
    for prefix in before:
        assert 'class="scroll-x"' in prefix, f"a table outside a wrapper: {prefix!r}"


def test_build_is_self_contained(tmp_path):
    h = _build(tmp_path)
    assert "/* stub */" in h  # vega inlined from the cache
    assert "<script src=" not in h and "<link" not in h  # nothing external


def test_empty_shelf_still_advertises_the_one_liner(tmp_path):
    empty = tmp_path / "published"
    empty.mkdir()
    h = _build(tmp_path, published=empty)
    assert "No submissions yet" in h
    assert site.INSTALL_BASH in h


def test_hostile_submission_strings_never_execute(tmp_path):
    """Every free-text field a submission controls, filled with a script-
    breaking payload — the built page must contain no live script tag from it."""
    poisoned = tmp_path / "published"
    shutil.copytree(FIXTURES, poisoned)
    doc_path = next(poisoned.glob("*/llamacpp-results.json"))
    doc = json.loads(doc_path.read_text())
    doc["machine"]["cpu"] = PAYLOAD
    doc["machine"]["gpus"] = [PAYLOAD]
    for run in doc["runs"]:
        run["device"] = PAYLOAD
        if run["job"].get("sample_completions"):
            run["job"]["sample_completions"] = [PAYLOAD]
    doc_path.write_text(json.dumps(doc))

    h = _build(tmp_path, published=poisoned)
    assert "<script>alert(1)" not in h  # never as live markup
    # The payload IS present — as inert, escaped text in both contexts.
    assert "\\u003c/script\\u003e" in h or "&lt;/script&gt;" in h


# Every device string the shelf has produced so far, with the lane label and band
# it must resolve to. The driver names a GPU only sometimes; when it doesn't, the
# lane wears the chip it lives in.
DEVICES = [
    # (device, machine cpu, family) → (lane chip, device class)
    ("Apple M5 Pro", "Apple M5 Pro", "cpu", "Apple M5 Pro", "CPU"),
    ("Apple M5 Pro", "Apple M5 Pro", "mtl", "Apple M5 Pro", "integrated GPU"),
    (
        "Intel(R) Core(TM) Ultra 5 125U",
        "Intel(R) Core(TM) Ultra 5 125U",
        "cpu",
        "Core Ultra 5 125U",
        "CPU",
    ),
    (
        "Intel(R) Graphics (MTL)",
        "Intel(R) Core(TM) Ultra 5 125U",
        "vulkan",
        "Core Ultra 5 125U iGPU",
        "integrated GPU",
    ),
    (
        "AMD Ryzen 7 255 w/ Radeon 780M Graphics",
        "AMD Ryzen 7 255 w/ Radeon 780M Graphics",
        "cpu",
        "Ryzen 7 255",
        "CPU",
    ),
    (
        "AMD Radeon Graphics (RADV PHOENIX)",
        "AMD Ryzen 7 255 w/ Radeon 780M Graphics",
        "vulkan",
        "Ryzen 7 255 iGPU",
        "integrated GPU",
    ),
    (
        "AMD Radeon 760M Graphics (RADV PHOENIX)",
        "AMD Ryzen 5 PRO 230 w/ Radeon 760M Graphics",
        "vulkan",
        "Ryzen 5 PRO 230 iGPU",
        "integrated GPU",
    ),
    (
        "AMD Ryzen 9 9950X 16-Core Processor",
        "AMD Ryzen 9 9950X 16-Core Processor",
        "cpu",
        "Ryzen 9 9950X",
        "CPU",
    ),
    # the desktop APU's iGPU: RADV reports the CPU's own brand string
    (
        "AMD Ryzen 9 9950X 16-Core Processor (RADV RAPHAEL_MENDOCINO)",
        "AMD Ryzen 9 9950X 16-Core Processor",
        "vulkan",
        "Ryzen 9 9950X iGPU",
        "integrated GPU",
    ),
    (
        "NVIDIA GeForce RTX 5080",
        "AMD Ryzen 9 9950X 16-Core Processor",
        "vulkan",
        "RTX 5080",
        "discrete GPU",
    ),
    (
        "NVIDIA GeForce RTX 3090",
        "AMD Ryzen 9 5950X 16-Core Processor",
        "cuda",
        "RTX 3090",
        "discrete GPU",
    ),
]


@pytest.mark.parametrize(("device", "cpu", "family", "chip", "klass"), DEVICES)
def test_lane_identity(device, cpu, family, chip, klass):
    assert site._lane_chip(device, cpu, family) == chip
    assert site._dev_class(device, cpu, family) == klass


def test_lanes_stay_distinct_across_identical_machines():
    """Two of the same laptop must not pool into one lane."""
    df = pd.DataFrame(
        {
            "provider": ["vulkan:0", "vulkan:0"],
            "machine": ["nuc-a", "nuc-b"],
            "device": ["Intel(R) Graphics (MTL)"] * 2,
            "cpu": ["Intel(R) Core(TM) Ultra 5 125U"] * 2,
            "threads_batch": [12, 12],
            "threads_decode": [12, 12],
        }
    )
    lanes = site._with_lanes(df).lane
    assert lanes.nunique() == 2
    assert all("Core Ultra 5 125U iGPU · vulkan" in lane for lane in lanes)
