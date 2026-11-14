"""The site builder renders one self-contained page around one data pack — and
hostile submission strings can never reach an executable position: HTML context is
autoescaped, the pack goes through `tojson` (`<` → \\u003c), so a payload that tries
to terminate a script element survives only as inert text."""

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from analysis import (
    load_memory,
    load_probes,
    load_results,
    load_sweeps,
    load_thread_scaling,
    site,
)

FIXTURES = Path(__file__).parent / "fixtures"
PAYLOAD = "</script><script>alert(1)</script>"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _build(tmp_path, published=FIXTURES):
    out = tmp_path / "report.html"
    site.build(published, out)
    return out.read_text()


def _island(html: str) -> dict:
    found = re.search(r'<script type="application/json" id="pack">(.*?)</script>', html, re.S)
    assert found, "no data pack rendered"

    def strict(token):  # NaN / Infinity are not JSON; a browser would refuse them
        raise AssertionError(f"non-JSON constant in the pack: {token}")

    return json.loads(found.group(1), parse_constant=strict)


def _fixture_pack() -> dict:
    df = site._with_lanes(load_results(FIXTURES))
    loaders = (load_sweeps, load_memory, load_probes, load_thread_scaling)
    frames = [load(FIXTURES) for load in loaders]
    return site._pack(df, *(f if f.empty else site._with_lanes(f) for f in frames))


# ------------------------------------------------------------------- the page
def test_build_renders_the_three_panels_around_a_strict_json_pack(tmp_path):
    h = _build(tmp_path)
    for panel in ("view-coverage", "view-compare", "view-data"):
        assert f'id="{panel}"' in h
    for number in ("cov-num", "mem-num"):  # the two answers above the chart
        assert f'id="{number}"' in h
    pack = _island(h)
    assert pack == json.loads(json.dumps(_fixture_pack()))
    # the rail's controls are there for the page script to fill
    for control in ("models", "in-prompt", "in-out", "split-zone", "backends", "split-kind"):
        assert f'id="{control}"' in h
    # and the three scripts that read the pack ride along, calculator first
    order = [
        h.index(name) for name in ("window.taskMath", "window.fleetMath", 'getElementById("pack")')
    ]
    assert order == sorted(order)


def test_build_is_self_contained(tmp_path):
    h = _build(tmp_path)
    assert "<script src=" not in h and "<link" not in h  # nothing external
    assert "@import" not in h and "url(http" not in h


def test_wide_content_scrolls_inside_its_own_box(tmp_path):
    """The page never scrolls sideways as a whole: each table sits in a wrapper
    that scrolls on its own."""
    h = _build(tmp_path)
    css = h.split("<style>")[1].split("</style>")[0]
    assert ".scroll-x { overflow-x: auto; }" in css
    before = re.findall(r"(.{0,40})<table", h, re.S)
    assert before, "no table rendered"
    for prefix in before:
        assert 'class="scroll-x"' in prefix, f"a table outside a wrapper: {prefix!r}"


def test_empty_shelf_still_advertises_the_one_liner(tmp_path):
    empty = tmp_path / "published"
    empty.mkdir()
    h = _build(tmp_path, published=empty)
    assert "No submissions yet" in h
    assert site.INSTALL_BASH in h
    assert 'id="pack"' not in h and "<script" not in h  # nothing to price, nothing to run


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
        run["model"] = PAYLOAD
        if run["job"].get("sample_completions"):
            run["job"]["sample_completions"] = [PAYLOAD]
    doc_path.write_text(json.dumps(doc))

    h = _build(tmp_path, published=poisoned)
    assert "<script>alert(1)" not in h  # never as live markup
    assert "\\u003c/script\\u003e" in h  # the payload is present, as inert JSON text
    assert any(PAYLOAD in lane["id"] for lane in _island(h)["lanes"])


def test_the_page_script_escapes_every_string_it_takes_from_the_pack():
    """The pack is inert, but the page writes it into markup. Every interpolation of
    a submission-controlled field goes through `esc`; a bare one would be the one
    place a payload could run."""
    js = (site.PKG / "assets" / "report.js").read_text()
    fields = "|".join(
        (
            r"(?:mc|r\.mc|q\.d\.mc|s\.mc|d\.mc)\.(?:name|full|id|os)",
            r"(?:l|r\.lane|d\.lane)\.(?:short|id)",
            r"m\.id|mo\.id|state\.model",
        )
    )
    bare = re.findall(r"\$\{(" + fields + r")\}", js)
    assert not bare, f"written without esc: {bare}"


# ------------------------------------------------------------------- the pack
def test_pack_carries_the_measured_curves_and_what_a_launch_costs():
    """The page prices a task from this pack alone, so it has to hold both measured
    cost curves — the cumulative time-to-first-token points and the decode ladder,
    never a fit of them — how far each was measured, and what a launch costs once."""
    pack = _fixture_pack()
    records = {(r["lane"], r["model"]): r for r in pack["records"]}

    cuda = records[("RTX 3090 · cuda", "qwen3-4B")]
    # the sweep's own cumulative clock at every chunk boundary, from the origin
    assert cuda["curve"][:3] == [[0, 0], [512, 100.0], [1024, 206.1]]
    assert cuda["curve"][-1] == [4096, 979.9]
    assert cuda["curve"] == sorted(cuda["curve"])
    assert cuda["pre_max"] == 4096  # nine chunks deep, and not a token further
    assert cuda["ladder"] == [[0, 80.0], [2048, 75.0]] and cuda["kv_max"] == 2048

    # The job's own measured generation rate joins the ladder at its fill (prompt +
    # half the reply): on a lane whose sweep reached one deep fill this is what
    # keeps a shallow task from being priced at the deep-fill rate. A lane whose job
    # left no prefill rate (cuda and mtl above) has no fill to place it at, and a
    # point within a chunk of an existing rung stays out.
    cpu = records[("Apple M1 Max · cpu 8t (10C)", "gemma4-E2B")]
    assert [1682, 21.0] in cpu["ladder"]  # ptps·ttft + reply/2, at the job's tps
    assert cpu["ladder"] == sorted(cpu["ladder"])
    # Model load + context init, and not the warm pass: that span is the survey tool
    # forcing its dispatch widths, which no deployment pays for.
    assert cuda["load_s"] == 1.0
    # The model's own trained context: the page carries its curves past the sweep's
    # depths, so this is the one number that can rule a task out entirely.
    assert cuda["n_ctx_train"] == 4096

    # One measured chunk is still a curve — priceable to its depth, carried onward
    # and marked past it. With no ladder the page has nothing to price writing with.
    slow = records[("RTX 3090 · cuda", "gemma4-E4B")]
    assert slow["curve"] == [[0, 0], [512, 9000.0]] and slow["pre_max"] == 512
    assert slow["ladder"] == []

    # A CPU lane carries its thread ladder, each phase with the width the lane ran
    # it at; a GPU lane has none.
    assert cpu["scaling"] == {
        "prefill": {"threads": 8, "points": [[2, 177.9], [4, 338.09], [8, 602.64]]},
        "decode": {"threads": 8, "points": [[2, 15.2], [4, 20.4], [8, 22.1]]},
    }
    assert cuda["scaling"] is None

    # Nothing measured at all: nothing to evaluate, so no record.
    assert ("RTX 3090 · cuda", "gemma4-E2B") not in records  # errored
    assert ("Apple M1 Max · mtl", "qwen3-4B") not in records
    json.dumps(pack, allow_nan=False)  # strict — it ships as a JSON island


def test_compilation_is_claimed_only_where_it_can_be_estimated():
    """The 3090's cuda lane compiled pipelines into a cache the survey tool pinned,
    so its warm pass less its own prefill (1.83 s − 0.27 s) is a from-scratch
    compile. The M1's Metal lane exposes no cache path: that launch still cost what
    it cost, so the estimate is reported. A CPU lane compiles nothing, and a lane
    whose sweep reached one chunk has no fit to net the warm pass out with."""
    records = {(r["lane"], r["model"]): r for r in _fixture_pack()["records"]}
    assert records[("RTX 3090 · cuda", "qwen3-4B")]["compile_s"] == 1.56
    assert records[("Apple M1 Max · mtl", "gemma4-E2B")]["compile_s"] == 0.61
    assert records[("Apple M1 Max · cpu 8t (10C)", "gemma4-E2B")]["compile_s"] is None
    assert records[("RTX 3090 · cuda", "gemma4-E4B")]["compile_s"] is None


def _lane_row(**fields) -> pd.DataFrame:
    base = {
        "machine": "box",
        "provider": "vulkan:0",
        "device": "AMD Radeon Graphics (RADV PHOENIX)",
        "cpu": "AMD Ryzen 7 255 w/ Radeon 780M Graphics",
        "cpu_cores": 8,
        "threads_batch": 8,
        "threads_decode": 8,
        "shader_cache": "redirected",
        "shader_bytes": 2711058,
        "shader_warmup_ms": 2990.76,
        "fit_width": 512,
        "fit_intercept_ms": 100.0,
        "fit_slope_ms_per_1k": 0.0,
    }
    return site._with_lanes(pd.DataFrame([{**base, **fields}]))


def test_compile_is_the_warm_pass_less_its_own_prefill():
    """2.99 s of warm pass less the 0.26 s its 512+512+256+32-token walk costs at a
    flat 100 ms per full chunk."""
    (gpu,) = _lane_row().itertuples()
    assert site._compile_seconds(gpu) == 2.73
    # A CPU lane's non-zero `shader_bytes` is the GPU backend's registry init, not
    # this run's compilation.
    (cpu,) = _lane_row(
        provider="cpu:0", device="AMD Ryzen 7 255 w/ Radeon 780M Graphics"
    ).itertuples()
    assert site._compile_seconds(cpu) is None
    # A pinned cache left empty: the run compiled nothing.
    (clean,) = _lane_row(shader_bytes=0).itertuples()
    assert site._compile_seconds(clean) is None
    # No cost function: the compile is unestimable, which is not the span and not zero.
    (unfit,) = _lane_row(
        fit_width=None, fit_intercept_ms=None, fit_slope_ms_per_1k=None
    ).itertuples()
    assert site._compile_seconds(unfit) is None


def test_pack_names_lanes_machines_and_models_for_a_reader():
    pack = _fixture_pack()
    lanes = {lane["id"]: lane for lane in pack["lanes"]}
    assert lanes["RTX 3090 · cuda"]["short"] == "RTX 3090"
    assert lanes["Apple M1 Max · mtl"]["short"] == "Apple M1 Max GPU"
    cpu = lanes["Apple M1 Max · cpu 8t (10C)"]
    assert cpu["short"] == "Apple M1 Max CPU" and cpu["threads"] == 8
    # the bare probes ride on the lane they were run on; a lane with none says so
    assert (lanes["RTX 3090 · cuda"]["gemm"], lanes["RTX 3090 · cuda"]["d2d"]) == (55.2, 800.1)
    assert cpu["gemm"] is None and cpu["d2d"] is None

    machines = {m["id"]: m for m in pack["machines"]}
    assert machines["3090-box"]["name"] == "Ryzen 9 5950X + RTX 3090"
    assert machines["3090-box"]["kind"] == "pc-gpu"
    assert machines["m1-max"]["name"] == "Apple M1 Max" and machines["m1-max"]["kind"] == "mac"

    assert [m["id"] for m in pack["models"]] == ["gemma4-E2B", "gemma4-E4B", "qwen3-4B"]
    assert pack["default_model"] == site.DEFAULT_MODEL
    # the page's vocabulary: every class it can draw, every backend the shelf holds
    assert [c["key"] for c in pack["classes"]] == [k for k, *_ in site.CLASSES]
    assert sum(c["share"] for c in pack["classes"]) == 100
    # the two PC classes draw on the same CPUs, held to the same core counts; a
    # Mac draws on its own, as measured
    assert [c["cpus"] for c in pack["classes"]] == ["pc", "pc", "mac"]
    pc = {"typical": 6, "weak": 4}
    assert [c["cores"] for c in pack["classes"]] == [pc, pc, None]
    assert pack["families"] == [
        {"key": "cpu", "label": "CPU"},
        {"key": "mtl", "label": "Metal"},
        {"key": "cuda", "label": "CUDA"},
    ]


def test_every_class_a_machine_can_have_has_a_colour():
    """A machine's class is its colour on the page; a class the stylesheet does not
    know would draw black."""
    css = (site.PKG / "assets" / "report.css").read_text()
    for key, *_ in site.CLASSES:
        assert f"--c-{key}:" in css and f".k-{key} " in css and f".bg-{key} " in css


@pytest.mark.parametrize(
    ("os", "cpu", "dev_classes", "kind"),
    [
        ("macos", "Apple M4 Pro", {"integrated GPU", "CPU"}, "mac"),
        (
            "linux",
            "AMD Ryzen 9 9950X 16-Core Processor",
            {"discrete GPU", "integrated GPU", "CPU"},
            "pc-gpu",
        ),
        ("windows", "AMD Ryzen 5 PRO 230 w/ Radeon 760M Graphics", {"integrated GPU", "CPU"}, "pc"),
        ("linux", "Intel(R) Xeon(R) Gold 6130", {"CPU"}, "pc"),
        # an Intel Mac is PC hardware as far as a fleet is concerned
        ("macos", "Intel(R) Core(TM) i9-9980HK", {"integrated GPU", "CPU"}, "pc"),
    ],
)
def test_machine_class(os, cpu, dev_classes, kind):
    assert site._machine_class(os, cpu, dev_classes) == kind


def test_display_names_drop_what_a_reader_does_not_need():
    assert site._brief("13th Gen Intel Core i7-1370P") == "Core i7-1370P"
    assert site._brief("Ryzen Threadripper PRO 3975WX 32-Cores") == "Threadripper 3975WX"
    assert site._brief("Radeon Pro WX 3200 Series") == "Radeon Pro WX 3200"
    assert site._brief("RTX 5080") == "RTX 5080"
    intel = ("Intel(R) Graphics (MTL)", "Intel(R) Core(TM) Ultra 5 125U", "vulkan")
    assert site._lane_short(*intel) == "Core Ultra 5 125U iGPU"
    assert site._lane_short("Apple M5 Pro", "Apple M5 Pro", "mtl") == "Apple M5 Pro GPU"
    assert site._lane_short("Apple M5 Pro", "Apple M5 Pro", "cpu") == "Apple M5 Pro CPU"


def test_memory_is_a_line_through_the_allocators_own_numbers():
    """The context buffer grows with depth and need not start at zero; the weights
    and the compute buffers do not move."""
    mem = pd.DataFrame(
        {
            "lane": ["a"] * 3,
            "model": ["m"] * 3,
            "quant": ["q4"] * 3,
            "n_ctx": [512, 2048, 8192],
            "weights_mb": [3251.4] * 3,
            "compute_mb": [537.9] * 3,
            "kv_mb": [69.5, 119.8, 321.1],
        }
    )
    (line,) = site._memory(mem).values()
    assert line == {"weights": 3251.4, "compute": 537.9, "kv_base": 52.7, "kv_per_tok": 0.03276}
    # one sized context is still a line: through the origin
    (single,) = site._memory(mem.iloc[[1]]).values()
    assert single["kv_base"] == 0.0 and single["kv_per_tok"] == 0.0585
    assert site._memory(pd.DataFrame()) == {}


def test_a_model_held_at_two_quants_is_two_models_on_the_page():
    df = pd.DataFrame({"model": ["a", "a", "b"], "quant": ["q4", "q8", "q4"]})
    assert site._model_ids(df) == {("a", "q4"): "a q4", ("a", "q8"): "a q8", ("b", "q4"): "b"}


# ------------------------------------------------------- the page's arithmetic
def _node(tmp_path, body: str) -> list:
    """The calculator and the fleet model run the way the page runs them: one JSON
    value back per `console.log` the body writes."""
    script = tmp_path / "run.js"
    assets = [(site.PKG / "assets" / name).read_text() for name in ("tasks.js", "fleet.js")]
    script.write_text("var window = {};\n" + "\n".join(assets) + "\n" + body)
    out = subprocess.run(
        ["node", str(script)],  # noqa: S603 — a file this test wrote
        capture_output=True,
        text=True,
        check=True,
    )
    return [json.loads(line) for line in out.stdout.strip().splitlines()]


def _line_curve(b: float, m: float, depth: int, w: int = 512) -> list[list[float]]:
    """A cumulative prefill curve whose chunks cost `b + m·d/1k` — what the sweep
    would have measured on a lane that really were linear, so the expectations
    below stay hand-checkable arithmetic."""
    curve, t = [[0, 0.0]], 0.0
    for k in range(depth // w):
        t += b + m * (w * k) / 1000
        curve.append([w * (k + 1), round(t, 1)])
    return curve


# One record shaped like the shelf's slowest CPU lanes: a curve measured to 7,168
# tokens and a decode ladder that stops at the same fill, on a model whose own
# context is far larger.
RECORD = {
    "curve": _line_curve(2451.831, 666.999, 7168),
    "pre_max": 7168,
    "ladder": [[0, 31.0], [2048, 28.0], [7168, 22.0]],
    "kv_max": 7168,
    "n_ctx_train": 262144,
}


@needs_node
def test_inside_the_measured_range_a_price_is_the_measurement(tmp_path):
    """A prompt that ends on a measured chunk boundary costs exactly what the sweep
    clocked there, and the reply is written at the ladder's own rates."""
    (v,) = _node(
        tmp_path,
        f"var rec = {json.dumps(RECORD)};\n"
        "console.log(JSON.stringify(window.taskMath.price(rec, {prompt: 2048, out: 64})));\n",
    )
    assert v["read_s"] == pytest.approx(RECORD["curve"][4][1] / 1000)
    assert v["est"] is False
    # 64 tokens just past a 2,048 fill, between the 28 and 22 tok/s rungs
    assert v["gen"] == pytest.approx(27.96, abs=0.01)
    assert v["total"] == pytest.approx(v["read_s"] + v["write_s"])
    assert v["read"] == pytest.approx(2048 / v["read_s"])


@needs_node
def test_past_the_measured_depth_the_curve_is_carried_on_and_marked(tmp_path):
    """Past the sweep's deepest point the measured curve is carried onward at its
    last segment's per-token cost — a prediction, not a fiction, and the optimistic
    side since the marginal cost was still rising. The generation rate is held at
    the deepest fill measured. The price is marked as the estimate it is."""
    (v,) = _node(
        tmp_path,
        f"var rec = {json.dumps(RECORD)};\n"
        "console.log(JSON.stringify(window.taskMath.price(rec, {prompt: 8192, out: 300})));\n",
    )
    assert v["est"] is True
    # 65.4 s measured to 7,168 tokens, then 1,024 more at the last segment's
    # 13.46 ms/token
    assert v["read_s"] == pytest.approx(79.2, abs=0.1)
    assert v["gen"] == pytest.approx(22.0, abs=0.1)  # the held rate


@needs_node
def test_a_price_is_refused_only_where_there_is_nothing_to_evaluate(tmp_path):
    """A task longer than the context the model was trained for is a fact about the
    model, not about the sweep's budget. So is a lane with no curve to read or no
    rate to write at. Everything else gets a number."""
    out = _node(
        tmp_path,
        f"var rec = {json.dumps(RECORD)}, price = window.taskMath.price;\n"
        "var small = {prompt: 100, out: 10};\n"
        "console.log(JSON.stringify(price(rec, {prompt: 262000, out: 300})));\n"
        "console.log(JSON.stringify(price(Object.assign({}, rec, {ladder: []}), small)));\n"
        "console.log(JSON.stringify(price(Object.assign({}, rec, {curve: []}), small)));\n"
        "console.log(JSON.stringify(price(rec, {prompt: 0, out: 0}) !== null));\n"
        "console.log(JSON.stringify(price(rec, {prompt: 200000, out: 300}) !== null));\n",
    )
    assert out == [None, None, None, True, True]


@needs_node
def test_a_cpu_held_to_fewer_threads_costs_its_ladders_own_ratio(tmp_path):
    """Reading and writing slow down by different amounts as threads are taken away,
    each by the ratio its own thread ladder measured. A lane is never priced wider
    than it ran, and a record with no ladder is priced as it ran."""
    scaling = {
        "prefill": {"threads": 16, "points": [[4, 100], [8, 180], [16, 300]]},
        "decode": {"threads": 16, "points": [[4, 20], [8, 24], [16, 25]]},
    }
    full, eight, six, wide, bare = _node(
        tmp_path,
        f"var rec = {json.dumps({**RECORD, 'scaling': scaling})}, bare = {json.dumps(RECORD)};\n"
        "var price = window.taskMath.price, task = {prompt: 2048, out: 64};\n"
        "[price(rec, task), price(rec, task, 8), price(rec, task, 6), price(rec, task, 32),\n"
        " price(bare, task, 6)].forEach(function (v) { console.log(JSON.stringify(v)); });\n",
    )
    assert eight["read_s"] == pytest.approx(full["read_s"] * 300 / 180)
    assert eight["write_s"] == pytest.approx(full["write_s"] * 25 / 24)
    # between two measured widths the rate is interpolated
    assert six["read_s"] == pytest.approx(full["read_s"] * 300 / 140)
    assert six["total"] == pytest.approx(six["read_s"] + six["write_s"])
    assert six["read"] == pytest.approx(2048 / six["read_s"])
    assert wide == full and bare == full


# One class, in seconds for one task: what its CPUs take as a typical machine has
# them, every time they can show, and its four machines on their GPUs. `none` is the
# same four with no GPU in use.
FLEET = """
var F = window.fleetMath;
var typical = [20], cpu = [10, 20, 40], gpu = [1.2, 1.6, 6.5, 14];
var none = [null, null, null, null];
function one(cpu, gpu, split, optimal, target, mid) {
  return F.fleet([{key: "a", share: 1, typical: mid || typical, cpu: cpu, gpu: gpu}],
                 split || [16, 68, 16], optimal || 15, target || 30)[0];
}
"""
PAD = 1.25  # a stretch reaches this far past its slowest and its fastest time


@needs_node
def test_a_tier_split_is_the_share_of_the_curve_in_each_third(tmp_path):
    """The split is honoured exactly, for a bell, a lean and a split that is heavy
    at both ends — and an even split is the flat curve."""
    splits = [[25, 50, 25], [44, 37, 19], [10, 30, 60], [40, 20, 40], [1, 1, 1]]
    out = _node(
        tmp_path,
        FLEET + f"{json.dumps(splits)}.forEach(function (s) {{\n"
        "  var c = F.tierCurve(s);\n"
        "  var low = F.betaBelow(1 / 3, c.a, c.b), upto_mid = F.betaBelow(2 / 3, c.a, c.b);\n"
        "  console.log(JSON.stringify([low, upto_mid, c.a, c.b]));\n"
        "});\n",
    )
    for split, (low, upto_mid, _a, _b) in zip(splits, out, strict=True):
        total = sum(split)
        assert low == pytest.approx(split[0] / total, abs=1e-4)
        assert 1 - upto_mid == pytest.approx(split[2] / total, abs=1e-4)
    assert out[-1][2:] == [pytest.approx(1, abs=1e-3), pytest.approx(1, abs=1e-3)]
    assert out[0][2] == pytest.approx(out[0][3], abs=1e-3)  # a symmetric split, a symmetric curve


@needs_node
def test_fleet_coverage_is_the_share_under_each_limit(tmp_path):
    (fleet,) = _node(
        tmp_path,
        FLEET
        + """
console.log(JSON.stringify(F.fleet([
  {key: "a", share: 30, typical: typical, cpu: cpu, gpu: gpu},
  {key: "b", share: 60, typical: typical, cpu: cpu, gpu: none},
  {key: "c", share: 10, typical: [], cpu: [], gpu: []},
], [16, 68, 16], 15, 30)));
""",
    )
    a, b, c = fleet
    assert [x["share"] for x in fleet] == [0.3, 0.6, 0.1]  # shares of a whole
    # every machine of the class on its GPUs is within the tolerable limit; some of
    # the class on its CPUs is past it
    assert a["cov"] == pytest.approx(1, abs=0.02) and 0.2 < b["cov"] < 0.95
    for x, times in ((a, [1.2, 1.6, 6.5, 14]), (b, [10, 20, 40])):
        assert 0 <= x["opt"] <= x["cov"] <= 1  # optimal is a part of covered
        # the stretch reaches a little past the slowest and the fastest machine
        assert x["fast"] == pytest.approx(min(times) / PAD)
        assert x["slow"] == pytest.approx(max(times) * PAD)
        # the curve's area is the class's share of the fleet, so areas compare
        area = sum(
            (p["d"] + q["d"]) / 2 * abs(math.log(p["t"] / q["t"]))
            for p, q in zip(x["curve"], x["curve"][1:], strict=False)
        )
        assert area == pytest.approx(x["share"], rel=0.02)
        # tiers: slowest first, their shares of the fleet add up to the class's
        assert sum(t["w"] for t in x["tiers"]) == pytest.approx(x["share"])
        assert x["tiers"][0]["typical"] > x["tiers"][1]["typical"] > x["tiers"][2]["typical"]
        assert x["tiers"][0]["frac"] <= x["tiers"][1]["frac"] <= x["tiers"][2]["frac"]
    # a class with no time to place is in the fleet and covers none of it
    assert c["curve"] is None and c["cov"] == 0 and c["opt"] == 0


@needs_node
def test_classes_on_the_same_cpus_are_one_curve_until_a_gpu_moves_one(tmp_path):
    same, moved = _node(
        tmp_path,
        FLEET
        + """
[none, gpu].forEach(function (g) {
  var f = F.fleet([{key: "a", share: 1, typical: typical, cpu: cpu, gpu: g},
                   {key: "b", share: 1, typical: typical, cpu: cpu, gpu: none}],
                  [16, 68, 16], 15, 30);
  console.log(JSON.stringify(f.map(function (c) { return [c.fast, c.slow, c.cov, c.opt]; })));
});
""",
    )
    assert same[0] == same[1] == moved[1]
    # the class with the GPUs reaches further toward fast and covers more
    assert moved[0][0] < moved[1][0] and moved[0][2] > moved[1][2]


@needs_node
def test_the_typical_machine_is_the_middle_of_the_stretch(tmp_path):
    """The stretch reaches the time furthest from the typical one and as far on the
    other side, so half of an even-handed class is faster than its typical machine
    however lopsided the measured times are."""
    out = _node(
        tmp_path,
        FLEET
        + """
[[10, 20, 40], [5, 20, 40], [20, 25, 80], [2, 40]].forEach(function (times) {
  var c = one(times, none, [16, 68, 16], 10, 20);
  console.log(JSON.stringify([c.fast, c.slow, c.cov]));
});
var lean = one([5, 20, 40], none, [16, 68, 16], 10, 20, [10, 40]);
console.log(JSON.stringify([lean.fast, lean.slow, lean.cov]));
""",
    )
    even, fast_outlier, slow_outlier, absent, mean = out
    assert even == [pytest.approx(10 / PAD), pytest.approx(40 * PAD), pytest.approx(0.5)]
    assert fast_outlier == [pytest.approx(5 / PAD), pytest.approx(80 * PAD), pytest.approx(0.5)]
    assert slow_outlier == [pytest.approx(5 / PAD), pytest.approx(80 * PAD), pytest.approx(0.5)]
    # the typical time need not be one of the measured ones
    assert absent == [pytest.approx(2 / PAD), pytest.approx(200 * PAD), pytest.approx(0.5)]
    # several typical times: their mean in log time, here 20 s again
    assert mean == pytest.approx(fast_outlier)


@needs_node
def test_a_class_leans_toward_its_own_machines_inside_the_shared_stretch(tmp_path):
    """Two classes that draw on the same CPUs cover the same stretch; the one whose
    own machines are the faster ones has more of its weight at the fast end, by a
    part of the gap that grows with how many machines it has."""
    out = _node(
        tmp_path,
        FLEET
        + """
var pool = [10, 40], lean = function (own) { return F.lean(pool, cpu, own); };
console.log(JSON.stringify([lean([10]), lean([10, 10, 10, 10]), lean([40]), lean(pool), lean([])]));
[0.1, 0, -0.1].forEach(function (by) {
  var c = F.fleet([{key: "a", share: 1, typical: typical, cpu: cpu, gpu: none, lean: by}],
                  [16, 68, 16], 10, 20)[0];
  console.log(JSON.stringify([c.fast, c.slow, c.cov, c.tiers.map(function (t) { return t.w; })]));
});
""",
    )
    leans, fast, even, slow = out
    # the pool is centred on 20 s and its stretch runs from 8 s to 50 s
    gap, span = math.log(20 / 10), math.log(50 / 8)
    assert leans[0] == pytest.approx(1 / 5 * gap / span)  # one machine counts for a fifth
    assert leans[1] == pytest.approx(4 / 8 * gap / span)  # four for a half
    assert leans[2] == pytest.approx(-leans[0])
    assert leans[3:] == [pytest.approx(0), 0]  # the pool itself does not lean
    # the stretch is the same; the weight inside it is not
    assert fast[:2] == even[:2] == slow[:2]
    assert fast[2] > even[2] > slow[2] and even[2] == pytest.approx(0.5)
    assert even[3] == [pytest.approx(w, abs=1e-4) for w in (0.16, 0.68, 0.16)]
    assert fast[3][2] > even[3][2] > slow[3][2] and fast[3][0] < even[3][0] < slow[3][0]
    for c in (fast, even, slow):
        assert sum(c[3]) == pytest.approx(1)


@needs_node
def test_a_gpu_only_pulls_an_end_of_the_stretch_in(tmp_path):
    cases = {
        "cpu": "one(cpu, none)",
        "some": "one(cpu, [null, 3, null, null])",  # one machine of four has a GPU
        "slower": "one(cpu, [60, 80, 70, 90])",  # every GPU is slower than every CPU
        "all": "one(cpu, gpu)",
        "mixed": "one(cpu, [3, 80, 6, 20])",  # the slowest GPU still loses to the slow CPUs
        "gpu_only": "one([], [null, 3, 12, null])",
        "single": "one([10], [null], null, null, null, [10])",
    }
    out = _node(
        tmp_path,
        FLEET
        + "".join(
            f"var c = {call}; console.log(JSON.stringify([c.fast, c.slow]));\n"
            for call in cases.values()
        ),
    )
    ends = dict(zip(cases, out, strict=True))
    assert ends["cpu"] == [pytest.approx(10 / PAD), pytest.approx(40 * PAD)]
    # the fastest machine has the GPU; the slowest is still on the slowest CPU
    assert ends["some"] == [pytest.approx(3 / PAD), pytest.approx(40 * PAD)]
    assert ends["slower"] == ends["cpu"]
    # the slowest machine has the slowest of each and runs on the better one
    assert ends["all"] == [pytest.approx(1.2 / PAD), pytest.approx(14 * PAD)]
    assert ends["mixed"] == [pytest.approx(3 / PAD), pytest.approx(40 * PAD)]
    # with no CPU time, the machines that have a GPU are the class
    assert ends["gpu_only"] == [pytest.approx(3 / PAD), pytest.approx(12 * PAD)]
    # one time is not a class: the stretch has a least length, about its middle
    assert ends["single"] == [pytest.approx(5), pytest.approx(20)]


@needs_node
def test_coverage_never_falls_when_a_backend_is_added(tmp_path):
    """A backend switched on gives some machines a GPU or a faster one. Neither end
    of the stretch moves later, so no part of the class gets slower — whatever the
    limit, the split or the class's lean."""
    out = _node(
        tmp_path,
        FLEET
        + """
var steps = [none, [50, null, null, null], [50, 3, null, null], [50, 3, null, 30],
             [50, 3, 9, 30], [20, 3, 9, 30], [2, 3, 9, 5], [1.2, 1.6, 6.5, 5]];
var worst = 0;
[[16, 68, 16], [60, 30, 10], [10, 30, 60], [40, 20, 40]].forEach(function (split) {
  [0.5, 1, 2, 5, 10, 20, 30, 45, 60, 120, 600].forEach(function (limit) {
    [0, 0.08, -0.08].forEach(function (by) {
    var c = steps.map(function (g) {
      return F.fleet([{key: "a", share: 1, typical: typical, cpu: cpu, gpu: g, lean: by}],
                     split, limit / 2, limit)[0];
    });
    for (var i = 1; i < c.length; i++) {
      worst = Math.max(worst, c[i - 1].cov - c[i].cov, c[i - 1].opt - c[i].opt);
    }
    });
  });
});
console.log(JSON.stringify(worst));
""",
    )
    assert out == [0]


# -------------------------------------------------------------- lane identity
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
    # windows pads the brand string to a fixed width
    (
        "AMD Ryzen 7 9800X3D 8-Core Processor           ",
        "AMD Ryzen 7 9800X3D 8-Core Processor",
        "cpu",
        "Ryzen 7 9800X3D",
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


def test_width_says_the_machine_when_the_lane_undercounts_it():
    """' 4t' alone reads as the size of the chip: an M1 lane running llama.cpp's
    macOS default (the 4-core performance cluster) looked like a 4-core machine.
    The suffix carries the machine wherever the width leaves cores idle, and
    stays terse where it does not."""
    assert site._width("cpu", 4, 4, 8) == " 4t (8C)"
    assert site._width("cpu", 8, 8, 8) == " 8t"
    assert site._width("cpu", 18, 6, 18) == " 6t/18t"
    assert site._width("cpu", 4, 4, None) == " 4t"  # a submission predating the field
    assert site._width("vulkan", 4, 4, 8) == ""


def test_lanes_stay_distinct_across_identical_machines():
    """Two of the same laptop must not pool into one lane."""
    df = pd.DataFrame(
        {
            "provider": ["vulkan:0", "vulkan:0"],
            "machine": ["nuc-a", "nuc-b"],
            "device": ["Intel(R) Graphics (MTL)"] * 2,
            "cpu": ["Intel(R) Core(TM) Ultra 5 125U"] * 2,
            "cpu_cores": [12, 12],
            "threads_batch": [12, 12],
            "threads_decode": [12, 12],
        }
    )
    lanes = site._with_lanes(df).lane
    assert lanes.nunique() == 2
    assert all("Core Ultra 5 125U iGPU · vulkan" in lane for lane in lanes)
