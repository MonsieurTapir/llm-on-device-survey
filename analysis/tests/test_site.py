"""The site builder renders one self-contained, data-first page — and hostile
submission strings can never reach an executable position: HTML context is
autoescaped, spec islands go through `tojson` (`<` → \\u003c), so a payload
that tries to terminate a script element survives only as inert text."""

import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from bench_analysis import site

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
    for anchor in ('data-spec="grid"', "Cost curves", "Machines",
                   "cmd-bash", "cmd-ps", 'button class="copy"'):
        assert anchor in h
    islands = re.findall(
        r'<script type="application/json"[^>]*>(.*?)</script>', h, re.S)
    assert islands, "no spec islands rendered"
    for body in islands:
        json.loads(body)  # strict — raises on NaN/Infinity


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
    ("Intel(R) Core(TM) Ultra 5 125U", "Intel(R) Core(TM) Ultra 5 125U", "cpu",
     "Core Ultra 5 125U", "CPU"),
    ("Intel(R) Graphics (MTL)", "Intel(R) Core(TM) Ultra 5 125U", "vulkan",
     "Core Ultra 5 125U iGPU", "integrated GPU"),
    ("AMD Ryzen 7 255 w/ Radeon 780M Graphics", "AMD Ryzen 7 255 w/ Radeon 780M Graphics",
     "cpu", "Ryzen 7 255", "CPU"),
    ("AMD Radeon Graphics (RADV PHOENIX)", "AMD Ryzen 7 255 w/ Radeon 780M Graphics",
     "vulkan", "Ryzen 7 255 iGPU", "integrated GPU"),
    ("AMD Radeon 760M Graphics (RADV PHOENIX)",
     "AMD Ryzen 5 PRO 230 w/ Radeon 760M Graphics", "vulkan",
     "Ryzen 5 PRO 230 iGPU", "integrated GPU"),
    ("AMD Ryzen 9 9950X 16-Core Processor", "AMD Ryzen 9 9950X 16-Core Processor", "cpu",
     "Ryzen 9 9950X", "CPU"),
    # the desktop APU's iGPU: RADV reports the CPU's own brand string
    ("AMD Ryzen 9 9950X 16-Core Processor (RADV RAPHAEL_MENDOCINO)",
     "AMD Ryzen 9 9950X 16-Core Processor", "vulkan", "Ryzen 9 9950X iGPU",
     "integrated GPU"),
    ("NVIDIA GeForce RTX 5080", "AMD Ryzen 9 9950X 16-Core Processor", "vulkan",
     "RTX 5080", "discrete GPU"),
    ("NVIDIA GeForce RTX 3090", "AMD Ryzen 9 5950X 16-Core Processor", "cuda",
     "RTX 3090", "discrete GPU"),
]


@pytest.mark.parametrize(("device", "cpu", "family", "chip", "klass"), DEVICES)
def test_lane_identity(device, cpu, family, chip, klass):
    assert site._lane_chip(device, cpu, family) == chip
    assert site._dev_class(device, cpu, family) == klass


def test_lanes_stay_distinct_across_identical_machines():
    """Two of the same laptop must not pool into one lane."""
    df = pd.DataFrame({
        "provider": ["vulkan:0", "vulkan:0"], "machine": ["nuc-a", "nuc-b"],
        "device": ["Intel(R) Graphics (MTL)"] * 2,
        "cpu": ["Intel(R) Core(TM) Ultra 5 125U"] * 2,
        "threads_batch": [12, 12], "threads_decode": [12, 12],
    })
    lanes = site._with_lanes(df).lane
    assert lanes.nunique() == 2
    assert all("Core Ultra 5 125U iGPU · vulkan" in lane for lane in lanes)
