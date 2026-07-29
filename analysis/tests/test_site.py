"""The site builder renders one self-contained, data-first page — and hostile
submission strings can never reach an executable position: HTML context is
autoescaped, spec islands go through `tojson` (`<` → \\u003c), so a payload
that tries to terminate a script element survives only as inert text."""

import json
import re
import shutil
from pathlib import Path

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
    doc_path = next(poisoned.glob("*/ggml-results.json"))
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
