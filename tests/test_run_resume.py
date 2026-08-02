"""`survey run` checkpoint/resume.

An interrupted run leaves `<backend>-checkpoint.json.gz` in `--out`; the next
invocation must skip the cells it holds, re-measure only the rest, and treat a
mid-flight model as already page-cache-warm. A checkpoint from a different
experiment must refuse to resume, and a completed run must delete the file.
The spawn layer is stubbed with the aggregation fixtures — no exe involved.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_aggregate import _probe_events, _raw, _run_events, _samples, _sweep_events

from survey.commands.run import _read_checkpoint, _write_checkpoint, cmd_run
from survey.registry import Lane, Variant
from survey.spawn import SpawnResult


def _expect() -> dict:
    """What this run would measure — the fields a checkpoint must match."""
    raw = _raw([])
    return {
        "schema_version": "1",
        "backend": "llamacpp",
        "machine": raw["machine"],
        "sampling": raw["sampling"],
        "job_spawns": 2,
        "job_iters": 5,
    }


def _checkpoint_doc(**over) -> dict:
    doc = {
        **_expect(),
        "probes": [],
        "cells": [],
        "resume": {"touched": [], "cold_load": {}, "cold_used": [], "overruns": []},
    }
    doc.update(over)
    return doc


def test_read_checkpoint_missing_and_unreadable(tmp_path):
    path = tmp_path / "llamacpp-checkpoint.json.gz"
    assert _read_checkpoint(path, _expect(), lambda: None) is None
    path.write_bytes(b"not gzip at all")
    assert _read_checkpoint(path, _expect(), lambda: None) is None


def test_read_checkpoint_round_trip(tmp_path):
    path = tmp_path / "llamacpp-checkpoint.json.gz"
    _write_checkpoint(path, _checkpoint_doc())
    assert _read_checkpoint(path, _expect(), lambda: None) == _checkpoint_doc()
    assert not path.with_name(path.name + ".tmp").exists()


def test_read_checkpoint_refuses_a_different_experiment(tmp_path):
    path = tmp_path / "llamacpp-checkpoint.json.gz"
    _write_checkpoint(path, _checkpoint_doc(job_iters=3))
    with pytest.raises(SystemExit, match="(?s)job_iters.*--fresh") as err:
        _read_checkpoint(path, _expect(), lambda: None)
    assert "refusing to resume" in str(err.value)

    _write_checkpoint(path, _checkpoint_doc(machine={**_expect()["machine"], "cpu": "other"}))
    with pytest.raises(SystemExit, match="machine.cpu"):
        _read_checkpoint(path, _expect(), lambda: None)


def test_read_checkpoint_refuses_a_rebuilt_stack(tmp_path):
    """The stack identity travels in the traces; a checkpoint whose cells were
    measured by a different exe build must not be extended."""
    path = tmp_path / "llamacpp-checkpoint.json.gz"
    cell = {"model": "demo", "quant": "q4", "provider": "cuda:0", "sweep": None, "job": None}
    probe = {"provider": "cuda:0", "trace": {"events": _probe_events(), "samples": []}}
    _write_checkpoint(path, _checkpoint_doc(probes=[probe], cells=[cell]))
    # _probe_events carries versions == {"threads": ...} only → identity is {}.
    assert _read_checkpoint(path, _expect(), lambda: {}) is not None
    with pytest.raises(SystemExit, match="versions"):
        _read_checkpoint(path, _expect(), lambda: {"llama_cpp_commit": "def456"})


# --- end-to-end: interrupt after one of two cells, resume, finish -----------


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        backend="llamacpp",
        models=tmp_path / "models",
        tasks=tmp_path / "tasks",
        out=tmp_path / "out",
        iters=5,
        spawns=2,
        max_ms=0,
        sweep_ms=0,
        backstop_ms=120000,
        providers=None,
        model=None,
        machine=None,
        fresh=False,
    )


@pytest.fixture
def stubbed_world(monkeypatch, tmp_path):
    """cmd_run's world with the exe layer replaced: two variants on one lane,
    spawns answering with the aggregation fixtures, call log recorded."""
    calls = {"probe": [], "sweep": [], "run": []}
    variants = [
        Variant(model="demo", model_path=tmp_path / "a.gguf", quant="q4"),
        Variant(model="demo2", model_path=tmp_path / "b.gguf", quant="q4"),
    ]
    tasks = [
        SimpleNamespace(role="gate", name="brain-check", spec={}),
        SimpleNamespace(role="timed", name="t", spec={}),
    ]
    monkeypatch.setattr(
        "survey.config.load_backend",
        lambda key: SimpleNamespace(key="llamacpp", cmd=["/nonexistent-exe"]),
    )
    monkeypatch.setattr("survey.commands.run.load_tasks", lambda d: tasks)
    monkeypatch.setattr("survey.registry.variants", lambda d, k: variants)
    monkeypatch.setattr(
        "survey.registry.providers", lambda b, p: [Lane(id="cuda:0", description="stub GPU")]
    )
    monkeypatch.setattr("survey.machine.info", lambda name=None: _raw([])["machine"])
    monkeypatch.setattr("survey.commands.run._exe_versions", lambda cmd: {})

    def fake_probe(cmd, **kw):
        calls["probe"].append(kw)
        return SpawnResult(events=_probe_events(), samples=[], cold=False, error=None)

    def fake_sweep(cmd, **kw):
        calls["sweep"].append(kw)
        return SpawnResult(events=_sweep_events(), samples=[], cold=kw["cold"], error=None)

    def fake_run(cmd, **kw):
        calls["run"].append(kw)
        return SpawnResult(events=_run_events(), samples=_samples(), cold=False, error=None)

    monkeypatch.setattr("survey.spawn.probe", fake_probe)
    monkeypatch.setattr("survey.spawn.sweep", fake_sweep)
    monkeypatch.setattr("survey.spawn.run", fake_run)
    # `base` snapshots the run box's sampling sources; pin them so the resumed
    # run always matches the checkpoint, whatever box the tests run on.
    monkeypatch.setattr("survey.sampling.NVML_AVAILABLE", True)
    monkeypatch.setattr("survey.sampling.DRM_AVAILABLE", False)
    return calls


def test_interrupted_run_resumes_without_remeasuring(stubbed_world, monkeypatch, tmp_path):
    calls = stubbed_world
    args = _args(tmp_path)
    ckpt_path = args.out / "llamacpp-checkpoint.json.gz"

    # First invocation dies inside cell 2's sweep, after cell 1 completed.
    real_sweep = __import__("survey.spawn", fromlist=["sweep"]).sweep

    def dying_sweep(cmd, **kw):
        if kw["model_path"].name == "b.gguf":
            raise KeyboardInterrupt
        return real_sweep(cmd, **kw)

    monkeypatch.setattr("survey.spawn.sweep", dying_sweep)
    with pytest.raises(KeyboardInterrupt):
        cmd_run(args)
    assert ckpt_path.exists()
    with gzip.open(ckpt_path, "rt") as fh:
        ckpt = json.load(fh)
    assert [c["model"] for c in ckpt["cells"]] == ["demo"]
    # The interrupted cell's model counts as touched — its load began.
    assert sorted(Path(p).name for p in ckpt["resume"]["touched"]) == ["a.gguf", "b.gguf"]

    # Second invocation: only cell 2 is measured, and not as a cold load.
    monkeypatch.setattr("survey.spawn.sweep", real_sweep)
    for log in calls.values():
        log.clear()
    cmd_run(args)
    assert [kw["model_path"].name for kw in calls["sweep"]] == ["b.gguf"]
    assert calls["sweep"][0]["cold"] is False
    assert calls["probe"] == []  # cuda:0's ceiling probe rode along in the checkpoint

    assert not ckpt_path.exists()
    results = json.loads((args.out / "llamacpp-results.json").read_text())
    assert {r["model"] for r in results["runs"]} == {"demo", "demo2"}


def test_fresh_discards_the_checkpoint(stubbed_world, tmp_path):
    calls = stubbed_world
    args = _args(tmp_path)
    args.out.mkdir(parents=True)
    ckpt_path = args.out / "llamacpp-checkpoint.json.gz"
    # A checkpoint that would otherwise refuse (different job shape) is simply
    # ignored — and overwritten — under --fresh.
    _write_checkpoint(ckpt_path, _checkpoint_doc(job_iters=3))
    args.fresh = True
    cmd_run(args)
    assert len(calls["sweep"]) == 2
    assert not ckpt_path.exists()
