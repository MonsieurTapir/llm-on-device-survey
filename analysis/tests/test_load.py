import json
import math
from pathlib import Path

import pytest

from bench_analysis import (
    load_memory,
    load_probes,
    load_results,
    load_sweeps,
    load_thread_scaling,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_merges_machines_by_directory():
    df = load_results(FIXTURES)
    assert set(df["machine"]) == {"3090-box", "m1-max"}  # label = subdir name
    assert set(df["backend"]) == {"llamacpp"}


def test_one_row_per_run_with_job_status():
    df = load_results(FIXTURES)
    counts = df["status"].value_counts().to_dict()
    assert counts["ok"] == 3  # m1-max scored both its lanes, 3090-box one
    assert counts["too_slow"] == 1  # 3090-box gemma4-E4B
    assert counts["errored"] == 1  # 3090-box gemma4-E2B
    assert counts["unhealthy"] == 1  # m1-max qwen3-4B failed its brain-check
    # the sweep verdict rides along even when the job scored
    assert set(df["sweep_status"]) == {"ok", "too_slow", "errored", "skipped"}


def test_machine_memory_and_geometry_ride_along():
    df = load_results(FIXTURES)
    box = df[df.machine == "3090-box"].iloc[0]
    assert box["ram_gb"] == 64.0 and box["ram_channels"] == 2 and box["ram_mts"] == 3200
    ok = df[(df.machine == "3090-box") & (df.status == "ok")].iloc[0]
    assert ok["geo_n_params"] == 1_000_000 and ok["geo_n_layer"] == 2
    mac = df[df.machine == "m1-max"].iloc[0]
    assert math.isnan(mac["ram_channels"]) or mac["ram_channels"] is None


def test_stats_explode_into_p50_max():
    df = load_results(FIXTURES)
    row = df[(df.machine == "3090-box") & (df.status == "ok")].iloc[0]
    assert row["decode_tps_p50"] == 80 and row["decode_tps_max"] == 75
    assert row["decode_vram_peak_mb_p50"] == 1700
    assert row["decode_vram_sustained_mb_p50"] == 1650


def test_memory_curve_loads_long_with_pooled_buffers():
    mem = load_memory(FIXTURES)
    box = mem[mem.machine == "3090-box"].sort_values("n_ctx")
    assert list(box.n_ctx) == [512, 2048, 8192]
    assert list(box.weights_mb) == [600.0] * 3  # 500 MB device + 100 MB host, pooled
    assert list(box.kv_mb) == [37.5, 150.0, 600.0]  # grows with context
    assert list(box.compute_mb) == [50.0] * 3
    # the lane's thread widths ride along, so these rows label their lane the same
    # way the results and sweep frames do
    assert list(box.threads_batch) == [16] * 3
    assert list(box.threads_decode) == [16] * 3
    # runs without the curve contribute no rows — absent, not invented
    assert mem[mem.machine == "m1-max"].empty


def test_null_stats_become_nan_not_zero():
    df = load_results(FIXTURES)
    row = df[(df.machine == "m1-max") & (df.status == "ok")].iloc[0]
    assert math.isnan(row["decode_vram_peak_mb_p50"])  # unified -> null -> NaN
    assert math.isnan(row["prefill_tps_p50"])


def test_sweep_chunks_load_long_with_cumulative_ttft():
    sweeps = load_sweeps(FIXTURES)
    pre = sweeps[(sweeps.machine == "3090-box") & (sweeps.kind == "prefill")
                 & (sweeps.sweep_status == "ok")]
    # Depths stay contiguous through the subdivided chunk (512-wide, then a
    # half-width pair at 1024, then 512-wide again), so the ladder has a 256 step
    # in it and the cumulative sum is still TTFT at each depth.
    assert list(pre.tokens) == [512, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096]
    running, expected = 0.0, []
    for ms in pre.chunk_ms:
        running += ms
        expected.append(round(running, 2))
    assert list(pre.ttft_ms) == expected
    dec = sweeps[(sweeps.machine == "m1-max") & (sweeps.kind == "decode")
                 & (sweeps.provider == "mtl")]
    assert list(dec.kv_fill) == [0, 2048] and list(dec.tps_p50) == [80.0, 75.0]


def test_sweep_derived_parameters_ride_along():
    """The cost function reduced to parameters, plus the first-launch compile —
    the columns a cross-machine comparison actually reads."""
    df = load_results(FIXTURES)
    cuda = df[(df.machine == "3090-box") & (df.model == "qwen3-4B")].iloc[0]
    assert cuda["fit_slope_ms_per_1k"] > 0  # the attention term
    assert cuda["fit_intercept_ms"] > 0  # the per-dispatch term
    assert cuda["fit_r2"] > 0.99
    assert cuda["ubatch_penalty_pct_max"] > 0  # narrower dispatches cost more
    assert cuda["shader_cache"] == "redirected"  # so the compile number compares
    assert cuda["shader_bytes"] > 0

    # Too few full-width points to fit a line: the parameters are absent, not zero.
    short = df[(df.machine == "3090-box") & (df.model == "gemma4-E4B")].iloc[0]
    assert math.isnan(short["fit_slope_ms_per_1k"])
    assert math.isnan(short["ubatch_penalty_pct_max"])

    # macOS can't pin Metal's cache, so its warmup span doesn't compare across
    # machines even though it is measured.
    mac = df[(df.machine == "m1-max") & (df.model == "gemma4-E2B")].iloc[0]
    assert mac["shader_cache"] == "unavailable"
    assert mac["shader_warmup_ms"] == 990.0


def test_partial_sweep_points_survive_bad_status():
    sweeps = load_sweeps(FIXTURES)
    partial = sweeps[sweeps.sweep_status == "too_slow"]
    assert len(partial) == 1 and partial.iloc[0]["tokens"] == 512


def test_thread_ladder_loads_one_row_per_phase_and_width():
    """The ladder is a CPU-lane measurement: each phase walks the lane's own width and
    its halves, on its own small work unit. Prefill is timed from an empty cache, so it
    has no fill; decode is timed at a fill the pass already primed, so it does."""
    thr = load_thread_scaling(FIXTURES)
    assert set(thr.provider) == {"cpu:0"}  # a GPU lane measures no ladder
    assert set(thr.machine) == {"m1-max"}
    for phase, tokens in (("prefill", 128), ("decode", 16)):
        g = thr[thr.phase == phase]
        assert sorted(g.threads) == [2, 4, 8]  # the width the lane runs, and halves
        assert set(g.tokens) == {tokens}

    pre = thr[thr.phase == "prefill"].sort_values("threads", ascending=False)
    assert pre.kv_fill.isna().all()  # measured from an empty cache
    assert list(pre.ms) == [212.4, 378.6, 719.5]  # narrower is slower
    dec = thr[thr.phase == "decode"].sort_values("threads", ascending=False)
    assert set(dec.kv_fill) == {2048}
    assert list(dec.tps) == [22.1, 20.4, 15.2]
    assert dec.ms.isna().all()  # a decode point is a rate, not a duration
    # the lane's widths ride along, so these rows label their lane like every frame
    assert set(thr.threads_batch) == {8} and set(thr.threads_decode) == {8}


def test_thread_fits_ride_along_with_the_width_they_size():
    """The results frame carries the two fits as parameters, plus the one derived
    number a power-management caller acts on: the width that reaches 90% of peak
    decode, which is the saturating fit inverted at a tenth."""
    df = load_results(FIXTURES)
    cpu = df[(df.machine == "m1-max") & (df.provider == "cpu:0")].iloc[0]
    assert cpu["thr_prefill_floor_ms"] == 41.95  # what no width removes
    assert cpu["thr_prefill_scaled_ms"] == 1353.886  # what threads divide
    assert cpu["thr_prefill_floor_pct"] == 3.01  # so prefill keeps paying for cores
    assert cpu["thr_decode_rate_max_tps"] == 22.439
    assert cpu["thr_decode_threads_p90"] == round(
        cpu["thr_decode_threads_scale"] * math.log(10), 2)
    assert cpu["thr_decode_threads_p90"] < cpu["threads_decode"]  # saturates early

    mtl = df[(df.machine == "m1-max") & (df.provider == "mtl")].iloc[0]
    assert math.isnan(mtl["thr_prefill_floor_ms"])  # absent, not zero
    assert math.isnan(mtl["thr_decode_threads_p90"])


def test_probes_load_with_throughputs():
    probes = load_probes(FIXTURES)
    gemm = probes[(probes.machine == "3090-box") & (probes.kind == "gemm")].iloc[0]
    assert gemm["tflops"] == 55.2
    d2d = probes[(probes.machine == "m1-max") & (probes.kind == "d2d")].iloc[0]
    assert d2d["gbs"] == 350.0


def test_flat_file_labels_by_host_not_gpu(tmp_path):
    """A results file straight under the root (no machine subdir) is labelled by its
    machine `host`, not by slugging the GPU."""
    doc = {
        "schema_version": "2",
        "backend": "llamacpp",
        "machine": {
            "host": "leaf-desktop", "os": "linux", "cpu": "x",
            "cpu_cores": 16, "cpu_threads": 32, "gpus": ["NVIDIA RTX 5090"],
            "memory": {"total_gb": 32.0, "channels": None, "configured_mts": None,
                       "rated_mts": None, "rank": None, "dimms": None},
        },
        "job_spawns": 2,
        "probes": [],
        "runs": [],
    }
    (tmp_path / "llamacpp-results.json").write_text(json.dumps(doc))
    assert load_results(tmp_path).empty  # no runs, but it loaded without error
    doc["runs"] = [{
        "provider": "cuda", "device": "cuda:0", "model": "m", "quant": "q4",
        "healthy": False, "unhealthy_reason": "x", "vram_method": "nvml",
        "threads": None,  # nothing ran, so no width was observed
        "geometry": None,
        "sweep": {"status": "skipped", "prefill": [], "decode": [], "fit": None,
                  "ubatch": [], "warmup_ms": None, "shader_bytes": None,
                  "shader_cache": "redirected"},
        "job": {"status": "skipped", "task": "summarize-large"},
    }]
    (tmp_path / "llamacpp-results.json").write_text(json.dumps(doc))
    assert set(load_results(tmp_path)["machine"]) == {"leaf-desktop"}


def test_schema_version_mismatch_is_loud(tmp_path):
    (tmp_path / "llamacpp-results.json").write_text(
        json.dumps({"schema_version": "99", "backend": "llamacpp",
                    "machine": {"os": "linux", "cpu": "x", "gpus": []},
                    "iters": 1, "spawns": 1, "runs": []})
    )
    for loader in (load_results, load_sweeps, load_probes, load_memory,
                   load_thread_scaling):
        with pytest.raises(ValueError, match="schema_version"):
            loader(tmp_path)


def test_empty_dir_returns_empty_frame(tmp_path):
    assert load_results(tmp_path).empty
    assert load_sweeps(tmp_path).empty
    assert load_probes(tmp_path).empty
    assert load_memory(tmp_path).empty
    assert load_thread_scaling(tmp_path).empty
