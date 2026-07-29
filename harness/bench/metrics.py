"""Events object → timing numbers. Pure: no I/O, no spawning, no aggregation.

One events object is one (model, provider, task) run with K timed iterations
. This module reduces each iteration to the four timing metrics and
reads the once-per-process load spans. Aggregation across iterations and spawns
(→ [p50, max]) is aggregate.py's job; here we only ever look at a single run.

Metric definitions, all in milliseconds / tokens-per-second:
  • ttft        — prefill start → first decoded token. The primary prefill-side
                  metric; all backends measure it identically.
  • prefill_tps — prompt tokens / prefill time. None when an iteration carries no
                  isolable prefill event (both backends isolate prefill).
  • decode_tps  — steady-state over decode steps 2..N (step 1's latency lives in
                  ttft).
  • completion  — prefill start → last decoded token, over the whole iteration.

A multi-turn iteration (several prefill/decode/turn-end triples — e.g. the
brain-check) pools token-weighted across its turns.
"""

from __future__ import annotations

NS_PER_MS = 1e6
NS_PER_S = 1e9


def _of_type(events: list[dict], kind: str) -> list[dict]:
    return [e for e in events if e["type"] == kind]


def iteration_timings(iteration: dict) -> dict:
    """Reduce one timed iteration to {ttft_ms, prefill_tps, decode_tps, completion_ms}.

    prefill_tps and decode_tps are None when the iteration carries no isolable
    prefill / no steady-state decode window (e.g. a single-token decode).
    """
    events = iteration["events"]
    prefills = _of_type(events, "prefill")
    decodes = _of_type(events, "decode")
    if not decodes:
        raise ValueError("iteration has no decode events")

    # Generation starts at the first prefill if the backend isolates it, else at
    # the first decode (defensive: a backend that fuses prompt + first token).
    gen_start = prefills[0]["start_ns"] if prefills else decodes[0]["start_ns"]
    first_token = decodes[0]["token_ns"][0]

    prefill_ns = sum(p["end_ns"] - p["start_ns"] for p in prefills)
    prefill_tokens = sum(p["tokens_count"] for p in prefills)
    prefill_tps = prefill_tokens / (prefill_ns / NS_PER_S) if prefills and prefill_ns else None

    # Steady state = every decoded token except the first of each decode window
    # (token_ns[0] is the TTFT-side token). Pool token-weighted across turns.
    steady_tokens = sum(len(d["token_ns"]) - 1 for d in decodes)
    steady_ns = sum(d["token_ns"][-1] - d["token_ns"][0] for d in decodes)
    decode_tps = steady_tokens / (steady_ns / NS_PER_S) if steady_ns else None

    return {
        "ttft_ms": (first_token - gen_start) / NS_PER_MS,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "completion_ms": (decodes[-1]["end_ns"] - gen_start) / NS_PER_MS,
    }


def timing_samples(events: dict) -> list[dict]:
    """One timing dict per iteration in the events object (the K samples per spawn)."""
    return [iteration_timings(it) for it in events["iterations"]]


def load_ms(events: dict) -> float:
    """User-perceived cold load = model-load + context-init (warmup excluded — see
    `load_components`). Used only for the cold first-touch number."""
    spans = {e["type"]: (e["end_ns"] - e["start_ns"]) / NS_PER_MS for e in events["load"]}
    return spans.get("model-load", 0.0) + spans.get("context-init", 0.0)


def load_components(events: dict) -> dict:
    """Split the per-process load into its three phases (ms): weight load,
    context/KV allocation, and the one-time kernel warmup. A backend that doesn't
    emit a given span contributes 0.0 for it, so the three always sum to the full
    pre-first-token cost."""
    spans = {e["type"]: (e["end_ns"] - e["start_ns"]) / NS_PER_MS for e in events["load"]}
    return {
        "model_load_ms": spans.get("model-load", 0.0),
        "context_init_ms": spans.get("context-init", 0.0),
        "warmup_ms": spans.get("warmup", 0.0),
    }


def _full_width(chunks: list[dict]) -> int:
    """The pass's full dispatch width — the widest chunk it ingested."""
    return max((c["tokens"] for c in chunks), default=0)


def prefill_fit(chunks: list[dict]) -> dict | None:
    """Least-squares line through the full-width chunks: cost of one dispatch as a
    function of how much KV precedes it.

    These two numbers *are* the prefill cost function, which is what the survey is
    for. `intercept_ms` is the depth-independent per-dispatch term (the matmuls);
    `slope_ms_per_1k` is the attention term, the marginal cost of another 1k of
    context. Integrating the line is where a time-to-first-token estimate at an
    arbitrary prompt length comes from, rather than only at the depths measured.

    `r2` and `resid_max_pct` are the quality indicators. The prefill chunks are the
    one measurement family with no repeats — a single pass, nothing measured twice —
    so scatter around a fit is the only available signal that a point was disturbed
    rather than real. Costs nothing extra: it reads the points already collected.

    None below three points, where a line says nothing."""
    pts = [(c["context"], c["ms"]) for c in chunks if c["tokens"] == _full_width(chunks)]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    resid = [y - (intercept + slope * x) for x, y in pts]
    ss_res = sum(r * r for r in resid)
    return {
        "width": _full_width(chunks),
        "intercept_ms": round(intercept, 3),
        "slope_ms_per_1k": round(slope * 1000, 4),
        "r2": round(1 - ss_res / ss_tot, 5) if ss_tot else None,
        "resid_max_pct": round(max(abs(r) for r in resid) / my * 100, 2) if my else None,
        "n_points": n,
    }


def ubatch_points(chunks: list[dict]) -> list[dict]:
    """Cost of ingesting a full chunk as several narrower dispatches, against what
    the fit says one full-width dispatch costs at the same depth.

    The sweep splits a couple of its chunks into half-width dispatches: same
    tokens, same depth reached, so the envelope is unchanged and this is close to
    free. The ratio says whether the silicon cares about micro-batch width — a wide
    GPU does, a CPU with a handful of cores does not — which is the one thing a
    single fixed operating point otherwise can't tell you.

    It is an indicator, not a second operating point: the dispatch is narrower but
    the context is still open at the full n_ubatch, and the pair pays two
    synchronizations where one wide dispatch pays one. Expect it to overstate a
    true narrower-n_ubatch run somewhat."""
    fit = prefill_fit(chunks)
    if not fit:
        return []
    full = fit["width"]
    out: list[dict] = []
    group: list[dict] = []
    for c in chunks:
        if c["tokens"] == full:
            group = []
            continue
        group.append(c)
        if sum(g["tokens"] for g in group) < full:
            continue
        depth = group[0]["context"]
        parts_ms = sum(g["ms"] for g in group)
        full_ms = fit["intercept_ms"] + fit["slope_ms_per_1k"] * depth / 1000
        out.append({
            "context": depth,
            "width": group[0]["tokens"],
            "n_parts": len(group),
            "parts_ms": round(parts_ms, 2),
            "full_ms_fitted": round(full_ms, 2),
            "penalty_pct": round((parts_ms / full_ms - 1) * 100, 2) if full_ms > 0 else None,
        })
        group = []
    return out


def peak_context(events: dict) -> int:
    """Longest running sequence (context_size + tokens_count) over all prefill/
    decode events — what must stay within the task's max_context_length. The exe
    is the only actor that tokenizes, so this is the real, per-backend length."""
    peak = 0
    for it in events["iterations"]:
        for e in it["events"]:
            if e["type"] in ("prefill", "decode"):
                peak = max(peak, e["context_size"] + e["tokens_count"])
    return peak


def completions(events: dict) -> list[str]:
    """Decoded generated text of every turn-end, across all iterations — the
    eyeball-it-isn't-garbage sample carried up into results."""
    return [
        e["completion"]
        for it in events["iterations"]
        for e in it["events"]
        if e["type"] == "turn-end"
    ]
