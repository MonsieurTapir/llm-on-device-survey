# analysis — read the submissions, build the report

A package of the root uv project, behind its own dependency group on
purpose: the survey tool ships to every benchmark box and stays lean, while
analysis pulls in pandas/jinja and runs only where you crunch
numbers. It is a pure consumer of
`results.schema.json` — it reads what the survey tool writes and never
touches the contract.

```sh
uv sync --group analysis

# build the report — one self-contained HTML, viewable offline:
uv run python -m analysis.site   # → site/index.html
```

`--published <dir>` points it at a different submissions folder, `--out`
elsewhere. Nothing is fetched, at build time or when the page is read.

## The page

One question: how much of a fleet of machines finishes a task in time. A rail
holds the model, the task (a prompt to read, a reply to write), two time limits
(optimal, tolerable), the backends in use and the fleet's composition — a share
per class of machine (PC with GPU, PC without GPU, Apple Silicon Mac) and a low /
mid / high split inside each class. Two numbers answer it — the share of the
fleet within the limits, and the memory the task reserves — and three panels
show what is behind them:

- **Coverage** — each class as a distribution over whole-task time, with the
  measured machines on it. A class's area is its share of the fleet; the area
  within a limit is the share of the fleet within it.
- **Compare** — chosen machines side by side: the task on each of their
  devices, how the time grows with the task, and the numbers behind it.
- **Data** — every device as a searchable, sortable table.

A machine is priced on its fastest device among the backends in use. A class is
a fixed stretch of time anchored on its measured machines, and the split spreads
the class over it — `assets/fleet.js` states the model in full.

## Loaders

Five loaders fan over `root/**/*-results.json`; one subdir per machine, the
subdir name becoming the machine label:

- `load_results` — the validation job: one row per `(machine, backend,
  model, quant, provider)`, `[p50, max]` stats exploded into `<name>_p50` /
  `<name>_max`, geometry scalars as `geo_*`, machine memory config as
  `ram_*`. Gaps are visible, not absent: a null stat (VRAM on a CPU EP) is
  NaN, never 0, and a cell that produced no timing still gets a row, flagged
  by `status` (`ok` / `too_slow` / `errored` / `unhealthy`).
- `load_sweeps` — one row per measured sweep point (prefill ms vs tokens,
  decode tok/s vs KV fill), each with its min–max spread and repeat count.
- `load_memory` — the memory cost curve: one row per allocator context point
  (`n_ctx`, pooled `weights_mb` / `kv_mb` / `compute_mb`) from the sweep's
  geometry.
- `load_thread_scaling` — one row per thread-ladder point (`phase` ×
  `threads` → `tps`): what an intra-op thread buys each phase on a CPU lane.
  The points are the whole frame — no fitted asymptotes are carried, because
  the ladder ends at the widest width the lane runs and an asymptote fitted
  there rests on nothing.
- `load_probes` — one row per device-ceiling point (GEMM TFLOP/s, copy GB/s).

Every loader pins the results `schema_version` — a file at another version is
a loud error, not a silently-misaligned frame.

## Layout

- `analysis/load.py` — results JSON → tidy pandas frames (the loaders above).
- `analysis/site.py` — everything else on the build side: derives lanes, device
  classes and machine classes, gathers the measured cost curves into one data
  pack, and renders the page around it.
- `templates/report.html.j2` — the page's markup, with the pack as a JSON
  island.
- `assets/tasks.js` — the task calculator: one task on one device, priced from
  its measured curves.
- `assets/fleet.js` — the fleet model: classes, tiers and coverage.
- `assets/report.js` — the page itself: the controls' state and the three
  panels drawn from it.
- `assets/report.css` — every colour as custom properties, light and dark.

`load.py` and `site.py` are the tested modules (`uv run pytest tests`); the
template is exercised through `site.build`, and `tasks.js` and `fleet.js` are
run under node where it is installed, all in `tests/test_site.py`. Lint with
`uv run ruff check`.

## Theming

The page follows the reader's OS light/dark preference with no rebuild: every
colour is a CSS custom property, and the charts are inline SVG that wears them.
Hue means one thing — the class of machine — so the controls are neutral. The
three class colours have a light and a dark step each, chosen for their surface.
