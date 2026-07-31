# analysis — read the submissions, build the report

A package of the root uv project, behind its own dependency group on
purpose: the survey tool ships to every benchmark box and stays lean, while
analysis pulls in pandas/altair/jinja and runs only where you crunch
numbers. It is a pure consumer of
`results.schema.json` — it reads what the survey tool writes and never
touches the contract.

```sh
uv sync --group analysis

# build the report — one self-contained HTML, viewable offline:
uv run python -m analysis.site   # → site/index.html
```

`--published <dir>` points it at a different submissions folder, `--out`
elsewhere. The vega libraries are fetched once at build time (versions
pinned to the installed altair) and cached under `third_party/vega/` —
nothing is fetched when the page is read.

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
- `analysis/site.py` — everything else: derives lanes and device classes,
  prices tasks off the measured cost curves, computes the template context,
  builds every chart as an altair/vega-lite spec, and renders the page.
- `templates/report.html.j2` — the one-page report: section per question,
  charts as JSON islands, the task calculator inline.
- `assets/report.css` — every colour as custom properties, light and dark.
- `assets/report.js` — mounts the chart islands, swaps each spec's series
  colours for their dark-surface steps (the `#palette-dark-map` island) and
  re-embeds on scheme change.
- `assets/tasks.js` — the task calculator, priced from the lanes' measured
  curves.

`load.py` and `site.py` are the tested modules (`uv run pytest`); templates
and assets are exercised through `site.build` in `tests/test_site.py`. Lint
with `uv run ruff check`.

## Theming in a static export

The page follows the reader's OS light/dark preference with no rebuild:
chrome colours are CSS custom properties, and charts render on transparent
backgrounds. Series colours can't come from CSS (vega specs carry hex), so
specs are built with the light palette and `report.js` rewrites each colour
to its dark step when mounting on a dark page — both palettes are chosen for
their surface rather than one compromise holding up on both.
