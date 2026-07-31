# Published submissions

Shared, version-controlled benchmark runs — the baseline the report builder
(`analysis.site`) reads by default. Local runs land under `results/`,
which is gitignored; this folder is the tracked shelf.

A submission measures a machine's cost function: per provider a device ceiling
probe (GEMM, copy bandwidth), and per (model, quant, provider) the model
geometry as the runtime reports it, prefill/decode sweeps to 8k context, and
one real validation job. Run with `sudo` where possible so the installed
memory config (dmidecode) lands in the machine block — it is the source of the
machine's nominal bandwidth.

## Submitting

1. Run the benchmark locally: `uv run survey run --backend <key> --out results/<my-box>`.
2. Stage it as a submission:

   ```sh
   uv run survey publish results/<my-box> --name <my-box>
   ```

   This validates each `<backend>-results.json` against the contract, copies it
   and the matching `<backend>-raw.json.gz` into `submissions/<my-box>/`,
   and generates a `README.md` summarizing the spec (machine incl. memory
   config, ceiling probes, and a `model × quant × provider` sweep/job coverage
   table).
3. Optionally preview the report with the new submission included:

   ```sh
   uv run --group analysis python -m analysis.site
   # → site/index.html — a local preview, never committed
   ```

4. Review the generated folder (and preview), then open a PR adding it.

## The report

The report is built by `analysis.site` over this folder — every chart
and number in it is computed from the submissions below it. It is never
committed: on every push to `main` the pages workflow
(`.github/workflows/pages.yml`) rebuilds it from all published submissions
and deploys it as the repo's GitHub Pages site. A submission PR therefore
only ever adds its own `<name>/` folder, so two submissions can land in
parallel without touching a shared generated file. The built page is
self-contained: open it anywhere, no environment needed, nothing fetched at
view time.

## Layout

```
submissions/
  <name>/
    README.md             # auto-generated spec summary (edit freely after)
    <backend>-results.json   # aggregated metrics — what the report loads
    <backend>-raw.json.gz    # raw per-spawn trace — re-aggregate with `survey aggregate`
```

The folder name (`<name>`) becomes the submission label in the report, so pick
something that identifies the box. Keep raw traces in: they let anyone re-derive
results if a metric definition changes, no re-inference needed.
