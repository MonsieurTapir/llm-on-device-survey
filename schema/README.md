# schema — the wire contracts

The two JSON Schemas the components couple through. They are the field-level
truth: prose elsewhere summarizes them, these define them.

- `events.schema.json` — what a backend exe emits on stdout, one object per
  spawn (`run` / `sweep` / `probe`). The survey tool validates every spawn's
  output against it on the way in.
- `results.schema.json` — what the survey tool writes per backend after
  aggregation, and the only thing the analysis project reads. Validated on
  the way out; every analysis loader pins its `schema_version`.

Each schema carries its own `schema_version`. Changing a schema bumps that
version, updates every backend and the analysis loader in the same change,
and says so in the commit. Description-only edits don't bump.

Conventions the schemas encode: stdout carries the JSON object only (logging
goes to stderr); quant labels are a closed enum (`fp16|q8|q4`) naming
stack-specific math; every events object embeds the exact stack versions
that qualify its numbers.
