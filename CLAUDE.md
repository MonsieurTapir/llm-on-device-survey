# CLAUDE.md — working agreements

Guidance for anyone (human or agent) working in this repo. Keep it short; if it
grows stale, fix it.

## Orientation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the map: components, how they couple,
  the measurement model. Start there.
- Each component specifies itself in its own `README.md` (or directly in its
  sources): `harness/`, `analysis/`, `backends/*/`, `tasks/`, `results/published/`.
- `schema/` holds the wire formats (events, results); the JSON Schemas are the
  field-level truth.
- `models/` and `third_party/` are **untracked local data** — never commit them,
  never assume their contents in tests. Backends fetch their own inference stack
  at build time (pinned; overridable via `*_SOURCE_DIR` to a local checkout).
  Nothing is ever downloaded at *runtime*.

## The harness↔backend seam

Three things couple the harness and a backend — change them deliberately and keep
both sides in sync:

1. **CLI** — `providers` / `run` / `sweep` / `probe` / `version`.
2. **JSON schemas** — `schema/{events,results}.schema.json`. A backend's `run`
   output validates against `events.schema.json`; the harness's output against
   `results.schema.json`. A schema change bumps its `schema_version`, updates
   every backend and the analysis loader, and says so in the commit.
3. **`backend.toml`** — how the harness invokes an exe; its `key` matches the
   events `backend` field and the `models.yaml` block (`llamacpp`→`gguf`).

Backend invariants (enforced, easy to get wrong):

- stdout = the JSON object only; everything else → stderr.
- Greedy/argmax decode, exactly `nb_tokens` per turn, EOS ignored.
- Prompts render through the **model's own chat template**
  (`enable_thinking=false`) — never hand-concatenated role text.
- `turn-end.completion` carries the decoded text so results stay
  eyeball-inspectable.
- **The `warmup` span owns all first-touch setup.** Compute pipelines are built
  lazily, on first use, inside the graph compute that needs them, and which one
  that is depends on the dispatch width — so warmup walks *every* width the
  measured passes will run (full ubatch, full over existing history, half,
  short ragged, single-token decode; plus the task's own prompts in `run`) and
  ends with an explicit device sync. Miss a width and seconds of shader
  compilation land in the first measured span that uses it. The harness pins the
  shader cache per spawn, which turns this span into the first-launch number.

After building or changing a backend, validate it:
`uv run --project harness bench check --backend <key> --models models`.
A backend that doesn't pass is not done.

## Conventions

- **Conventional Commits** — `feat:` / `fix:` / `docs:` / `chore:` / `refactor:` /
  `test:`, scoped where it helps (`feat(harness):`). Imperative, lowercase subject;
  the *why* in the body.
- A published run is its own commit: `submission(<name>): <one-line hardware
  description>` — e.g. `submission(monsieurtapir-laptop): mid-range Dell Ryzen
  laptop`.
- Agent-authored commits carry a `Co-Authored-By` footer naming the agent
  (e.g. `Co-Authored-By: Claude <noreply@anthropic.com>`).
- Solo project: work directly on `main`, no feature branches. Commit/push only
  when asked.
- Python (`harness/`, `analysis/`) are **uv** projects: run with
  `uv run --project <dir>`, add deps in `pyproject.toml` and commit the updated
  `uv.lock`, lint/format with `ruff`.
- Docs and comments state **what is** — no history-telling about what was removed
  or used to exist. One README per folder is the documentation budget.
