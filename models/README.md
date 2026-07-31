# models — registry and local vault

`models.yaml` is the model registry: what the benchmark runs on, and where
`survey fetch` pulls it from. Per model, a backend block (`gguf` for
llamacpp) naming a Hub repo, a `common` glob list (shared config/tokenizer
files), and a `quants` map whose keys are the wire enum (`fp16|q8|q4`) —
one artifact per declared quant.

Everything else in this folder is the untracked local vault:
`<model>/gguf/…` trees that `survey fetch` fills and the backends consume
directly. Multi-GB, never committed — the registry is the only tracked file
here. Exes never read the registry: the survey tool resolves the artifact
path and hands the exe `--model` / `--quant` / `--ep`.

A declared-but-unfetched quant is skipped with a loud warning, never
silently. Quant labels are stack-specific math (`q4` = Q4_K_M): comparisons
hold within a label, qualified by the stack versions each result embeds.
