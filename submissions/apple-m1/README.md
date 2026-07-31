# apple-m1 — benchmark submission

| | |
|---|---|
| Host | apple-m1 |
| OS | macos |
| CPU | Apple M1 (8C/8T) |
| GPU | Apple M1 |
| Memory | 16 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## llamacpp  (6 runs)

| provider | device | threads | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|---|
| mtl:0 | Apple M1 | 4 | 2.02 | 21.42 | ok |
| cpu:0 | Apple M1 | 4 | 0.2 | 58.12 | ok |

| model | quant | provider | threads | sweep | job |
|---|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | 4 | ok (12 pts) | ok |
| Ministral3-3B | q4 | mtl:0 | 4 | ok (22 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | 4 | ok (10 pts) | ok |
| Qwen3.5-4B | q4 | mtl:0 | 4 | ok (20 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | 4 | ok (14 pts) | ok |
| gemma4-E2B | q4 | mtl:0 | 4 | ok (22 pts) | ok |
