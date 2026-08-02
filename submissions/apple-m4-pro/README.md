# apple-m4-pro — benchmark submission

| | |
|---|---|
| Host | apple-m4-pro |
| OS | macos |
| CPU | Apple M4 Pro (14C/14T) |
| GPU | Apple M4 Pro |
| Memory | 48 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## llamacpp  (6 runs)

| provider | device | threads | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|---|
| mtl:0 | Apple M4 Pro | 10 | 7.05 | 77.52 | ok |
| cpu:0 | Apple M4 Pro | 10 | 0.68 | 223.1 | ok |

| model | quant | provider | threads | sweep | job |
|---|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | 10 | ok (22 pts) | ok |
| Ministral3-3B | q4 | mtl:0 | 10 | ok (22 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | 10 | ok (20 pts) | ok |
| Qwen3.5-4B | q4 | mtl:0 | 10 | ok (20 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | 10 | ok (22 pts) | ok |
| gemma4-E2B | q4 | mtl:0 | 10 | ok (22 pts) | ok |
