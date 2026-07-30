# apple-m5-pro — benchmark submission

| | |
|---|---|
| Host | apple-m5-pro |
| OS | macos |
| CPU | Apple M5 Pro (18C/18T) |
| GPU | Apple M5 Pro |
| Memory | 48 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## llamacpp  (6 runs)

| provider | device | threads | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|---|
| mtl:0 | Apple M5 Pro | 6 | 7.24 | 138.6 | ok |
| cpu:0 | Apple M5 Pro | 6 | 0.47 | 155.14 | ok |

| model | quant | provider | threads | sweep | job |
|---|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | 6 | ok (21 pts) | ok |
| Ministral3-3B | q4 | mtl:0 | 6 | ok (22 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | 6 | ok (19 pts) | ok |
| Qwen3.5-4B | q4 | mtl:0 | 6 | ok (20 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | 6 | ok (22 pts) | ok |
| gemma4-E2B | q4 | mtl:0 | 6 | ok (22 pts) | ok |
