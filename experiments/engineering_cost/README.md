# Local-DeltaAvg engineering-cost benchmark

The full-pool main method uses deterministic semantic-unit reordering, not LLM
generation. This benchmark measures parsing plus one forward and one reversed
likelihood pass on a representative shard, then extrapolates GPU-hours to the
20B-token candidate pool. It also measures HES when requested and can ingest
the already-recorded GPU-hours of the paper's LLM-judge baseline.

## 1. Run a representative server benchmark

Use the same probe, precision, maximum context, batching/packing, and input
format as the 20B selection run. A 10M-token stratified shard is the default;
50M tokens gives a more stable throughput measurement.

```bash
python experiments/engineering_cost/benchmark_pipeline.py \
  --input /PATH/TO/stratified_candidates.jsonl \
  --output /PATH/TO/local_deltaavg_engineering.json \
  --model /PATH/TO/FROZEN_PROBE \
  --sample-source-tokens 50000000 \
  --pool-source-tokens 20000000000 \
  --null-source-tokens 250000000 \
  --batch-size 8 --max-length 4096 \
  --num-gpus 8 --num-cpu-workers 16 --device cuda --dtype bfloat16 \
  --measure-hes \
  --judge-gpu-hours-20b YOUR_RECORDED_JUDGE_GPU_HOURS
```

Input defaults to the JSONL schema used by the answer-masking experiment:

```json
{"id":"trace-1","question":"...","steps":["...","..."],"answer":"...","token_count":731}
```

The reported GPU-hours are throughput extrapolations. If the original full
20B logs contain actual device-hours, replace the extrapolated values in the
JSON before generating the paper macros; actual logs take precedence.

For multi-GPU execution, benchmark one representative GPU/process without data
parallel communication and use `--num-gpus` only to convert device-hours to
wall-clock hours. If production packing differs, benchmark the production
loader instead, because padding can dominate the measured throughput.

## 2. Write the measured paper values

From the repository root:

```bash
python experiments/engineering_cost/make_engineering_tex.py \
  --benchmark /PATH/TO/local_deltaavg_engineering.json
```

The generated macros are written to `outputs/tables/engineering_cost_results.tex`.
Use `--outputs` to specify paths. The original production measurements used
by the manuscript are in `results/manuscript_tables/engineering_cost_results.tex`.

## 3. What the benchmark establishes

- `training-free` means no selector/verifier training, not zero compute;
- the 20B reversal is a deterministic permutation of existing units;
- alignment is inherited from unit identity, so accepted traces cannot drop or
  duplicate units silently;
- the principal overhead relative to forward PPL is a second likelihood pass;
- the 2,000-trace multi-view diagnostic may use generated paraphrases, but it
  is not part of the full-pool selection cost.
