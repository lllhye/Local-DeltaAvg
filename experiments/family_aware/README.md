# Local-DeltaAvg-Family on the natural candidate pool

This experiment uses zero-shot mechanism assignment and concave
exposure gains to select a token-budgeted subset.

## Input

Export the scalar Local-DeltaAvg results as JSONL:

```json
{"id":"trace-1","question":"...","peak_window":"...","proofarrow_score":1.27,"token_count":731}
```

`peak_window` is the 64-token window attaining the centered Local-DeltaAvg score.

## 1. Assign the fixed 24 mechanism codes

```bash
python experiments/family_aware/assign_mechanism_families.py \
  --input /PATH/TO/local_deltaavg_candidates.jsonl \
  --output /PATH/TO/family_assignments.jsonl \
  --model /PATH/TO/FROZEN_PROBE \
  --batch-size 16 --max-length 1024 \
  --device cuda --dtype bfloat16
```

Run this data-parallel by input shard on the server, then concatenate the
assignment shards. The taxonomy, including the A–X choice symbols, is fixed in `taxonomy.json`. The scorer checks that each choice encodes to one token.

## 2. Select the 2B-token manifest

```bash
python experiments/family_aware/select_family_aware.py \
  --input /PATH/TO/family_assignments.jsonl \
  --manifest /PATH/TO/family_aware_2b.jsonl \
  --summary /PATH/TO/family_aware_summary.json \
  --budget-tokens 2000000000
```

The selector implements lazy greedy maximization of
`sum_r log(1 + A_r / tau_r)` with marginal gain divided by trace tokens.
For very large candidate counts, use the same calculation in the production
columnar loader; the JSONL implementation is the reference calculation.

## 3. Run the existing CPT, SFT, and evaluation protocol

Use `family_aware_2b.jsonl` in place of the scalar selection manifest. Keep the
backbone, token budget, optimizer schedule, SFT data, seeds, and evaluation
harness unchanged.

## 4. Export paper macros

`experiments/family_aware/results.json` contains the manuscript's scalar average, five-task results, assignment audit, coverage, fit diagnostics, and costs. Use the same schema to supply a new run's results.

```bash
python experiments/family_aware/make_family_results_tex.py \
  --results experiments/family_aware/results.json
```

The default output is `outputs/tables/family_aware_results.tex`.
