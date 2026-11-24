# Local-DeltaAvg answer-masking experiment

This directory implements the minimum experiment needed to answer the
answer-first leakage concern. It produces three directly comparable selectors:

1. `answer_first_raw`: answer is visible in the reversed context and answer
   target losses are included (the raw ablation).
2. `answer_loss_masked`: answer is still visible in the reversed context, but
   answer target losses are excluded. This isolates the effect of scoring the
   answer; it **does not remove answer leakage from the context**.
3. `answer_context_masked`: answer is absent from both scored views and answer
   target losses are excluded. This is the default Local-DeltaAvg scoring construction.

## 1. Expected input

Both the natural pool and matched-null pool are JSONL files. The default
schema is:

```json
{"id":"trace-1","question":"...","steps":["...","..."],"answer":"...","token_count":731}
```

Field names can be changed with `--id-key`, `--question-key`, `--steps-key`,
and `--answer-key`. First validate the formatting without loading a model:

```bash
python experiments/answer_masking/run_answer_masking_scores.py \
  --input /PATH/TO/candidates.jsonl \
  --output /tmp/not-used.csv \
  --dry-run
```

## 2. Score the natural and matched-null pools

Use exactly the same frozen probe, prompt formatting, context length, window,
and null construction as the existing raw run.

```bash
python experiments/answer_masking/run_answer_masking_scores.py \
  --input /PATH/TO/candidates.jsonl \
  --output /PATH/TO/scores/candidates_answer_masking.csv \
  --model /PATH/TO/FROZEN_PROBE \
  --window 64 --max-length 4096 --batch-size 8 \
  --trace-chunk-size 8 --device cuda --dtype bfloat16 --resume

python experiments/answer_masking/run_answer_masking_scores.py \
  --input /PATH/TO/matched_null.jsonl \
  --output /PATH/TO/scores/null_answer_masking.csv \
  --model /PATH/TO/FROZEN_PROBE \
  --window 64 --max-length 4096 --batch-size 8 \
  --trace-chunk-size 8 --device cuda --dtype bfloat16 --resume
```

`--resume` appends only missing IDs. For multi-GPU sharding, shard both JSONL
files by ID, run one output CSV per shard, keep one header, and concatenate the
rows before the next step.

## 3. Center scores and build 2B-token manifests

```bash
python experiments/answer_masking/build_selection_manifests.py \
  --scores /PATH/TO/scores/candidates_answer_masking.csv \
  --null-scores /PATH/TO/scores/null_answer_masking.csv \
  --output-dir /PATH/TO/manifests/answer_masking \
  --budget 2000000000 \
  --length-bin-width 256 --window-bin-width 32 --min-null-bin 20
```

This writes one centered manifest per variant. The manifest contains IDs,
token counts, raw/null/centered scores, and ranks. The reference converter estimates null statistics from the supplied CSV strata. The supplement specifies the per-trace null matching used for the full-pool run.

## 4. Run CPT and evaluation

Use the matching selection manifest with the CPT, SFT, and evaluation
configuration in `configs/public_pool_recipe.yaml`. Keep the backbone, token
budget, data packing, optimizer, and evaluation pipeline fixed across
the three variants. Record results for each of the three seeds.

## 5. Write the measured rows into the paper

Record measured five-task averages in the schema below, or use the bundled manuscript aggregates in `experiments/answer_masking/results.json`.

```json
{
  "answer_loss_masked_centered": {
    "GSM8K": 0.0, "MATH500": 0.0, "AIME24": 0.0,
    "GPQA-D": 0.0, "LCB": 0.0
  },
  "answer_context_masked_centered": {
    "GSM8K": 0.0, "MATH500": 0.0, "AIME24": 0.0,
    "GPQA-D": 0.0, "LCB": 0.0
  }
}
```

Then run from the repository root:

```bash
python experiments/answer_masking/make_results_tex.py \
  --results /PATH/TO/answer_masking_eval.json \
  --output outputs/tables/answer_masking_results.tex
```

The generator writes values and averages as LaTeX macros. The paper snapshot is in `results/manuscript_tables/`.
