# Reviewer controls: target invariance and acquisition specificity

This directory contains the matched experiments for target invariance and
acquisition specificity. `results.json` contains the measurements used by the
main paper and supplement.

The aggregate values in `results.json` match the current Local-DeltaAvg manuscript. The corresponding macros are in `results/manuscript_tables/reviewer_control_results.tex`.

## A. Stratified human target audit

Create the blinded 1,000-trace sheet:

```bash
python experiments/reviewer_controls/build_target_audit.py \
  --input /PATH/TO/candidates.jsonl \
  --output /PATH/TO/target_audit.csv
```

Use two independent annotators. Allowed primary failure labels are
`anaphora`, `local_variable`, `branch_or_code_state`, and
`discourse_ellipsis`. After adjudication:

```bash
python experiments/reviewer_controls/summarize_target_audit.py \
  --input /PATH/TO/target_audit.csv \
  --output /PATH/TO/target_audit_summary.json
```

## B. Context-safe reversal

The filter removes only leading discourse connectives and rejects unresolved
context-dependent references. It never generates content.

```bash
python experiments/reviewer_controls/build_context_safe_pool.py \
  --input /PATH/TO/candidates.jsonl \
  --accepted /PATH/TO/context_safe_candidates.jsonl \
  --rejected /PATH/TO/context_unsafe_ids.jsonl
```

Run the existing answer-context-masked scorer on the accepted pool and its
matched null pool. Report raw-safe Spearman, top-10/top-20 overlap, utility
correlation, critical-step IoU, and one matched 2B CPT run using the safe
manifest. Keep rejected traces out of both scoring and selection.

## C. Single-view versus multi-view CPT

Export already null-centered scores in this form:

```json
{"id":"trace-1","token_count":731,"view_scores":[1.27,1.18,1.31,1.24,1.29]}
```

Build the three matched manifests:

```bash
python experiments/reviewer_controls/build_multiview_manifests.py \
  --input /PATH/TO/five_view_centered_scores.jsonl \
  --output-dir /PATH/TO/manifests/multiview \
  --budget 2000000000 --lcb-lambda 1.0
```

The first view is the deterministic production reversal. Use the same three
seeds, CPT/SFT data, optimizer, packing, decoding, and evaluation harness
for the single-view, mean, and conservative aggregates.

## D. HES/AskLLM/label-only mechanism acquisition

Put all scalar scores in the candidate JSONL:

```json
{"id":"trace-1","question":"...","peak_window":"...","token_count":731,
 "proofarrow_score":1.27,"hes_score":8.14,"askllm_score":84.0}
```

Assign the fixed mechanisms once:

```bash
python experiments/family_aware/assign_mechanism_families.py \
  --input /PATH/TO/scored_candidates.jsonl \
  --output /PATH/TO/mechanism_assignments.jsonl \
  --model /PATH/TO/FROZEN_PROBE \
  --score-keys proofarrow_score hes_score askllm_score \
  --batch-size 16 --max-length 1024 --device cuda --dtype bfloat16
```

Build the three control manifests:

```bash
python experiments/family_aware/select_family_aware.py \
  --input /PATH/TO/mechanism_assignments.jsonl \
  --manifest /PATH/TO/hes_mechanism_2b.jsonl \
  --summary /PATH/TO/hes_mechanism_summary.json \
  --score-key hes_score --run-name hes_mechanism --budget-tokens 2000000000

python experiments/family_aware/select_family_aware.py \
  --input /PATH/TO/mechanism_assignments.jsonl \
  --manifest /PATH/TO/askllm_mechanism_2b.jsonl \
  --summary /PATH/TO/askllm_mechanism_summary.json \
  --score-key askllm_score --run-name askllm_mechanism --budget-tokens 2000000000

python experiments/family_aware/select_family_aware.py \
  --input /PATH/TO/mechanism_assignments.jsonl \
  --manifest /PATH/TO/mechanism_only_2b.jsonl \
  --summary /PATH/TO/mechanism_only_summary.json \
  --exposure-mode label-only --run-name mechanism_only --budget-tokens 2000000000
```

## E. Directional score with generic STEM topics

Assign the fixed 24-topic control taxonomy, then use the unchanged acquisition:

```bash
python experiments/family_aware/assign_mechanism_families.py \
  --input /PATH/TO/scored_candidates.jsonl \
  --output /PATH/TO/stem_topic_assignments.jsonl \
  --taxonomy experiments/family_aware/stem_taxonomy.json \
  --model /PATH/TO/FROZEN_PROBE \
  --score-keys proofarrow_score \
  --batch-size 16 --max-length 1024 --device cuda --dtype bfloat16

python experiments/family_aware/select_family_aware.py \
  --input /PATH/TO/stem_topic_assignments.jsonl \
  --manifest /PATH/TO/directional_stem_2b.jsonl \
  --summary /PATH/TO/directional_stem_summary.json \
  --score-key proofarrow_score --run-name directional_stem \
  --budget-tokens 2000000000
```

Run the same three-seed CPT/SFT/evaluation protocol for every new manifest.

## F. Regenerate the paper tables

Generate manuscript macros from the recorded aggregates:

```bash
python experiments/reviewer_controls/make_reviewer_results_tex.py \
  --results experiments/reviewer_controls/results.json
```

The default output is `outputs/tables/reviewer_control_results.tex`.
