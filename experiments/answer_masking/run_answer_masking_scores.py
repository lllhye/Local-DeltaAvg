#!/usr/bin/env python3
"""Score ProofArrow answer-masking variants on a JSONL trace corpus.

The script deliberately separates two frequently conflated interventions:

* answer_loss_masked keeps the answer in the reversed context but does not
  score answer tokens;
* answer_context_masked removes the answer from both aligned views and does
  not score answer tokens.

Input defaults to the fields ``id``, ``question``, ``steps``, ``answer``, and
optionally ``token_count``. Field names are configurable from the CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


VARIANTS = (
    "answer_first_raw",
    "answer_loss_masked",
    "answer_context_masked",
)


@dataclass
class Trace:
    trace_id: str
    question: str
    steps: list[str]
    answer: str
    token_count: int | None = None


@dataclass
class TargetExample:
    key: tuple[int, str, int]
    prefix: str
    target: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input JSONL corpus")
    p.add_argument("--output", required=True, help="Output score CSV")
    p.add_argument("--model", help="Hugging Face model name or local path")
    p.add_argument("--id-key", default="id")
    p.add_argument("--question-key", default="question")
    p.add_argument("--steps-key", default="steps")
    p.add_argument("--answer-key", default="answer")
    p.add_argument("--token-count-key", default="token_count")
    p.add_argument("--step-separator", default="\n", help="Used when steps is a string")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--trace-chunk-size", type=int, default=8)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and print the three contexts for the first trace without loading a model",
    )
    return p.parse_args()


def normalize_steps(value: object, separator: str) -> list[str]:
    if isinstance(value, list):
        steps = [str(x).strip() for x in value]
    elif isinstance(value, str):
        steps = [x.strip() for x in value.split(separator)]
    else:
        raise TypeError(f"steps must be a list or string, got {type(value).__name__}")
    return [x for x in steps if x]


def load_traces(args: argparse.Namespace, completed: set[str]) -> Iterator[Trace]:
    emitted = 0
    with Path(args.input).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            trace_id = str(obj.get(args.id_key, line_no))
            if trace_id in completed:
                continue
            steps = normalize_steps(obj[args.steps_key], args.step_separator)
            if not steps:
                raise ValueError(f"line {line_no}: no reasoning steps")
            tc = obj.get(args.token_count_key)
            yield Trace(
                trace_id=trace_id,
                question=str(obj[args.question_key]).strip(),
                steps=steps,
                answer=str(obj[args.answer_key]).strip(),
                token_count=int(tc) if tc is not None else None,
            )
            emitted += 1
            if args.limit is not None and emitted >= args.limit:
                break


def chunks(items: Iterable[Trace], size: int) -> Iterator[list[Trace]]:
    chunk: list[Trace] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def q_prefix(question: str) -> str:
    return f"Question:\n{question}\n"


def reasoning_prefix(question: str, steps: Sequence[str], reversed_view: bool) -> str:
    label = "Reversed reasoning:" if reversed_view else "Reasoning:"
    body = "".join(f"\nStep: {step}" for step in steps)
    return q_prefix(question) + label + body


def raw_reverse_prefix(trace: Trace, later_steps: Sequence[str]) -> str:
    body = "".join(f"\nStep: {step}" for step in later_steps)
    return q_prefix(trace.question) + f"Answer:\n{trace.answer}\nReversed reasoning:" + body


def build_examples(chunk_traces: Sequence[Trace]) -> list[TargetExample]:
    examples: list[TargetExample] = []
    for j, trace in enumerate(chunk_traces):
        for i, step in enumerate(trace.steps):
            target = f"\nStep: {step}"
            earlier = trace.steps[:i]
            later_reversed = list(reversed(trace.steps[i + 1 :]))
            examples.append(
                TargetExample(
                    key=(j, "step_forward", i),
                    prefix=reasoning_prefix(trace.question, earlier, False),
                    target=target,
                )
            )
            examples.append(
                TargetExample(
                    key=(j, "step_reverse_raw", i),
                    prefix=raw_reverse_prefix(trace, later_reversed),
                    target=target,
                )
            )
            examples.append(
                TargetExample(
                    key=(j, "step_reverse_context_masked", i),
                    prefix=reasoning_prefix(trace.question, later_reversed, True),
                    target=target,
                )
            )

        examples.append(
            TargetExample(
                key=(j, "answer_forward", 0),
                prefix=reasoning_prefix(trace.question, trace.steps, False),
                target=f"\nAnswer: {trace.answer}",
            )
        )
        examples.append(
            TargetExample(
                key=(j, "answer_reverse", 0),
                prefix=q_prefix(trace.question),
                target=f"\nAnswer: {trace.answer}",
            )
        )
    return examples


def rolling_max(values: Sequence[float], window: int) -> tuple[float, float, int]:
    if not values:
        return math.nan, math.nan, 0
    mean = sum(values) / len(values)
    if len(values) <= window:
        return mean, mean, 1
    running = sum(values[:window])
    best = running / window
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        best = max(best, running / window)
    return best, mean, len(values) - window + 1


def select_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


class CausalNLLScorer:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = select_device(torch, args.device)
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if args.dtype == "auto":
            dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else (
                torch.float16 if self.device in {"cuda", "mps"} else torch.float32
            )
        else:
            dtype = dtype_map[args.dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        ).to(self.device)
        self.model.eval()
        self.max_length = args.max_length
        self.batch_size = args.batch_size

    def token_count(self, trace: Trace) -> int:
        text = reasoning_prefix(trace.question, trace.steps, False) + f"\nAnswer: {trace.answer}"
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _encode(self, example: TargetExample) -> tuple[list[int], int]:
        prefix_ids = self.tokenizer.encode(example.prefix, add_special_tokens=True)
        target_ids = self.tokenizer.encode(example.target, add_special_tokens=False)
        if not target_ids:
            raise ValueError(f"empty target after tokenization: {example.key}")
        target_ids = target_ids[: self.max_length - 1]
        room = self.max_length - len(target_ids)
        prefix_ids = prefix_ids[-room:]
        return prefix_ids + target_ids, len(prefix_ids)

    def score(self, examples: Sequence[TargetExample]) -> dict[tuple[int, str, int], list[float]]:
        torch = self.torch
        results: dict[tuple[int, str, int], list[float]] = {}
        encoded = [(ex, *self._encode(ex)) for ex in examples]
        for start in range(0, len(encoded), self.batch_size):
            batch = encoded[start : start + self.batch_size]
            max_len = max(len(ids) for _, ids, _ in batch)
            pad = self.tokenizer.pad_token_id
            input_rows: list[list[int]] = []
            attention_rows: list[list[int]] = []
            for _, ids, _ in batch:
                padding = max_len - len(ids)
                input_rows.append(ids + [pad] * padding)
                attention_rows.append([1] * len(ids) + [0] * padding)
            input_ids = torch.tensor(input_rows, device=self.device)
            attention = torch.tensor(attention_rows, device=self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attention).logits.float()
                log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            next_ids = input_ids[:, 1:]
            token_nll = -log_probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
            for row, (example, ids, prefix_len) in enumerate(batch):
                first = max(prefix_len - 1, 0)
                last = len(ids) - 1
                results[example.key] = token_nll[row, first:last].detach().cpu().tolist()
        return results


def pair_delta(reverse: Sequence[float], forward: Sequence[float], label: str) -> list[float]:
    if len(reverse) != len(forward):
        raise ValueError(f"unaligned token losses for {label}: {len(reverse)} != {len(forward)}")
    return [r - f for r, f in zip(reverse, forward)]


def aggregate_trace(
    trace_index: int,
    trace: Trace,
    losses: dict[tuple[int, str, int], list[float]],
    window: int,
    token_count: int,
) -> dict[str, object]:
    step_raw: list[float] = []
    step_context_masked: list[float] = []
    for i in range(len(trace.steps)):
        forward = losses[(trace_index, "step_forward", i)]
        step_raw.extend(pair_delta(losses[(trace_index, "step_reverse_raw", i)], forward, f"step {i}"))
        step_context_masked.extend(
            pair_delta(losses[(trace_index, "step_reverse_context_masked", i)], forward, f"step {i}")
        )
    answer_delta = pair_delta(
        losses[(trace_index, "answer_reverse", 0)],
        losses[(trace_index, "answer_forward", 0)],
        "answer",
    )
    variant_values = {
        "answer_first_raw": step_raw + answer_delta,
        "answer_loss_masked": step_raw,
        "answer_context_masked": step_context_masked,
    }
    row: dict[str, object] = {
        "id": trace.trace_id,
        "token_count": token_count,
        "n_steps": len(trace.steps),
    }
    for name, values in variant_values.items():
        score, mean, n_windows = rolling_max(values, window)
        row[f"{name}_score"] = f"{score:.8f}"
        row[f"{name}_mean"] = f"{mean:.8f}"
        row[f"{name}_n_tokens"] = len(values)
        row[f"{name}_n_windows"] = n_windows
    return row


def output_fields() -> list[str]:
    fields = ["id", "token_count", "n_steps"]
    for name in VARIANTS:
        fields.extend(
            [f"{name}_score", f"{name}_mean", f"{name}_n_tokens", f"{name}_n_windows"]
        )
    return fields


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def print_dry_run(trace: Trace) -> None:
    examples = build_examples([trace])
    print(json.dumps({
        "id": trace.trace_id,
        "raw_reverse_prefix_for_first_step": examples[1].prefix,
        "context_masked_reverse_prefix_for_first_step": examples[2].prefix,
        "answer_loss_masked_note": "same reverse contexts as raw; answer target loss is excluded",
        "answer_context_masked_note": "answer absent from reverse contexts; answer target loss is excluded",
    }, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.model:
        raise SystemExit("--model is required unless --dry-run is used")
    if args.window <= 0 or args.max_length < 2 or args.batch_size <= 0 or args.trace_chunk_size <= 0:
        raise SystemExit("window, max-length, batch-size, and trace-chunk-size must be positive")

    output = Path(args.output)
    done = completed_ids(output) if args.resume else set()
    trace_iter = load_traces(args, done)
    if args.dry_run:
        try:
            print_dry_run(next(trace_iter))
        except StopIteration:
            raise SystemExit("no unprocessed traces found")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    append = args.resume and output.exists() and output.stat().st_size > 0
    scorer = CausalNLLScorer(args)
    with output.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields())
        if not append:
            writer.writeheader()
        processed = 0
        for trace_chunk in chunks(trace_iter, args.trace_chunk_size):
            losses = scorer.score(build_examples(trace_chunk))
            for j, trace in enumerate(trace_chunk):
                tc = trace.token_count if trace.token_count is not None else scorer.token_count(trace)
                writer.writerow(aggregate_trace(j, trace, losses, args.window, tc))
                processed += 1
            f.flush()
            print(f"processed={processed} output={output}", flush=True)


if __name__ == "__main__":
    main()
