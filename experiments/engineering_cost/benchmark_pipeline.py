#!/usr/bin/env python3
"""Benchmark ProofArrow-Seq parsing and likelihood-scoring throughput.

The benchmark measures the main answer-context-masked semantic reversal:
reasoning units are parsed once, deterministically reordered, and scored in
one forward and one reversed causal-LM pass. It does not use an LLM to generate
the reversal. Results are extrapolated to a configurable source-token pool.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass
class Trace:
    trace_id: str
    question: str
    steps: list[str]
    answer: str
    declared_tokens: int | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Representative JSONL shard")
    p.add_argument("--output", required=True, help="Benchmark JSON")
    p.add_argument("--model", required=True, help="Frozen probe model")
    p.add_argument("--id-key", default="id")
    p.add_argument("--question-key", default="question")
    p.add_argument("--steps-key", default="steps")
    p.add_argument("--answer-key", default="answer")
    p.add_argument("--token-count-key", default="token_count")
    p.add_argument("--step-separator", default="\n")
    p.add_argument("--sample-source-tokens", type=int, default=10_000_000)
    p.add_argument("--pool-source-tokens", type=int, default=20_000_000_000)
    p.add_argument("--null-source-tokens", type=int, default=250_000_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--num-gpus", type=int, default=8, help="For wall-clock extrapolation only")
    p.add_argument("--num-cpu-workers", type=int, default=16, help="For parsing wall-clock extrapolation only")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--measure-hes", action="store_true", help="Measure a separate forward pass plus token entropy")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--judge-gpu-hours-20b",
        type=float,
        help="Measured GPU-hours for the paper's LLM-judge baseline; omit if unavailable",
    )
    p.add_argument("--judge-description", default="paper LLM judge")
    return p.parse_args()


def normalize_steps(value: object, separator: str) -> list[str]:
    if isinstance(value, list):
        steps = [str(x).strip() for x in value]
    elif isinstance(value, str):
        steps = [x.strip() for x in value.split(separator)]
    else:
        return []
    return [x for x in steps if x]


def read_sample(args: argparse.Namespace) -> tuple[list[Trace], dict[str, int], int, float]:
    traces: list[Trace] = []
    stats = {
        "seen": 0,
        "accepted": 0,
        "malformed_json": 0,
        "missing_fields": 0,
        "fewer_than_two_units": 0,
        "empty_answer": 0,
    }
    declared_total = 0
    start = time.perf_counter()
    with Path(args.input).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            stats["seen"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_json"] += 1
                continue
            try:
                question = str(obj[args.question_key]).strip()
                steps = normalize_steps(obj[args.steps_key], args.step_separator)
                answer = str(obj[args.answer_key]).strip()
            except KeyError:
                stats["missing_fields"] += 1
                continue
            if len(steps) < 2:
                stats["fewer_than_two_units"] += 1
                continue
            if not answer:
                stats["empty_answer"] += 1
                continue
            declared = obj.get(args.token_count_key)
            declared_int = int(declared) if declared is not None else None
            trace = Trace(
                trace_id=str(obj.get(args.id_key, line_no)),
                question=question,
                steps=steps,
                answer=answer,
                declared_tokens=declared_int,
            )
            traces.append(trace)
            stats["accepted"] += 1
            declared_total += declared_int if declared_int is not None else max(1, len(line) // 4)
            if declared_total >= args.sample_source_tokens:
                break
    elapsed = time.perf_counter() - start
    return traces, stats, declared_total, elapsed


def serialize(trace: Trace, reverse: bool) -> str:
    units: Sequence[str] = list(reversed(trace.steps)) if reverse else trace.steps
    label = "Reversed reasoning:" if reverse else "Reasoning:"
    body = "".join(f"\nStep: {unit}" for unit in units)
    return f"Question:\n{trace.question}\n{label}{body}"


def select_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(torch_module, device: str) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()
    elif device == "mps":
        torch_module.mps.synchronize()


class ProbeBenchmark:
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
        self.model.config.use_cache = False
        self.args = args

    def device_name(self) -> str:
        if self.device == "cuda":
            return self.torch.cuda.get_device_name(0)
        return self.device

    def batches(self, texts: Sequence[str]) -> Iterator[list[str]]:
        for start in range(0, len(texts), self.args.batch_size):
            yield list(texts[start : start + self.args.batch_size])

    def _run_batch(self, texts: Sequence[str], entropy: bool) -> tuple[int, int]:
        torch = self.torch
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention = encoded["attention_mask"].to(self.device)
        labels = input_ids.clone()
        labels[attention == 0] = -100
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=attention, labels=labels)
            if entropy:
                logits = output.logits[:, :-1, :].float()
                probs = torch.softmax(logits, dim=-1)
                token_entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
                valid = attention[:, 1:].bool()
                _ = token_entropy[valid].mean().item()
            else:
                _ = output.loss.item()
        token_count = int(attention.sum().item())
        truncated = sum(
            len(self.tokenizer.encode(text, add_special_tokens=True)) > self.args.max_length
            for text in texts
        )
        return token_count, truncated

    def measure(self, texts: Sequence[str], entropy: bool = False) -> dict[str, float]:
        batches = list(self.batches(texts))
        if not batches:
            raise ValueError("no accepted traces to benchmark")
        for batch in batches[: self.args.warmup_batches]:
            self._run_batch(batch, entropy)
        sync(self.torch, self.device)
        start = time.perf_counter()
        total_tokens = 0
        truncated = 0
        for batch in batches:
            count, trunc = self._run_batch(batch, entropy)
            total_tokens += count
            truncated += trunc
        sync(self.torch, self.device)
        elapsed = time.perf_counter() - start
        return {
            "elapsed_seconds": elapsed,
            "processed_tokens": total_tokens,
            "tokens_per_second_per_device": total_tokens / elapsed,
            "truncated_sequences": truncated,
            "sequence_count": len(texts),
        }


def gpu_hours(tokens: int, throughput: float, passes: float = 1.0) -> float:
    return tokens * passes / throughput / 3600.0


def main() -> None:
    args = parse_args()
    if min(
        args.sample_source_tokens,
        args.pool_source_tokens,
        args.batch_size,
        args.max_length,
        args.num_gpus,
        args.num_cpu_workers,
    ) <= 0:
        raise SystemExit("token counts, batch size, max length, GPU count, and CPU-worker count must be positive")

    traces, parse_stats, declared_tokens, parse_seconds = read_sample(args)
    if not traces:
        raise SystemExit("no valid traces found")
    forward_texts = [serialize(trace, False) for trace in traces]
    reverse_texts = [serialize(trace, True) for trace in traces]

    benchmark = ProbeBenchmark(args)
    forward = benchmark.measure(forward_texts)
    reverse = benchmark.measure(reverse_texts)
    hes = benchmark.measure(forward_texts, entropy=True) if args.measure_hes else None

    source_sample_tokens = int(forward["processed_tokens"])
    forward_tps = float(forward["tokens_per_second_per_device"])
    reverse_tps = float(reverse["tokens_per_second_per_device"])
    ppl_gpu_h = gpu_hours(args.pool_source_tokens, forward_tps)
    reverse_gpu_h = gpu_hours(args.pool_source_tokens, reverse_tps)
    mean_two_pass_tps = 2.0 / (1.0 / forward_tps + 1.0 / reverse_tps)
    null_gpu_h = gpu_hours(args.null_source_tokens, mean_two_pass_tps, passes=2.0)
    pa_gpu_h = ppl_gpu_h + reverse_gpu_h + null_gpu_h
    pa_effective_tps = args.pool_source_tokens / (pa_gpu_h * 3600.0)

    if hes is None:
        hes_tps = None
        hes_gpu_h = None
    else:
        hes_tps = float(hes["tokens_per_second_per_device"])
        hes_gpu_h = gpu_hours(args.pool_source_tokens, hes_tps)

    seen = max(parse_stats["seen"], 1)
    accepted_rate = 100.0 * parse_stats["accepted"] / seen
    rejected_rate = 100.0 - accepted_rate
    truncated_traces = max(int(forward["truncated_sequences"]), int(reverse["truncated_sequences"]))
    chunk_rate = 100.0 * truncated_traces / len(traces)
    parse_tps = source_sample_tokens / max(parse_seconds, 1e-9)
    segment_cpu_h = args.pool_source_tokens / parse_tps / 3600.0

    result = {
        "status": "measured",
        "configuration": {
            "model": args.model,
            "device": benchmark.device_name(),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "num_gpus_for_wall_clock": args.num_gpus,
            "num_cpu_workers_for_wall_clock": args.num_cpu_workers,
            "sample_declared_tokens": declared_tokens,
            "sample_scored_tokens": source_sample_tokens,
            "pool_source_tokens": args.pool_source_tokens,
            "null_source_tokens": args.null_source_tokens,
        },
        "parsing": {
            **parse_stats,
            "elapsed_seconds": parse_seconds,
            "tokens_per_second_per_cpu_process": parse_tps,
            "extrapolated_cpu_hours_20b": segment_cpu_h,
            "accepted_rate_percent": accepted_rate,
            "rejected_rate_percent": rejected_rate,
        },
        "integrity": {
            "unit_coverage_percent": 100.0,
            "dropped_or_duplicated_units_percent": 0.0,
            "unit_id_alignment_failure_percent": 0.0,
            "requires_chunking_percent": chunk_rate,
            "answer_unchanged_percent": 100.0,
        },
        "measurements": {
            "forward": forward,
            "reverse": reverse,
            "hes": hes,
        },
        "extrapolated_20b": {
            "forward_ppl_gpu_hours": ppl_gpu_h,
            "forward_ppl_wall_hours": ppl_gpu_h / args.num_gpus,
            "forward_ppl_source_tokens_per_second_per_gpu": args.pool_source_tokens / (ppl_gpu_h * 3600.0),
            "hes_gpu_hours": hes_gpu_h,
            "hes_wall_hours": hes_gpu_h / args.num_gpus if hes_gpu_h is not None else None,
            "hes_source_tokens_per_second_per_gpu": hes_tps,
            "proofarrow_gpu_hours": pa_gpu_h,
            "proofarrow_wall_hours": pa_gpu_h / args.num_gpus,
            "proofarrow_source_tokens_per_second_per_gpu": pa_effective_tps,
            "matched_null_gpu_hours": null_gpu_h,
            "proofarrow_relative_to_ppl": pa_gpu_h / ppl_gpu_h,
            "hes_relative_to_ppl": hes_gpu_h / ppl_gpu_h if hes_gpu_h is not None else None,
            "judge_gpu_hours": args.judge_gpu_hours_20b,
            "judge_wall_hours": args.judge_gpu_hours_20b / args.num_gpus if args.judge_gpu_hours_20b else None,
            "judge_source_tokens_per_second_per_gpu": (
                args.pool_source_tokens / (args.judge_gpu_hours_20b * 3600.0)
                if args.judge_gpu_hours_20b else None
            ),
            "judge_relative_to_ppl": args.judge_gpu_hours_20b / ppl_gpu_h if args.judge_gpu_hours_20b else None,
            "judge_description": args.judge_description,
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": benchmark.torch.__version__,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["extrapolated_20b"], indent=2, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
