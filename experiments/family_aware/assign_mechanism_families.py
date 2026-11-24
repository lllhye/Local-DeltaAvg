#!/usr/bin/env python3
"""Assign frozen-probe zero-shot mechanism probabilities to ProofArrow traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="JSONL with id, question, peak_window, score, token_count")
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--taxonomy", default=str(Path(__file__).with_name("taxonomy.json")))
    p.add_argument("--id-key", default="id")
    p.add_argument("--question-key", default="question")
    p.add_argument("--window-key", default="peak_window")
    p.add_argument("--score-key", default="proofarrow_score")
    p.add_argument(
        "--score-keys",
        nargs="+",
        help=(
            "Numeric selector fields copied to the assignment output. "
            "Defaults to --score-key; pass proofarrow_score hes_score "
            "askllm_score to reuse one assignment run across controls."
        ),
    )
    p.add_argument("--token-count-key", default="token_count")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--confidence-margin", type=float, default=0.15)
    p.add_argument("--device", default="cuda", choices=("cuda", "mps", "cpu"))
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def build_prefix(taxonomy: list[dict[str, str]]) -> str:
    labels = "\n".join(f"{x['code']}: {x['description']}" for x in taxonomy)
    return (
        "Choose the single prerequisite mechanism that best describes the critical reasoning window.\n"
        "Return exactly one code.\nMechanisms:\n" + labels + "\n"
    )


def prompt(prefix: str, question: str, window: str) -> str:
    return f"{prefix}\nQuestion:\n{question}\nCritical window:\n{window}\nCode:"


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            yield row


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise SystemExit("batch size and max length must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    if len(taxonomy) != 24:
        raise SystemExit("the paper protocol requires exactly 24 mechanism codes")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    code_ids = []
    for item in taxonomy:
        ids = tokenizer.encode(" " + item["code"], add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"code {item['code']} is not one token for this tokenizer")
        code_ids.append(ids[0])

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()
    prefix = build_prefix(taxonomy)
    rows = iter_rows(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    score_keys = args.score_keys or [args.score_key]
    with output.open("w", encoding="utf-8") as out:
        while True:
            batch = []
            try:
                for _ in range(args.batch_size):
                    batch.append(next(rows))
            except StopIteration:
                pass
            if not batch:
                break
            texts = [
                prompt(prefix, str(x[args.question_key]), str(x[args.window_key]))
                for x in batch
            ]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(args.device)
            with torch.inference_mode():
                logits = model(**encoded).logits[:, -1, code_ids].float()
                probs = torch.softmax(logits, dim=-1).cpu()
            top2 = torch.topk(probs, k=2, dim=-1)
            for i, row in enumerate(batch):
                top_idx = int(top2.indices[i, 0])
                margin = float(top2.values[i, 0] - top2.values[i, 1])
                record = {
                    "id": str(row.get(args.id_key, row["_line_no"])),
                    "token_count": int(row[args.token_count_key]),
                    "family_codes": [x["code"] for x in taxonomy],
                    "family_probs": [round(float(v), 8) for v in probs[i]],
                    "top_family": taxonomy[top_idx]["code"],
                    "top_probability": float(top2.values[i, 0]),
                    "posterior_margin": margin,
                    "high_confidence": margin >= args.confidence_margin,
                }
                for key in score_keys:
                    if key not in row:
                        raise KeyError(f"line {row['_line_no']}: missing score field {key}")
                    record[key] = float(row[key])
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(batch) < args.batch_size:
                break
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
