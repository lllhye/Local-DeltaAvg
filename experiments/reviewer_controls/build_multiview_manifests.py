#!/usr/bin/env python3
"""Build fixed-token single-view, multi-view mean, and multi-view LCB manifests."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="JSONL with id, token_count, and centered view_scores")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--budget", type=int, default=2_000_000_000)
    p.add_argument("--id-key", default="id")
    p.add_argument("--token-count-key", default="token_count")
    p.add_argument("--views-key", default="view_scores")
    p.add_argument("--lcb-lambda", type=float, default=1.0)
    return p.parse_args()


def select(rows: list[dict[str, object]], score_key: str, budget: int, path: Path) -> dict[str, object]:
    ranked = sorted(rows, key=lambda x: (-float(x[score_key]), str(x["id"])))
    used = 0
    chosen = []
    for row in ranked:
        size = int(row["token_count"])
        if size > 0 and used + size <= budget:
            chosen.append(row)
            used += size
    with path.open("w", encoding="utf-8") as f:
        for rank, row in enumerate(chosen, 1):
            f.write(json.dumps({
                "rank": rank,
                "id": row["id"],
                "token_count": row["token_count"],
                "selection_score": row[score_key],
                "score_type": score_key,
                "n_views": row["n_views"],
            }, ensure_ascii=False) + "\n")
    return {"manifest": str(path), "score_type": score_key, "selected_tokens": used, "selected_examples": len(chosen)}


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise SystemExit("budget must be positive")
    rows = []
    with Path(args.input).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            src = json.loads(line)
            scores = [float(v) for v in src[args.views_key]]
            if not scores:
                raise ValueError(f"line {line_no}: empty view_scores")
            sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
            mean = statistics.fmean(scores)
            rows.append({
                "id": str(src.get(args.id_key, line_no)),
                "token_count": int(src[args.token_count_key]),
                "n_views": len(scores),
                "single_view": scores[0],
                "five_view_mean": mean,
                "five_view_lcb": mean - args.lcb_lambda * sd / math.sqrt(len(scores)),
            })
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = [
        select(rows, key, args.budget, out / f"{key}_{args.budget}.jsonl")
        for key in ("single_view", "five_view_mean", "five_view_lcb")
    ]
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
