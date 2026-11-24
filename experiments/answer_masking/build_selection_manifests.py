#!/usr/bin/env python3
"""Null-center answer-masking scores and build fixed-token CPT manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev


DEFAULT_VARIANTS = (
    "answer_first_raw",
    "answer_loss_masked",
    "answer_context_masked",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True, help="Scores on the natural candidate pool")
    p.add_argument("--null-scores", required=True, help="Scores on the matched-null pool")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--budget", type=int, default=2_000_000_000)
    p.add_argument("--length-bin-width", type=int, default=256)
    p.add_argument("--window-bin-width", type=int, default=32)
    p.add_argument("--min-null-bin", type=int, default=20)
    p.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    return p.parse_args()


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bin_key(row: dict[str, str], variant: str, length_width: int, window_width: int) -> tuple[int, int]:
    length_bin = int(row["token_count"]) // length_width
    window_bin = int(row[f"{variant}_n_windows"]) // window_width
    return length_bin, window_bin


def fit_null(
    null_rows: list[dict[str, str]],
    variant: str,
    length_width: int,
    window_width: int,
) -> tuple[dict[tuple[int, int], list[float]], dict[int, list[float]], list[float]]:
    exact: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_length: dict[int, list[float]] = defaultdict(list)
    all_values: list[float] = []
    score_key = f"{variant}_score"
    for row in null_rows:
        value = float(row[score_key])
        key = bin_key(row, variant, length_width, window_width)
        exact[key].append(value)
        by_length[key[0]].append(value)
        all_values.append(value)
    if not all_values:
        raise ValueError("null score file is empty")
    return exact, by_length, all_values


def null_stats(
    row: dict[str, str],
    variant: str,
    exact: dict[tuple[int, int], list[float]],
    by_length: dict[int, list[float]],
    all_values: list[float],
    length_width: int,
    window_width: int,
    min_count: int,
) -> tuple[float, float, str, int]:
    key = bin_key(row, variant, length_width, window_width)
    if len(exact.get(key, [])) >= min_count:
        values, source = exact[key], "length+windows"
    elif len(by_length.get(key[0], [])) >= min_count:
        values, source = by_length[key[0]], "length"
    else:
        values, source = all_values, "global"
    scale = pstdev(values) if len(values) > 1 else 0.0
    return fmean(values), scale, source, len(values)


def build_manifest(
    rows: list[dict[str, str]],
    null_rows: list[dict[str, str]],
    variant: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    score_key = f"{variant}_score"
    windows_key = f"{variant}_n_windows"
    for required in (score_key, windows_key):
        if rows and required not in rows[0]:
            raise KeyError(f"missing column {required} in --scores")
        if null_rows and required not in null_rows[0]:
            raise KeyError(f"missing column {required} in --null-scores")

    exact, by_length, all_values = fit_null(
        null_rows, variant, args.length_bin_width, args.window_bin_width
    )
    ranked: list[dict[str, object]] = []
    for row in rows:
        raw = float(row[score_key])
        mean, scale, source, null_n = null_stats(
            row,
            variant,
            exact,
            by_length,
            all_values,
            args.length_bin_width,
            args.window_bin_width,
            args.min_null_bin,
        )
        ranked.append({
            "id": row["id"],
            "token_count": int(row["token_count"]),
            "raw_score": raw,
            "null_mean": mean,
            "null_scale": scale,
            "centered_score": raw - mean,
            "null_source": source,
            "null_n": null_n,
        })
    ranked.sort(key=lambda x: (-float(x["centered_score"]), str(x["id"])))

    selected: list[dict[str, object]] = []
    used = 0
    for item in ranked:
        size = int(item["token_count"])
        if size <= 0:
            continue
        if used + size <= args.budget:
            item["rank"] = len(selected) + 1
            selected.append(item)
            used += size

    manifest = output_dir / f"{variant}_centered_{args.budget}.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {
        "variant": variant,
        "manifest": str(manifest),
        "selected_examples": len(selected),
        "selected_tokens": used,
        "budget": args.budget,
        "unused_tokens": args.budget - used,
    }


def main() -> None:
    args = parse_args()
    if args.budget <= 0 or args.length_bin_width <= 0 or args.window_bin_width <= 0:
        raise SystemExit("budget and bin widths must be positive")
    rows = read_csv(args.scores)
    null_rows = read_csv(args.null_scores)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [build_manifest(rows, null_rows, v, args, output_dir) for v in args.variants]
    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
