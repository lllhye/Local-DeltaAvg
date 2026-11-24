#!/usr/bin/env python3
"""Lazy-greedy maximization of the concave Local-DeltaAvg-Family objective."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Output of assign_mechanism_families.py")
    p.add_argument("--manifest", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--budget-tokens", type=int, default=2_000_000_000)
    p.add_argument("--active-threshold", type=float, default=0.005)
    p.add_argument("--score-key", default="proofarrow_score")
    p.add_argument(
        "--exposure-mode",
        choices=("score", "label-only"),
        default="score",
        help="Use positive selector score times family probabilities, or probabilities alone.",
    )
    p.add_argument("--run-name", default="family_aware")
    return p.parse_args()


def gain(exposure: np.ndarray, current: np.ndarray, tau: np.ndarray) -> float:
    return float(np.log1p((current + exposure) / tau).sum() - np.log1p(current / tau).sum())


def main() -> None:
    args = parse_args()
    if args.budget_tokens <= 0:
        raise SystemExit("budget must be positive")
    ids: list[str] = []
    tokens: list[int] = []
    scores: list[float] = []
    top_family: list[str] = []
    exposures: list[np.ndarray] = []
    codes: list[str] | None = None
    with Path(args.input).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row_codes = list(row["family_codes"])
            if codes is None:
                codes = row_codes
            elif row_codes != codes:
                raise SystemExit("inconsistent family-code order")
            if args.exposure_mode == "label-only":
                score = 1.0
            else:
                score = max(float(row[args.score_key]), 0.0)
            probs = np.asarray(row["family_probs"], dtype=np.float64)
            ids.append(str(row["id"]))
            tokens.append(int(row["token_count"]))
            scores.append(score)
            top_family.append(str(row["top_family"]))
            exposures.append(score * probs)
    if not exposures or codes is None:
        raise SystemExit("no candidates")
    x = np.stack(exposures)
    token_array = np.asarray(tokens, dtype=np.int64)
    tau = np.median(x[x.sum(axis=1) > 0], axis=0)
    positive = x[x > 0]
    fallback = float(np.median(positive)) if positive.size else 1.0
    tau = np.maximum(tau, max(fallback * 1e-3, 1e-12))
    current = np.zeros(x.shape[1], dtype=np.float64)

    heap: list[tuple[float, int]] = []
    for i in range(len(ids)):
        initial = gain(x[i], current, tau) / max(tokens[i], 1)
        heapq.heappush(heap, (-initial, i))
    selected: list[tuple[int, float]] = []
    used = 0
    while heap:
        _, i = heapq.heappop(heap)
        if used + tokens[i] > args.budget_tokens:
            continue
        exact = gain(x[i], current, tau) / max(tokens[i], 1)
        next_bound = -heap[0][0] if heap else -math.inf
        if exact + 1e-15 < next_bound:
            heapq.heappush(heap, (-exact, i))
            continue
        current += x[i]
        used += tokens[i]
        selected.append((i, exact))
        if used >= args.budget_tokens:
            break

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        for rank, (i, marginal) in enumerate(selected, 1):
            record = {
                "rank": rank,
                "id": ids[i],
                "token_count": tokens[i],
                "selection_score": scores[i],
                "score_key": args.score_key if args.exposure_mode == "score" else "label_only",
                "top_family": top_family[i],
                "marginal_gain_per_token": marginal,
            }
            if args.exposure_mode == "score":
                record[args.score_key] = scores[i]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    share = current / max(float(current.sum()), 1e-12)
    active = int((share >= args.active_threshold).sum())
    sorted_share = np.sort(share)
    n = len(sorted_share)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(sorted_share) / max(n * sorted_share.sum(), 1e-12))
    summary = {
        "run_name": args.run_name,
        "score_key": args.score_key,
        "exposure_mode": args.exposure_mode,
        "selected_examples": len(selected),
        "selected_tokens": used,
        "budget_tokens": args.budget_tokens,
        "family_codes": codes,
        "tau": tau.tolist(),
        "selected_exposure": current.tolist(),
        "active_families": active,
        "active_threshold": args.active_threshold,
        "exposure_gini": gini,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_tokens": used, "active_families": active, "exposure_gini": gini}, indent=2))


if __name__ == "__main__":
    main()
