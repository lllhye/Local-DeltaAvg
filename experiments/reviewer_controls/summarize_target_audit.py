#!/usr/bin/env python3
"""Summarize two-annotator target-invariance labels and adjudicated outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


TRUE = {"1", "true", "yes", "y", "pass"}
FALSE = {"0", "false", "no", "n", "fail"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def boolean(value: str, field: str) -> bool:
    v = value.strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    raise ValueError(f"{field}: expected pass/fail, got {value!r}")


def wilson(success: int, n: int, z: float = 1.95996398454) -> tuple[float, float]:
    p = success / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n
    pb = sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main() -> None:
    args = parse_args()
    with Path(args.input).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("empty audit sheet")
    a1, a2, final = [], [], []
    by_domain: dict[str, list[bool]] = defaultdict(list)
    failures: Counter[str] = Counter()
    for row in rows:
        a1.append(boolean(row["ann1_pass"], "ann1_pass"))
        a2.append(boolean(row["ann2_pass"], "ann2_pass"))
        verdict = boolean(row["adjudicated_pass"], "adjudicated_pass")
        final.append(verdict)
        by_domain[row["domain"]].append(verdict)
        if not verdict:
            failures[row["adjudicated_failure"].strip() or "unspecified"] += 1
    successes = sum(final)
    low, high = wilson(successes, len(final))
    result = {
        "n": len(final),
        "overall_pass_percent": 100 * successes / len(final),
        "ci95_low_percent": 100 * low,
        "ci95_high_percent": 100 * high,
        "kappa": kappa(a1, a2),
        "domains": {
            domain: {"n": len(values), "pass_percent": 100 * sum(values) / len(values)}
            for domain, values in sorted(by_domain.items())
        },
        "failures": dict(failures),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
