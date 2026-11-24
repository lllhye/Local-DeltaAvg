#!/usr/bin/env python3
"""Create a stratified, blinded target-invariance audit sheet."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_QUOTAS = {"math": 300, "code": 250, "science": 250, "logic": 200}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--domain-key", default="domain")
    p.add_argument("--id-key", default="id")
    p.add_argument("--question-key", default="question")
    p.add_argument("--steps-key", default="steps")
    p.add_argument("--quotas-json", help="JSON object such as '{\"math\":300,...}'")
    p.add_argument("--seed", type=int, default=2027)
    return p.parse_args()


def normalize_steps(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).splitlines() if x.strip()]


def main() -> None:
    args = parse_args()
    quotas = json.loads(args.quotas_json) if args.quotas_json else DEFAULT_QUOTAS
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    with Path(args.input).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            domain = str(row.get(args.domain_key, "")).strip().lower()
            if domain not in quotas:
                continue
            steps = normalize_steps(row[args.steps_key])
            if len(steps) < 2:
                continue
            by_domain[domain].append({
                "id": str(row.get(args.id_key, line_no)),
                "domain": domain,
                "question": str(row[args.question_key]),
                "forward_trace": "\n".join(f"[{i+1}] {s}" for i, s in enumerate(steps)),
                "reversed_trace": "\n".join(f"[{len(steps)-i}] {s}" for i, s in enumerate(reversed(steps))),
            })
    rng = random.Random(args.seed)
    sampled = []
    for domain, quota in quotas.items():
        rows = by_domain.get(domain, [])
        if len(rows) < int(quota):
            raise SystemExit(f"domain {domain}: need {quota}, found {len(rows)}")
        sampled.extend(rng.sample(rows, int(quota)))
    rng.shuffle(sampled)
    fields = [
        "id", "domain", "question", "forward_trace", "reversed_trace",
        "ann1_pass", "ann1_failure", "ann2_pass", "ann2_failure",
        "adjudicated_pass", "adjudicated_failure", "notes",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sampled:
            writer.writerow(row)
    print(json.dumps({"output": str(output), "n": len(sampled), "quotas": quotas}, indent=2))


if __name__ == "__main__":
    main()
