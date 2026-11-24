#!/usr/bin/env python3
"""Build a deterministic context-safe subset for reversal robustness tests.

The script removes only leading discourse connectives in both views and rejects
traces containing unresolved context-dependent references. It never generates
new reasoning content, so the forward and reversed target spans remain equal.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LEADING_MARKER = re.compile(
    r"^\s*(therefore|thus|hence|consequently|so|accordingly|as a result)\s*[,;:]?\s*",
    re.IGNORECASE,
)
UNSAFE = {
    "anaphora": re.compile(r"\b(this|that|these|those)\s+(value|result|expression|equation|case|step|claim|quantity|variable)\b|\b(the former|the latter|the above|below)\b", re.I),
    "local_variable": re.compile(r"\b(the current|the previous|the newly defined|as defined|defined above|aforementioned)\s+(variable|value|term|symbol|function|object)\b", re.I),
    "branch_or_code_state": re.compile(r"\b(otherwise|in this branch|in the other branch|current state|previous state|after the update|before the update)\b", re.I),
    "discourse_ellipsis": re.compile(r"\b(by the same argument|similarly as above|continuing|substituting this|using this)\b", re.I),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--accepted", required=True)
    p.add_argument("--rejected", required=True)
    p.add_argument("--steps-key", default="steps")
    p.add_argument("--id-key", default="id")
    return p.parse_args()


def steps(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).splitlines() if x.strip()]


def main() -> None:
    args = parse_args()
    accepted_path, rejected_path = Path(args.accepted), Path(args.rejected)
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    total = kept = 0
    reasons: dict[str, int] = {key: 0 for key in UNSAFE}
    with Path(args.input).open(encoding="utf-8") as src, accepted_path.open("w", encoding="utf-8") as good, rejected_path.open("w", encoding="utf-8") as bad:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            original = steps(row[args.steps_key])
            normalized = [LEADING_MARKER.sub("", unit).strip() for unit in original]
            matched = sorted({name for name, pattern in UNSAFE.items() if any(pattern.search(unit) for unit in normalized)})
            if matched or len(normalized) < 2 or any(not unit for unit in normalized):
                for name in matched:
                    reasons[name] += 1
                bad.write(json.dumps({
                    "id": str(row.get(args.id_key, line_no)),
                    "rejection_reasons": matched or ["malformed"],
                }, ensure_ascii=False) + "\n")
                continue
            row[args.steps_key] = normalized
            row["context_safe"] = True
            good.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(json.dumps({
        "total": total,
        "accepted": kept,
        "retained_percent": 100 * kept / total if total else 0.0,
        "rejection_counts": reasons,
    }, indent=2))


if __name__ == "__main__":
    main()
