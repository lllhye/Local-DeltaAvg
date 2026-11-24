#!/usr/bin/env python3
"""Write measured answer-masking results into the paper's LaTeX macros."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("GSM8K", "MATH500", "AIME24", "GPQA-D", "LCB")
ROWS = (
    ("answer_loss_masked_centered", "MaskLoss"),
    ("answer_context_masked_centered", "MaskContext"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, help="Measured evaluation JSON")
    p.add_argument(
        "--output",
        default="outputs/tables/answer_masking_results.tex",
        help="Output LaTeX macro file",
    )
    return p.parse_args()


def value(row: dict[str, object], task: str) -> float:
    aliases = {"GPQA-D": ("GPQA-D", "GPQAD", "gpqa_d"), "LCB": ("LCB", "LiveCodeBench", "livecodebench")}
    for key in aliases.get(task, (task, task.lower())):
        if key in row:
            return float(row[key])
    raise KeyError(f"missing task {task}; found {sorted(row)}")


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    lines = [
        "% Auto-generated measured results. Do not edit by hand.",
    ]
    for result_key, macro in ROWS:
        if result_key not in data:
            raise KeyError(f"missing result row {result_key}")
        values = [value(data[result_key], task) for task in TASKS]
        avg = sum(values) / len(values)
        for suffix, number in zip(("GSM", "MATH", "AIME", "GPQA", "LCB"), values):
            lines.append(f"\\newcommand{{\\{macro}{suffix}}}{{{number:.1f}}}")
        lines.append(f"\\newcommand{{\\{macro}Avg}}{{{avg:.1f}}}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote measured results to {output}")


if __name__ == "__main__":
    main()
