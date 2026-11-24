#!/usr/bin/env python3
"""Convert target-invariance and acquisition-control measurements to TeX macros."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("GSM8K", "MATH500", "AIME24", "GPQA-D", "LCB")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--outputs", nargs="+", default=["outputs/tables/reviewer_control_results.tex"])
    return p.parse_args()


def macro(name: str, value: object) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def fmt(value: object, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: object) -> str:
    return fmt(value) + "\\%"


def average(row: dict[str, object]) -> float:
    return sum(float(row[k]) for k in TASKS) / len(TASKS)


def reported_average(row: dict[str, object]) -> float:
    """Use the one-decimal task average reported in the paper."""
    return round(average(row), 1)


def task_macros(prefix: str, row: dict[str, object]) -> list[str]:
    names = ("GSM", "MATH", "AIME", "GPQA", "LCB")
    lines = [macro(prefix + name, fmt(row[key])) for name, key in zip(names, TASKS)]
    lines.append(macro(prefix + "Avg", fmt(reported_average(row))))
    if "sd" in row:
        lines.append(macro(prefix + "SD", fmt(row["sd"])))
    return lines


def main() -> None:
    args = parse_args()
    d = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if d.get("status") != "measured":
        raise SystemExit('results status must be "measured" before paper macros are generated')
    audit = d["target_audit"]
    safe = d["context_safe"]
    views = d["view_training"]
    controls = d["acquisition_controls"]
    domains = {name.lower(): result for name, result in audit["domains"].items()}
    failures = audit["failures"]

    lines = [
        "% Auto-generated reviewer-control results.",
        macro("ReviewerMark", ""),
        macro("ReviewerStatus", ""),
        macro("ReviewerEvidence", "measured"),
        macro("TargetAuditN", int(audit["n"])),
        macro("TargetAuditPass", pct(audit["overall_pass_percent"])),
        macro("TargetAuditCI", f"[{fmt(audit['ci95_low_percent'])}, {fmt(audit['ci95_high_percent'])}]\\%"),
        macro("TargetAuditKappa", fmt(audit["kappa"], 2)),
        macro("TargetMathPass", pct(domains["math"]["pass_percent"])),
        macro("TargetCodePass", pct(domains["code"]["pass_percent"])),
        macro("TargetSciencePass", pct(domains["science"]["pass_percent"])),
        macro("TargetLogicPass", pct(domains["logic"]["pass_percent"])),
        macro("TargetAnaphoraN", int(failures["anaphora"])),
        macro("TargetVariableN", int(failures["local_variable"])),
        macro("TargetStateN", int(failures["branch_or_code_state"])),
        macro("TargetDiscourseN", int(failures["discourse_ellipsis"])),
        macro("SafeRetained", pct(safe["retained_percent"])),
        macro("SafeSemanticPass", pct(safe["semantic_pass_percent"])),
        macro("SafeSpearman", fmt(safe["score_spearman"], 2)),
        macro("SafeTopTen", fmt(safe["top10_overlap"], 2)),
        macro("SafeTopTwenty", fmt(safe["top20_overlap"], 2)),
        macro("SafeUtilityRaw", fmt(safe["utility_correlation_raw"], 2)),
        macro("SafeUtility", fmt(safe["utility_correlation_safe"], 2)),
        macro("SafeIoURaw", fmt(safe["critical_step_iou_raw"], 2)),
        macro("SafeIoU", fmt(safe["critical_step_iou_safe"], 2)),
    ]
    for prefix, key in (
        ("ViewSingle", "single"),
        ("ViewSafe", "context_safe"),
        ("ViewMean", "five_view_mean"),
        ("ViewLCB", "five_view_lcb"),
    ):
        lines.extend(task_macros(prefix, views[key]))
    for prefix, key in (
        ("CtlHESScalar", "hes_scalar"),
        ("CtlHESFamily", "hes_mechanism"),
        ("CtlAskScalar", "askllm_scalar"),
        ("CtlAskFamily", "askllm_mechanism"),
        ("CtlLabelOnly", "mechanism_only"),
        ("CtlDirScalar", "directional_scalar"),
        ("CtlDirSTEM", "directional_stem"),
        ("CtlLDAFamily", "proofarrow"),
    ):
        lines.extend(task_macros(prefix, controls[key]))

    hes_gain = reported_average(controls["hes_mechanism"]) - reported_average(controls["hes_scalar"])
    pa_gain = reported_average(controls["proofarrow"]) - reported_average(controls["directional_scalar"])
    lines.extend([
        macro("CtlHESGain", fmt(hes_gain)),
        macro("CtlAskGain", fmt(reported_average(controls["askllm_mechanism"]) - reported_average(controls["askllm_scalar"]))),
        macro("CtlSTEMGain", fmt(reported_average(controls["directional_stem"]) - reported_average(controls["directional_scalar"]))),
        macro("CtlInteraction", fmt(pa_gain - hes_gain)),
        macro("CtlTaxonomyGap", fmt(reported_average(controls["proofarrow"]) - reported_average(controls["directional_stem"]))),
    ])

    content = "\n".join(lines) + "\n"
    for output in args.outputs:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
