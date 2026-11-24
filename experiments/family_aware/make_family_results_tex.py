#!/usr/bin/env python3
"""Write measured natural-pool family-aware result macros for the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--outputs", nargs="+", default=["outputs/tables/family_aware_results.tex"])
    return p.parse_args()


def macro(name: str, value: object) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def number(value: object, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def percent(value: object) -> str:
    return number(value, 1) + "\\%"


def main() -> None:
    args = parse_args()
    d = json.loads(Path(args.results).read_text(encoding="utf-8"))
    tasks = d["tasks"]
    a = d["assignment"]
    q = d["acquisition"]
    fit = d["natural_fit"]
    cost = d["cost"]
    scalar_avg = float(d["scalar_average"])
    avg = float(tasks["average"])
    lines = [
        "% Auto-generated measured natural-pool family-aware results.",
        macro("FamilyMark", ""),
        macro("FamilyEvidenceVerb", "reaches"),
        macro("FamilyStatusSentence", ""),
        macro("FamilyCaptionStatus", ""),
        macro("FamilyGSM", number(tasks["GSM8K"])),
        macro("FamilyMATH", number(tasks["MATH500"])),
        macro("FamilyAIME", number(tasks["AIME24"])),
        macro("FamilyGPQA", number(tasks["GPQA-D"])),
        macro("FamilyLCB", number(tasks["LCB"])),
        macro("FamilyAvg", number(avg)),
        macro("FamilyGain", number(avg - scalar_avg)),
        macro("FamilyAssignGPUh", number(cost["assignment_gpu_hours"], 0)),
        macro("FamilyTotalGPUh", number(cost["total_gpu_hours"], 0)),
        macro("FamilyWallh", number(cost["wall_hours"])),
        macro("FamilyThroughput", f"{float(cost['source_tokens_per_second_per_gpu']):,.0f}"),
        macro("FamilyRelativeCost", number(cost["relative_to_ppl"], 2)),
        macro("FamilyAuditN", int(a["audit_n"])),
        macro("FamilyAssignCoverage", percent(a["high_confidence_percent"])),
        macro("FamilyTopOneAccuracy", percent(a["top1_accuracy_percent"])),
        macro("FamilyMacroFOne", number(a["macro_f1"], 2)),
        macro("FamilyKappa", number(a["kappa"], 2)),
        macro("FamilyCrossProbe", percent(a["cross_probe_agreement_percent"])),
        macro("FamilyPromptStability", percent(a["prompt_stability_percent"])),
        macro("ScalarActiveMechanisms", f"{q['scalar_active_families']}/24"),
        macro("FamilyActiveMechanisms", f"{q['family_active_families']}/24"),
        macro("ScalarWorstFamily", number(q["scalar_worst_family_accuracy"])),
        macro("FamilyWorstFamily", number(q["family_worst_family_accuracy"])),
        macro("ScalarExposureGini", number(q["scalar_exposure_gini"], 2)),
        macro("FamilyExposureGini", number(q["family_exposure_gini"], 2)),
        macro("NaturalNegativeSlopes", f"{fit['negative_slope_families']}/24"),
        macro("NaturalConcavePreferred", f"{fit['concave_preferred_families']}/24"),
        macro("NaturalConcaveRSq", number(fit["concave_median_r2"], 2)),
        macro("NaturalLinearRSq", number(fit["linear_median_r2"], 2)),
        macro("NaturalPartialRSq", number(fit["partial_median_r2"], 2)),
        macro("NaturalInterferenceP", number(fit["interference_p95_nll"], 2)),
    ]
    content = "\n".join(lines) + "\n"
    for name in args.outputs:
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
