#!/usr/bin/env python3
"""Write measured engineering macros for the main paper and supplement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True)
    p.add_argument(
        "--outputs",
        nargs="+",
        default=["outputs/tables/engineering_cost_results.tex"],
    )
    return p.parse_args()


def fmt(value: object, digits: int = 1, missing: str = "--") -> str:
    if value is None:
        return missing
    return f"{float(value):,.{digits}f}"


def pct(value: object) -> str:
    return fmt(value, 1) + "\\%"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    cfg = data["configuration"]
    ext = data["extrapolated_20b"]
    parsing = data["parsing"]
    integrity = data["integrity"]
    gpus = int(cfg["num_gpus_for_wall_clock"])
    cpu_workers = int(cfg.get("num_cpu_workers_for_wall_clock", 16))
    hardware = str(cfg["device"]).replace("&", "\\&").replace("_", "\\_")
    status = (
        f"the measured configuration with {gpus}\\(\\times\\){hardware}, per-device batch size {cfg['batch_size']}, "
        f"a {cfg['max_length']}-token context, and {cpu_workers} CPU workers"
    )
    ppl_gpu = float(ext["forward_ppl_gpu_hours"])
    hes_gpu = ext.get("hes_gpu_hours")
    judge_gpu = ext.get("judge_gpu_hours")

    lines = [
        "% Auto-generated measured engineering profile. Do not edit by hand.",
        macro("EngineeringMark", ""),
        macro("EngineeringVerb", "requires"),
        macro("EngineeringStatusText", status),
        macro("EngineeringQualification", "Full-pool costs are extrapolated from the supplied shard benchmark."),
        macro("EngineeringCaptionNote", ""),
        macro("EngineeringHardware", hardware),
        macro("LengthGPUh", "0"),
        macro("LengthWallh", fmt(float(parsing["extrapolated_cpu_hours_20b"]) / cpu_workers)),
        macro("LengthThroughput", "--"),
        macro("LengthRelative", "--"),
        macro("PPLGPUh", fmt(ppl_gpu, 0)),
        macro("PPLWallh", fmt(ext["forward_ppl_wall_hours"])),
        macro("PPLThroughput", fmt(ext["forward_ppl_source_tokens_per_second_per_gpu"], 0)),
        macro("PPLRelative", "1.00"),
        macro("HESGPUh", fmt(hes_gpu, 0)),
        macro("HESWallh", fmt(ext.get("hes_wall_hours"))),
        macro("HESThroughput", fmt(ext.get("hes_source_tokens_per_second_per_gpu"), 0)),
        macro("HESRelative", fmt(ext.get("hes_relative_to_ppl"), 2)),
        macro("LDAGPUh", fmt(ext["proofarrow_gpu_hours"], 0)),
        macro("LDAWallh", fmt(ext["proofarrow_wall_hours"])),
        macro("LDAThroughput", fmt(ext["proofarrow_source_tokens_per_second_per_gpu"], 0)),
        macro("LDARelative", fmt(ext["proofarrow_relative_to_ppl"], 2)),
        macro("JudgeGPUh", fmt(judge_gpu, 0)),
        macro("JudgeWallh", fmt(ext.get("judge_wall_hours"))),
        macro("JudgeThroughput", fmt(ext.get("judge_source_tokens_per_second_per_gpu"), 0)),
        macro("JudgeRelative", fmt(ext.get("judge_relative_to_ppl"), 2)),
        macro("SegmentCPUh", fmt(parsing["extrapolated_cpu_hours_20b"])),
        macro("NullGPUh", fmt(ext["matched_null_gpu_hours"])),
        macro("SegmentSuccess", pct(parsing["accepted_rate_percent"])),
        macro("RejectedRate", pct(parsing["rejected_rate_percent"])),
        macro("UnitCoverage", pct(integrity["unit_coverage_percent"])),
        macro("DroppedDuplicated", pct(integrity["dropped_or_duplicated_units_percent"])),
        macro("AlignmentFailure", pct(integrity["unit_id_alignment_failure_percent"])),
        macro("ChunkingRate", pct(integrity["requires_chunking_percent"])),
        macro("AnswerUnchanged", pct(integrity["answer_unchanged_percent"])),
    ]
    content = "\n".join(lines) + "\n"
    for output_name in args.outputs:
        output = Path(output_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
