"""Make the Local-DeltaAvg experiment figures from the values in the handoff manuscript.

Run with Python 3 and Matplotlib 3.7+ using: python plot_experiments.py
Each data block names its source table in main.tex.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


# Table 1. The first five columns are benchmarks and the last is the reported mean.
BENCHMARKS = ["GSM8K", "MATH500", "AIME24", "GPQA-D", "LCB"]
MAIN_RESULTS = {
    "Random-STEM": (66.5, 52.5, 12.2, 45.5, 52.5, 45.8),
    "Forward PPL/loss": (67.5, 53.5, 12.2, 46.8, 52.5, 46.5),
    "LLM-as-Judge": (68.5, 54.3, 13.3, 46.4, 52.2, 46.9),
    "AskLLM": (68.8, 55.0, 13.3, 46.3, 53.2, 47.3),
    "HES": (69.5, 56.0, 14.4, 46.8, 53.5, 48.0),
    "ASLEC-CASL": (70.2, 57.0, 14.4, 47.0, 53.8, 48.5),
    "Answer-first, raw": (71.5, 58.5, 16.7, 47.3, 54.0, 49.6),
    "Answer-first, centered": (72.1, 59.0, 16.7, 47.6, 54.3, 49.9),
    "Answer-loss-masked": (72.2, 59.1, 16.7, 47.6, 54.3, 50.0),
    "Local-DeltaAvg": (72.6, 59.4, 17.8, 47.8, 54.5, 50.4),
    "Safe": (72.8, 59.7, 17.8, 47.9, 54.7, 50.6),
    "MV-Mean": (72.9, 59.9, 17.8, 48.0, 54.8, 50.7),
    "MV-Conservative": (73.0, 60.0, 17.8, 48.1, 54.9, 50.8),
    "Family": (73.4, 60.5, 17.8, 48.5, 55.1, 51.1),
}

# Table 2 and Table S5. The matched pilot uses its internal random reference.
STRUCTURE_SCORES = {"Full CoT": 0.54, "Compressed": 0.34, "Direct answer": 0.12, "Shuffled": 0.07}
PILOT_ACCURACY = {
    "Semantic": 46.7, "Sentence": 43.6, "Token": 42.2, "Shuffled": 43.1, "Random": 42.8,
}

# Table 3. Spearman correlations with trace length and window count.
CALIBRATION = {
    "Raw max": (0.14, 0.18),
    "Length-stratified": (0.03, 0.04),
    "Null-centered": (0.02, 0.03),
    "Standardized": (-0.01, 0.01),
}

# Tables 5, 6, S3. Coverage values are (scalar, family).
COVERAGE = {"Active mechanisms": (17, 23), "Exposure Gini": (0.41, 0.24), "Worst-family accuracy": (42.9, 45.7)}
GROUPED_MEANS = {"HES": (48.0, 48.3), "AskLLM": (47.3, 47.5), "Directional": (50.4, 51.1)}
SCORING_COST = {
    "Forward PPL/loss": 390, "HES": 420, "Local-DeltaAvg": 800, "Family": 875,
    "LLM-as-Judge": 4800, "AskLLM": 4800,
}

# Tables S8 and S9. Probe pairs are (main, independent). Margin rows are (rank flips, retention) in percent.
VIEW_AGREEMENT = {
    "Pairwise Spearman": (0.79, 0.73),
    "Single-view ICC": (0.48, 0.39),
    "Five-view ICC": (0.82, 0.76),
}
MARGIN = {"Q1": (25.0, 58), "Q2": (14.2, 70), "Q3": (7.1, 82), "Q4": (3.5, 91)}

# Tables S14 and S16. Exposure rows are (NLL, lower CI, upper CI).
# Controlled selection rows are (accuracy mean, across-seed SD).
EXPOSURE = {
    0: (0.80, 0.77, 0.83), 1: (0.56, 0.53, 0.59), 2: (0.41, 0.38, 0.44),
    3: (0.32, 0.29, 0.35), 4: (0.27, 0.24, 0.30),
}
CONTROLLED_SELECTION = {
    "Random": (63.0, 0.8), "Raw top-score": (68.8, 0.6),
    "Mechanism-aware": (72.1, 0.5), "Oracle": (73.4, 0.4),
}

# Table S23. Rows are (seed, five-task average, AIME24 count correct).
SEED_RESULTS = {
    "Random-STEM": [(17, 45.7, 4), (29, 46.1, 4), (43, 45.7, 3)],
    "Local-DeltaAvg": [(17, 50.0, 5), (29, 50.9, 6), (43, 50.4, 5)],
    "Family": [(17, 50.6, 5), (29, 51.5, 6), (43, 51.1, 5)],
}

# Tables S24 and S25. Cross-base rows are (random mean, Local-DeltaAvg mean).
CROSS_BASE = {"Qwen2.5-7B": (45.8, 50.4), "Llama-3.1-8B": (39.2, 45.0), "Qwen2.5-14B": (53.0, 57.8)}
CROSS_SCORER = {
    "Qwen2.5-7B / Llama-3.1-8B": 0.67, "Qwen2.5-7B / Mistral-7B": 0.62,
    "Llama-3.1-8B / Gemma2-9B": 0.64, "Qwen2.5-7B / Qwen2.5-14B": 0.76,
}

# Table S27. Each row is (relative cost, after-SFT average, IoU, recall@3).
WINDOW_SCAN = {
    "8": (1.12, 45.1, 0.28, 0.54), "16": (1.08, 46.0, 0.36, 0.65),
    "32": (1.06, 46.8, 0.41, 0.69), "64": (1.05, 47.7, 0.43, 0.71),
    "128": (1.04, 46.9, 0.39, 0.68), "256": (1.02, 46.1, 0.31, 0.55),
    "Full": (1.00, 46.5, None, None),
}

# Table S28. Values are (100M, 300M, 500M, SD at 300M).
SMALL_BUDGETS = {
    "Random-STEM": (41.5, 42.8, 44.2, 0.4),
    "Forward PPL/loss": (42.0, 43.4, 44.7, 0.3),
    "AskLLM": (42.4, 43.8, 45.2, 0.4),
    "Local-DeltaAvg": (45.3, 46.7, 47.7, 0.2),
}

BLUE = "#245D97"
TEAL = "#0B766D"
AMBER = "#9A541D"
GREY = "#677C90"
LIGHT = "#A9B8C5"
INK = "#1F303F"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.3,
    "axes.labelsize": 8.3,
    "axes.titleweight": "bold",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.edgecolor": "#A2ACB4",
    "axes.linewidth": 0.65,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.color": "#DCE3E9",
    "grid.linewidth": 0.6,
    "legend.fontsize": 7.8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def panels(count, height=2.35, width=7.0, ratios=None):
    return plt.subplots(1, count, figsize=(width, height), layout="constrained", width_ratios=ratios)


def style(ax, title, xlabel="", ylabel="", grid="x"):
    ax.set_title(title, loc="left")
    ax.set(xlabel=xlabel, ylabel=ylabel, axisbelow=True)
    ax.grid(True, axis=grid)
    if grid == "x":
        ax.tick_params(axis="y", length=0)


def row_labels(ax, labels):
    rows = np.arange(len(labels))
    ax.set_yticks(rows, labels)
    ax.set_ylim(len(labels) - 0.45, -0.55)
    return rows


def hbars(ax, rows, values, colors, fmt, height=0.52):
    bars = ax.barh(rows, values, height=height, color=colors, zorder=3)
    for y, value in zip(rows, values):
        positive = value >= 0
        ax.annotate(
            fmt.format(value), (value, y), xytext=(3 if positive else -3, 0),
            textcoords="offset points", va="center", ha="left" if positive else "right",
            fontsize=7.6,
        )
    return bars


def dots(ax, rows, values, colors, fmt):
    ax.scatter(values, rows, s=27, color=colors, zorder=3)
    for y, x in zip(rows, values):
        ax.annotate(fmt.format(x), (x, y), (5, 0), textcoords="offset points", va="center", fontsize=7.8)


def legend_below(fig, handles, labels, cols=2):
    fig.legend(handles, labels, ncols=cols, loc="outside lower center", handlelength=1.6, columnspacing=1.15)


def full_budget():
    fig, (left, right) = panels(2, 2.52)
    style(left, "(a) Matched-budget comparison", "Five-task average (points)")
    methods = ["Random-STEM", "HES", "ASLEC-CASL", "Local-DeltaAvg", "Safe", "MV-Mean", "MV-Conservative", "Family"]
    points = [MAIN_RESULTS[name][-1] for name in methods]
    dots(left, row_labels(left, methods), points, [GREY] * 3 + [BLUE] * 4 + [TEAL], "{:.1f}")
    left.set(xlim=(45, 52.2), xticks=[45, 47, 49, 51])

    style(right, "(b) Gains by benchmark", "Points above ASLEC-CASL")
    rows = row_labels(right, BENCHMARKS)
    baseline = np.array(MAIN_RESULTS["ASLEC-CASL"][:-1])
    scalar = np.round(np.array(MAIN_RESULTS["Local-DeltaAvg"][:-1]) - baseline, 1)
    family = np.round(np.array(MAIN_RESULTS["Family"][:-1]) - baseline, 1)
    p1 = hbars(right, rows - 0.17, scalar, BLUE, "{:+.1f}", 0.27)
    p2 = hbars(right, rows + 0.17, family, TEAL, "{:+.1f}", 0.27)
    right.set(xlim=(0, 4.15), xticks=[0, 1, 2, 3, 4])
    legend_below(fig, [p1, p2], ["Local-DeltaAvg", "Local-DeltaAvg-Family"])
    return fig


def construct_controls():
    fig, (left, right) = panels(2, 1.99)
    style(left, "(a) Reasoning structure", "Mean directional score")
    rows = row_labels(left, STRUCTURE_SCORES)
    hbars(left, rows, STRUCTURE_SCORES.values(), [BLUE, GREY, LIGHT, LIGHT], "{:.2f}")
    left.set(xlim=(0, 0.65), xticks=[0, 0.2, 0.4, 0.6])

    style(right, "(b) Reversal and pilot utility", "Pilot accuracy gain vs. random (points)")
    variants = ["Semantic", "Sentence", "Token", "Shuffled"]
    gains = [round(PILOT_ACCURACY[name] - PILOT_ACCURACY["Random"], 1) for name in variants]
    hbars(right, row_labels(right, variants), gains, [BLUE, LIGHT, AMBER, LIGHT], "{:+.1f}")
    right.axvline(0, color=GREY, linewidth=0.8)
    right.set(xlim=(-1.5, 4.7), xticks=[-1, 0, 1, 2, 3, 4])
    return fig


def calibration_and_masking():
    fig, (left, right) = panels(2, 2.16)
    style(left, "(a) Answer masking ablation", "Five-task average (points)")
    display = ["Answer-first, raw", "Answer-first, centered", "Answer-loss-masked", "Answer-context-masked"]
    rows = row_labels(left, display)
    values = [MAIN_RESULTS[name][-1] for name in display[:3] + ["Local-DeltaAvg"]]
    left.plot(values, rows, linewidth=1.0, color=LIGHT)
    dots(left, rows, values, [GREY, GREY, GREY, BLUE], "{:.1f}")
    left.set(xlim=(49.4, 50.73), xticks=[49.5, 49.8, 50.1, 50.4, 50.7])

    style(right, "(b) Window-opportunity bias", "Spearman correlation")
    rows = row_labels(right, CALIBRATION)
    length, windows = np.asarray(list(CALIBRATION.values())).T
    p1 = hbars(right, rows - 0.17, length, GREY, "{:.2f}", 0.27)
    p2 = hbars(right, rows + 0.17, windows, TEAL, "{:.2f}", 0.27)
    right.axvline(0, color=GREY, linewidth=0.8)
    right.set(xlim=(-0.061, 0.225), xticks=[0, 0.05, 0.10, 0.15, 0.20])
    legend_below(fig, [p1, p2], ["Trace length", "Window count"])
    return fig


def family_coverage():
    fig, axes = panels(3, 1.94)
    scales = [
        ("Active mechanisms", "Count (out of 24)", 27, "{:d}/24"),
        ("Exposure Gini", "Gini coefficient", 0.5, "{:.2f}"),
        ("Worst-family accuracy", "Accuracy (%)", 54, "{:.1f}"),
    ]
    for panel, ax, (metric, ylabel, high, fmt) in zip("abc", axes, scales):
        style(ax, f"({panel}) {metric}", ylabel=ylabel, grid="y")
        values = COVERAGE[metric]
        bars = ax.bar(["Scalar", "Family"], values, color=[BLUE, TEAL], width=0.57, zorder=3)
        ax.set_ylim(0, high)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=2.5, fontsize=8)
    return fig


def acquisition_cost():
    fig, (left, right) = panels(2, 2.16, ratios=[0.85, 1.15])
    style(left, "(a) Value of mechanism grouping", "Gain over scalar ranking (points)")
    gains = [round(grouped - scalar, 1) for scalar, grouped in GROUPED_MEANS.values()]
    hbars(left, row_labels(left, GROUPED_MEANS), gains, [GREY, GREY, TEAL], "{:+.1f}")
    left.set(xlim=(0, 0.86), xticks=[0, 0.2, 0.4, 0.6, 0.8])

    style(right, "(b) Scoring cost and accuracy", "Scoring cost (H100 GPU-h, log scale)", "Five-task average", "y")
    settings = {
        "Forward PPL/loss": ("Forward PPL", GREY, (5, -4), "left"),
        "HES": ("HES", GREY, (5, 0), "left"),
        "Local-DeltaAvg": ("Local-DeltaAvg", BLUE, (5, -5), "left"),
        "Family": ("Local-DeltaAvg-Family", TEAL, (5, 1), "left"),
        "LLM-as-Judge": ("LLM-as-Judge", LIGHT, (-5, -6), "right"),
        "AskLLM": ("AskLLM", AMBER, (-5, 3), "right"),
    }
    for key, (label, color, offset, align) in settings.items():
        xy = SCORING_COST[key], MAIN_RESULTS[key][-1]
        right.scatter(*xy, color=color, s=29, zorder=3)
        right.annotate(label, xy, offset, textcoords="offset points", va="center", ha=align, fontsize=7.5)
    right.set(xscale="log", xlim=(310, 5700), ylim=(45.9, 52.1), yticks=[46, 48, 50, 52])
    right.set_xticks([400, 800, 1600, 3200, 4800], ["400", "800", "1,600", "3,200", "4,800"])
    right.xaxis.set_minor_locator(NullLocator())
    return fig


def view_stability():
    fig, (left, right) = panels(2, 2.43)
    style(left, "(a) Reversal-view agreement", "Agreement coefficient")
    rows = row_labels(left, VIEW_AGREEMENT)
    main, independent = np.asarray(list(VIEW_AGREEMENT.values())).T
    p1 = hbars(left, rows - 0.17, main, BLUE, "{:.2f}", 0.27)
    p2 = hbars(left, rows + 0.17, independent, GREY, "{:.2f}", 0.27)
    left.set(xlim=(0, 1.04), xticks=[0, 0.25, 0.50, 0.75, 1.0])
    legend_below(fig, [p1, p2], ["Main probe", "Independent probe"])

    style(right, "(b) Stability by score margin", "Score-margin quartile (smallest to largest)", "Traces (%)", "y")
    flip, retain = np.asarray(list(MARGIN.values())).T
    right.plot(list(MARGIN), retain, "D-", color=BLUE, label="Retained", markersize=4)
    right.plot(list(MARGIN), flip, "o--", color=AMBER, label="Rank flips", markersize=4)
    right.set(ylim=(0, 101), yticks=[0, 25, 50, 75, 100])
    right.legend(loc="center right")
    return fig


def exposure_response():
    fig, (left, right) = panels(2, 2.15)
    style(left, "(a) Response to family exposure", "Exposure units (64 examples per unit)", "Excess NLL", "y")
    mean, lower, upper = np.asarray(list(EXPOSURE.values())).T
    left.errorbar(
        list(EXPOSURE), mean, yerr=[mean - lower, upper - mean], color=BLUE, marker="o",
        markersize=4, linewidth=1.5, capsize=2.5,
    )
    left.set(xticks=list(EXPOSURE), ylim=(0.17, 0.88), yticks=[0.2, 0.4, 0.6, 0.8])

    style(right, "(b) Controlled selection", "Accuracy (%)")
    row_labels(right, CONTROLLED_SELECTION)
    for i, ((mean, sd), color) in enumerate(zip(CONTROLLED_SELECTION.values(), [GREY, GREY, TEAL, BLUE])):
        right.errorbar(mean, i, xerr=sd, fmt="o", color=color, markersize=4.7, capsize=2.5)
        right.annotate(f"{mean:.1f}", (mean + sd, i), (4, 0), textcoords="offset points", va="center", fontsize=7.8)
    right.set(xlim=(60.9, 76.2), xticks=[62, 66, 70, 74])
    return fig


def seed_stability():
    fig, (left, right) = panels(2, 2.33)
    style(left, "(a) Five-task scores by seed", "Training seed", "Five-task average", "y")
    style(right, "(b) AIME24 correct counts", "Training seed", "Correct out of 30", "y")
    positions = np.arange(3)
    handles = []
    for index, (name, color, marker) in enumerate(zip(SEED_RESULTS, [GREY, BLUE, TEAL], ["s", "o", "D"])):
        _, score, correct = np.asarray(SEED_RESULTS[name]).T
        (line,) = left.plot(positions, score, color=color, marker=marker, markersize=4.5, label=name)
        handles.append(line)
        bars = right.bar(positions + (index - 1) * 0.25, correct, width=0.22, color=color, zorder=3)
        right.bar_label(bars, fmt="%d", padding=2, fontsize=7.6)
    labels = [row[0] for row in SEED_RESULTS["Random-STEM"]]
    for ax in (left, right):
        ax.set_xticks(positions, labels)
    left.set(ylim=(44.5, 52.2), yticks=[45, 47, 49, 51])
    right.set(ylim=(0, 7.25), yticks=[0, 2, 4, 6])
    legend_below(fig, handles, ["Random-STEM", "Local-DeltaAvg", "Local-DeltaAvg-Family"], 3)
    return fig


def cross_model():
    fig, (left, right) = panels(2, 2.46)
    style(left, "(a) Cross-backbone selection", "Five-task average (points)")
    rows = row_labels(left, CROSS_BASE)
    random, directional = np.asarray(list(CROSS_BASE.values())).T
    left.hlines(rows, random, directional, color=LIGHT, linewidth=1.4)
    p1 = left.scatter(random, rows, color=GREY, s=25, zorder=3)
    p2 = left.scatter(directional, rows, color=BLUE, s=28, marker="D", zorder=3)
    for row, low, high in zip(rows, random, directional):
        left.annotate(f"+{high - low:.1f}", ((low + high) / 2, row), (0, 6), textcoords="offset points", ha="center", fontsize=7.7)
    left.set(xlim=(37, 60), xticks=[40, 45, 50, 55, 60])

    style(right, "(b) Cross-scorer rank agreement", "Spearman correlation")
    rows = row_labels(right, [key.replace(" / ", "\n/ ") for key in CROSS_SCORER])
    dots(right, rows, list(CROSS_SCORER.values()), BLUE, "{:.2f}")
    right.set(xlim=(0.51, 0.87), xticks=[0.55, 0.65, 0.75, 0.85])
    legend_below(fig, [p1, p2], ["Random-STEM", "Local-DeltaAvg"])
    return fig


def window_tradeoffs():
    fig, axes = panels(3, 2.07)
    cost, score, iou, recall = np.asarray(list(WINDOW_SCAN.values()), dtype=float).T
    x = np.arange(len(WINDOW_SCAN))
    titles = ["(a) Downstream utility", "(b) Step alignment", "(c) Scoring cost"]
    ylabels = ["After-SFT average", "Alignment score", "Cost / full-window cost"]
    for ax, heading, ylabel in zip(axes, titles, ylabels):
        style(ax, heading, "Window (tokens)", ylabel, "y")
        ax.axvspan(2.7, 3.3, color="#E3EFF8", linewidth=0)
        ax.set_xticks(x, WINDOW_SCAN, fontsize=7.2)
    left, middle, right = axes
    left.plot(x[:-1], score[:-1], "o-", color=BLUE, markersize=4)
    left.plot(x[-1], score[-1], "o", color=GREY, markerfacecolor="white", markersize=4)
    left.annotate("47.7", (3, score[3]), (0, 5), textcoords="offset points", ha="center", fontsize=7.8)
    left.set(ylim=(44.5, 48.45), yticks=[45, 46, 47, 48])
    middle.plot(x, iou, "o-", color=BLUE, label="IoU", markersize=4)
    middle.plot(x, recall, "D--", color=TEAL, label="Recall@3", markersize=3.8)
    middle.set_xticks(x[:-1], list(WINDOW_SCAN)[:-1], fontsize=7.2)
    middle.legend(loc="center right", handlelength=1.2, borderaxespad=0.1)
    middle.set(ylim=(0.22, 0.78), yticks=[0.25, 0.45, 0.65])
    right.plot(x[:-1], cost[:-1], "o-", color=BLUE, markersize=4)
    right.plot(x[-1], cost[-1], "o", color=GREY, markerfacecolor="white", markersize=4)
    right.set(ylim=(0.986, 1.137), yticks=[1.00, 1.04, 1.08, 1.12])
    return fig


def smaller_budgets():
    fig, ax = panels(1, 2.44, 3.34)
    style(ax, "Smaller selection budgets", "Selected tokens", "After-SFT average", "y")
    tokens = [100, 300, 500]
    handles = []
    for name, color, marker, dash in zip(
        SMALL_BUDGETS, [GREY, GREY, AMBER, BLUE], ["s", "^", "D", "o"], [":", "--", "-.", "-"],
    ):
        low, mid, high, sd = SMALL_BUDGETS[name]
        (line,) = ax.plot(tokens, [low, mid, high], linestyle=dash, marker=marker, color=color, markersize=4.1)
        ax.errorbar(300, mid, yerr=sd, color=color, linewidth=0.8, capsize=2.2)
        handles.append(line)
    ax.set(xlim=(73, 532), ylim=(40.85, 48.37), yticks=[42, 44, 46, 48])
    ax.set_xticks(tokens, ["100M", "300M", "500M"])
    legend_below(fig, handles, list(SMALL_BUDGETS))
    return fig


FIGURES = {
    "fig03_matched_budget_results": full_budget,
    "fig04_construct_validity": construct_controls,
    "fig05_masking_and_calibration": calibration_and_masking,
    "fig06_mechanism_coverage": family_coverage,
    "fig07_acquisition_and_cost": acquisition_cost,
    "figS01_reversal_stability": view_stability,
    "figS02_exposure_response": exposure_response,
    "figS03_seed_stability": seed_stability,
    "figS04_cross_model_validation": cross_model,
    "figS05_window_size": window_tradeoffs,
    "figS06_smaller_budgets": smaller_budgets,
}


if __name__ == "__main__":
    output = Path(__file__).resolve().parent
    for filename, draw in FIGURES.items():
        figure = draw()
        figure.savefig(output / f"{filename}.pdf")
        plt.close(figure)
