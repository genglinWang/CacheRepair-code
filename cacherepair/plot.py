"""Summarize newly generated runs and draw one four-dataset figure per target LLM."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATASETS = {
    "musique": "MuSiQue",
    "hotpotqa": "HotpotQA",
    "multihoprag": "MultiHop-RAG",
    "triviaqa": "TriviaQA",
}
TARGETS = {
    "qwen2.5_3b_instruct": ("qwen3", "Qwen2.5-3B", "9/30/51M"),
    "llama3.1_8b_instruct": ("llama8", "Llama-3.1-8B", "23/59/93M"),
    "qwen2.5_14b_instruct": ("qwen14", "Qwen2.5-14B", "42/104/159M"),
}
STYLE = {
    "full": ("#111111", "*", "Full Prefill"),
    "stale": ("#7A7A7A", "x", "Stale KV"),
    "repair": ("#D55E00", "o", "CacheRepair"),
}


def collect(root, bootstrap):
    points, identities, configurations = [], {}, set()
    rng = np.random.default_rng(7)
    for config_path in sorted(root.rglob("run_config.json")):
        config = json.loads(config_path.read_text())
        rows = [
            json.loads(line)
            for line in config_path.with_name("rows.jsonl").read_text().splitlines()
        ]
        if len(rows) != config["requests"]:
            raise ValueError(f"Run is incomplete: {config_path.parent}")
        cell = config["target_model"], config["dataset"]
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)) or identities.setdefault(cell, ids) != ids:
            raise ValueError(f"Methods must use the same ordered requests: {cell}")
        key = (*cell, config["method"], config["size"])
        if key in configurations:
            raise ValueError(f"Duplicate configuration: {key}; select one results directory")
        configurations.add(key)
        values = np.asarray([row["f1"] for row in rows])
        point = dict(
            target=cell[0],
            dataset=cell[1],
            method=config["method"],
            size=config["size"] or "",
            requests=len(rows),
            f1=float(values.mean()),
            em=statistics.mean(row["em"] for row in rows),
            ttft_ms=1000 * statistics.median(row["ttft_s"] for row in rows),
        )
        if bootstrap:
            means = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
            point["f1_low"], point["f1_high"] = np.quantile(means, [0.025, 0.975])
        points.append(point)
    if not points:
        raise ValueError("No completed evaluation runs found")
    return points


def draw(points, target, output, intervals):
    short, title, sizes = TARGETS[target]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=(9, 2.6))
    expected = {("full", ""), ("stale", ""), ("repair", "S"), ("repair", "M"), ("repair", "L")}
    for letter, axis, (dataset, label) in zip("ABCD", axes, DATASETS.items()):
        panel = [p for p in points if p["target"] == target and p["dataset"] == dataset]
        if {(p["method"], p["size"]) for p in panel} != expected:
            raise ValueError(f"{title}/{label} needs Full Prefill, Stale KV and CacheRepair S/M/L")
        for method, (color, marker, _) in STYLE.items():
            series = sorted(
                [p for p in panel if p["method"] == method], key=lambda p: "SML".find(p["size"])
            )
            x, y = [p["ttft_ms"] for p in series], [p["f1"] for p in series]
            axis.plot(
                x,
                y,
                color=color,
                marker=marker,
                markersize=8 if method == "full" else 4.5,
                linestyle="-" if method == "repair" else "none",
                linewidth=1.1,
                zorder=3,
            )
            if intervals:
                axis.vlines(
                    x,
                    [p["f1_low"] for p in series],
                    [p["f1_high"] for p in series],
                    color=color,
                    alpha=0.45,
                    linewidth=0.8,
                )
        full = next(p for p in panel if p["method"] == "full")
        axis.axhline(full["f1"], color="#777777", linestyle=(0, (4, 3)), linewidth=0.8)
        axis.set(title=f"({letter}) {label}", xlabel="TTFT (ms, p50)", ylabel="F1")
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
        axis.margins(x=0.12, y=0.15)
    handles = [
        Line2D(
            [],
            [],
            color=c,
            marker=m,
            linestyle="-" if k == "repair" else "none",
            label=f"{label} ({sizes})" if k == "repair" else label,
        )
        for k, (c, m, label) in STYLE.items()
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(title, y=1.0, fontsize=11)
    figure.tight_layout(rect=(0, 0.13, 1, 0.95))
    for extension in ("pdf", "png"):
        figure.savefig(output / f"{short}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--intervals", action="store_true", help="Show 95% request-bootstrap F1 intervals"
    )
    args = parser.parse_args()
    points = collect(args.results, args.intervals)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    for target in TARGETS:
        if any(point["target"] == target for point in points):
            draw(points, target, args.output, args.intervals)
    print(f"Saved metrics.csv and figures to {args.output}")


if __name__ == "__main__":
    main()
