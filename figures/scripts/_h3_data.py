#!/usr/bin/env python3
"""Plot the matched H3 performance-representation dissociation.

Inputs are the compact, audited target/seed/variant table and its provenance
report.  All plotted values and annotations are computed from the CSV; no
experimental result is embedded in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


TARGETS = ("BoT-IoT", "CICIDS2018", "ToN-IoT")
SEEDS = (42, 123, 2024)
VARIANTS = ("H3-C0", "H3-C1")
REQUIRED_COLUMNS = {
    "target",
    "seed",
    "variant",
    "MCC",
    "Binary_F1",
    "MMD",
    "domain_discriminability",
    "domain_AUROC_mean",
    "domain_probe_replicates",
}
METRICS = ("MCC", "Binary_F1", "MMD", "domain_discriminability")

C0_COLOR = "#6B7280"
C1_COLOR = "#0072B2"
PAIR_COLOR = "#8A8F98"
MEAN_COLOR = "#111827"
GRID_COLOR = "#D8DCE2"
TON_HIGHLIGHT = "#FFF1BF"
SEED_MARKERS = {42: "o", 123: "^", 2024: "s"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir.parent / "data" / "h3_domain_loss_figure_data.csv",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=script_dir.parent / "data" / "h3_figure_validation.md",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir.parent / "generated")
    parser.add_argument("--png-dpi", type=int, default=600)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_and_validate(csv_path: Path, audit_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    require(csv_path.is_file(), f"Missing audited figure data: {csv_path}")
    require(audit_path.is_file(), f"Missing data audit report: {audit_path}")

    audit_text = audit_path.read_text(encoding="utf-8")
    require("## Figure readiness" in audit_text, "Audit report has no figure-readiness section")
    require("**Pass.**" in audit_text, "Audit report does not mark figure readiness as Pass")
    require(
        "mean over probe splits" in audit_text
        and "not treated as independent training seeds" in audit_text,
        "Audit report does not document within-training-seed domain-probe aggregation",
    )

    data = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    require(not missing, f"Missing required CSV columns: {sorted(missing)}")
    require(len(data) == len(TARGETS) * len(SEEDS) * len(VARIANTS), "Expected exactly 18 figure rows")
    require(set(data["target"]) == set(TARGETS), "Target coverage mismatch")
    require(set(data["seed"].astype(int)) == set(SEEDS), "Training-seed coverage mismatch")
    require(set(data["variant"]) == set(VARIANTS), "Variant coverage mismatch")
    require(
        not data.duplicated(["target", "seed", "variant"]).any(),
        "Duplicate target/seed/variant rows would inflate the plotted sample size",
    )

    expected_pairs = {(target, seed, variant) for target in TARGETS for seed in SEEDS for variant in VARIANTS}
    actual_pairs = set(zip(data["target"], data["seed"].astype(int), data["variant"]))
    require(actual_pairs == expected_pairs, "Incomplete 3-target x 3-seed x 2-variant Cartesian coverage")
    require(
        set(data["domain_probe_replicates"].astype(int)) == {3},
        "Domain discriminability must already average exactly three probe splits per training-seed cell",
    )
    require("probe_seed" not in data.columns, "Probe splits must not appear as independent plotted observations")

    for metric in METRICS + ("domain_AUROC_mean",):
        numeric = pd.to_numeric(data[metric], errors="coerce")
        require(np.isfinite(numeric).all(), f"Non-finite values in {metric}")
        data[metric] = numeric
    require(data["domain_discriminability"].between(0.0, 0.5).all(), "Invalid domain-discriminability range")
    require(data["domain_AUROC_mean"].between(0.0, 1.0).all(), "Invalid domain-AUROC range")

    data["target"] = pd.Categorical(data["target"], categories=TARGETS, ordered=True)
    data["variant"] = pd.Categorical(data["variant"], categories=VARIANTS, ordered=True)
    data = data.sort_values(["target", "seed", "variant"]).reset_index(drop=True)

    means = (
        data.groupby(["target", "variant"], observed=True)[list(METRICS)]
        .mean()
        .reset_index()
    )
    require(len(means) == len(TARGETS) * len(VARIANTS), "Mean recomputation coverage mismatch")

    pivot = data.pivot(index=["target", "seed"], columns="variant", values=list(METRICS))
    ton = pivot.loc["ToN-IoT"]
    for metric in ("MCC", "Binary_F1", "MMD"):
        deltas = ton[(metric, "H3-C1")] - ton[(metric, "H3-C0")]
        require((deltas < 0).all(), f"ToN-IoT {metric} is not lower under C1 in all three training seeds")
    domain_deltas = ton[("domain_discriminability", "H3-C1")] - ton[("domain_discriminability", "H3-C0")]
    require(
        (domain_deltas < 0).all(),
        "ToN-IoT domain discriminability is not lower under C1 in all three training seeds",
    )

    return data, means


def value_by_key(data: pd.DataFrame, target: str, seed: int, variant: str, metric: str) -> float:
    rows = data[
        (data["target"] == target)
        & (data["seed"].astype(int) == seed)
        & (data["variant"] == variant)
    ]
    require(len(rows) == 1, f"Expected one value for {target}/{seed}/{variant}/{metric}")
    return float(rows.iloc[0][metric])


def mean_by_key(means: pd.DataFrame, target: str, variant: str, metric: str) -> float:
    rows = means[(means["target"] == target) & (means["variant"] == variant)]
    require(len(rows) == 1, f"Expected one mean for {target}/{variant}/{metric}")
    return float(rows.iloc[0][metric])


def metric_limits(data: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = data[metric].to_numpy(dtype=float)
    minimum = float(values.min())
    maximum = float(values.max())
    span = max(maximum - minimum, 1e-6)
    if metric == "Binary_F1":
        return 0.0, min(1.0, maximum + 0.10 * span + 0.02)
    if metric == "MMD":
        return 0.0, maximum + 0.12 * span
    if metric == "MCC":
        return min(-0.05, minimum - 0.10 * span), max(0.05, maximum + 0.10 * span)
    # This proxy lies very close to its theoretical maximum of 0.5.  A visibly
    # labeled zoom is necessary to show the paired directions without pretending
    # that the small changes are comparable in magnitude to the MMD changes.
    # Keep the 0.5 theoretical limit visible as a reference line while adding
    # a tiny amount of headroom so markers near the limit are not clipped.
    return max(0.0, minimum - 0.12 * span), 0.5002


def delta_label(metric: str, delta: float) -> str:
    digits = 5 if metric == "domain_discriminability" else 3
    return rf"ToN mean $\Delta$={delta:+.{digits}f}"


def plot_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    means: pd.DataFrame,
    metric: str,
    title: str,
    show_x_labels: bool,
) -> None:
    centers = np.arange(len(TARGETS), dtype=float)
    half_gap = 0.14
    jitter = {42: -0.026, 123: 0.0, 2024: 0.026}

    for index, target in enumerate(TARGETS):
        center = centers[index]
        x0 = center - half_gap
        x1 = center + half_gap
        if target == "ToN-IoT":
            ax.axvspan(center - 0.40, center + 0.40, color=TON_HIGHLIGHT, alpha=0.58, zorder=0)

        for seed in SEEDS:
            y0 = value_by_key(data, target, seed, "H3-C0", metric)
            y1 = value_by_key(data, target, seed, "H3-C1", metric)
            xs = [x0 + jitter[seed], x1 + jitter[seed]]
            ax.plot(xs, [y0, y1], color=PAIR_COLOR, linewidth=0.85, alpha=0.80, zorder=2)
            ax.scatter(
                xs[0], y0, s=23, marker=SEED_MARKERS[seed], color=C0_COLOR,
                edgecolor="white", linewidth=0.45, zorder=3,
            )
            ax.scatter(
                xs[1], y1, s=23, marker=SEED_MARKERS[seed], color=C1_COLOR,
                edgecolor="white", linewidth=0.45, zorder=3,
            )

        mean0 = mean_by_key(means, target, "H3-C0", metric)
        mean1 = mean_by_key(means, target, "H3-C1", metric)
        ax.plot([x0, x1], [mean0, mean1], color=MEAN_COLOR, linewidth=1.65, zorder=4)
        ax.scatter(x0, mean0, s=49, marker="D", color=C0_COLOR, edgecolor=MEAN_COLOR, linewidth=0.75, zorder=5)
        ax.scatter(x1, mean1, s=49, marker="D", color=C1_COLOR, edgecolor=MEAN_COLOR, linewidth=0.75, zorder=5)

        if target == "ToN-IoT":
            ax.text(
                center,
                0.86,
                delta_label(metric, mean1 - mean0),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.9,
                color=MEAN_COLOR,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#C8B676", "linewidth": 0.55, "alpha": 0.93},
                zorder=6,
            )

    for boundary in (0.5, 1.5):
        ax.axvline(boundary, color=GRID_COLOR, linewidth=0.65, zorder=1)

    lower, upper = metric_limits(data, metric)
    ax.set_ylim(lower, upper)
    ax.set_xlim(-0.46, len(TARGETS) - 1 + 0.46)
    ax.set_title(title, fontsize=8.8, fontweight="semibold", pad=5)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5F6570")
    ax.spines["bottom"].set_color("#5F6570")
    ax.tick_params(axis="y", labelsize=7.1, length=2.5, width=0.6)

    if metric == "MCC":
        ax.axhline(0.0, color="#747A84", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    if metric == "domain_discriminability":
        ax.axhline(0.5, color="#747A84", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    pair_positions: list[float] = []
    pair_labels: list[str] = []
    for center in centers:
        pair_positions.extend([center - half_gap, center + half_gap])
        pair_labels.extend(["C0", "C1"])
    ax.set_xticks(pair_positions)
    if show_x_labels:
        ax.set_xticklabels(pair_labels, fontsize=6.8)
        ax.tick_params(axis="x", length=0, pad=2)
        for center, target in zip(centers, TARGETS):
            ax.text(
                center,
                -0.20,
                target,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.2,
                fontweight="semibold" if target == "ToN-IoT" else "normal",
                clip_on=False,
            )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)


def create_figure(data: pd.DataFrame, means: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.25, 5.15),
        sharex="col",
        gridspec_kw={"wspace": 0.24, "hspace": 0.30},
    )
    plot_metric(axes[0, 0], data, means, "MCC", "MCC (higher is better)", False)
    plot_metric(axes[1, 0], data, means, "Binary_F1", "Binary-F1 (higher is better)", True)
    plot_metric(axes[0, 1], data, means, "MMD", "MMD (lower is better)", False)
    plot_metric(
        axes[1, 1],
        data,
        means,
        "domain_discriminability",
        "Domain discriminability (lower is better; zoomed)",
        True,
    )

    figure.text(0.27, 0.962, "(a) Target detection performance", ha="center", va="top", fontsize=10.2, fontweight="semibold")
    figure.text(0.75, 0.962, "(b) Representation discrepancy", ha="center", va="top", fontsize=10.2, fontweight="semibold")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C0_COLOR, markeredgecolor="white", markersize=5.2, label="C0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C1_COLOR, markeredgecolor="white", markersize=5.2, label="C1"),
        Line2D([0], [0], marker="D", color=MEAN_COLOR, markerfacecolor="white", markeredgecolor=MEAN_COLOR, linewidth=1.5, markersize=5.2, label="3-seed mean"),
        Line2D([0], [0], marker=SEED_MARKERS[42], color="none", markerfacecolor="#A1A6AE", markersize=4.8, label="seed 42"),
        Line2D([0], [0], marker=SEED_MARKERS[123], color="none", markerfacecolor="#A1A6AE", markersize=4.8, label="seed 123"),
        Line2D([0], [0], marker=SEED_MARKERS[2024], color="none", markerfacecolor="#A1A6AE", markersize=4.8, label="seed 2024"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.014),
        ncol=6,
        frameon=False,
        fontsize=7.1,
        handletextpad=0.45,
        columnspacing=1.15,
    )
    figure.text(
        0.5,
        0.078,
        "Paired lines connect the same training seed; diamonds show means. The shaded ToN-IoT group exposes the proxy-utility dissociation.",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#3F4650",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.895, bottom=0.235)
    return figure


def main() -> None:
    args = parse_args()
    csv_path = args.input.resolve()
    audit_path = args.audit.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, means = load_and_validate(csv_path, audit_path)
    figure = create_figure(data, means)

    pdf_path = output_dir / "figure3_h3_dissociation.pdf"
    png_path = output_dir / "figure3_h3_dissociation.png"
    metadata = {
        "Title": "Matched H3 performance-representation dissociation",
        "Author": "Anonymous",
        "Subject": "Paired C0-C1 H3 evidence across targets and training seeds",
        "Creator": "plot_h3_dissociation.py",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.035, metadata=metadata)
    figure.savefig(png_path, format="png", dpi=args.png_dpi, bbox_inches="tight", pad_inches=0.035, metadata={"Software": "plot_h3_dissociation.py"})
    plt.close(figure)

    print(f"Validated data: {len(data)} rows (3 targets x 3 training seeds x 2 variants)")
    print("Domain-probe aggregation: 3 probe splits per training-seed/variant cell")
    print(f"Input SHA-256: {sha256(csv_path)}")
    print(f"Audit SHA-256: {sha256(audit_path)}")
    for metric in METRICS:
        c0 = mean_by_key(means, "ToN-IoT", "H3-C0", metric)
        c1 = mean_by_key(means, "ToN-IoT", "H3-C1", metric)
        print(f"ToN-IoT {metric}: C0={c0:.8f}, C1={c1:.8f}, delta={c1-c0:+.8f}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
