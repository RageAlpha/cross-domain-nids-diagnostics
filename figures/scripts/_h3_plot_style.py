#!/usr/bin/env python3
"""Render a restrained journal version of the audited H3 dissociation figure.

The design reduces visual encodings, states the C0/C1 semantics in the legend,
uses one marker for all training seeds, and presents domain discriminability on
its full theoretical scale with a ToN-IoT zoom inset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import _h3_data as source


C0_COLOR = "#767676"
C1_COLOR = "#0F4D92"
PAIR_COLOR = "#B8BDC6"
MEAN_COLOR = "#1F2937"
GRID_COLOR = "#D8DCE2"
TEXT_COLOR = "#303640"


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


def delta_text(metric: str, delta: float) -> str:
    digits = 5 if metric == "domain_discriminability" else 3
    return rf"$\Delta_{{\mathrm{{mean}}}}$={delta:+.{digits}f}"


def draw_pairs(ax, data, means, metric: str, target: str, center: float) -> tuple[float, float]:
    half_gap = 0.14
    jitter = {42: -0.026, 123: 0.0, 2024: 0.026}
    x0 = center - half_gap
    x1 = center + half_gap

    for seed in source.SEEDS:
        y0 = source.value_by_key(data, target, seed, "H3-C0", metric)
        y1 = source.value_by_key(data, target, seed, "H3-C1", metric)
        xs = [x0 + jitter[seed], x1 + jitter[seed]]
        ax.plot(xs, [y0, y1], color=PAIR_COLOR, linewidth=0.85, alpha=0.92, zorder=2)
        ax.scatter(
            xs[0], y0, s=22, marker="o", color=C0_COLOR,
            edgecolor="white", linewidth=0.45, zorder=3,
        )
        ax.scatter(
            xs[1], y1, s=22, marker="o", color=C1_COLOR,
            edgecolor="white", linewidth=0.45, zorder=3,
        )

    mean0 = source.mean_by_key(means, target, "H3-C0", metric)
    mean1 = source.mean_by_key(means, target, "H3-C1", metric)
    ax.plot([x0, x1], [mean0, mean1], color=MEAN_COLOR, linewidth=1.65, zorder=4)
    ax.scatter(
        x0, mean0, s=52, marker="D", color=C0_COLOR,
        edgecolor=MEAN_COLOR, linewidth=0.8, zorder=5,
    )
    ax.scatter(
        x1, mean1, s=52, marker="D", color=C1_COLOR,
        edgecolor=MEAN_COLOR, linewidth=0.8, zorder=5,
    )
    return mean0, mean1


def style_axis(ax, title: str, show_x_labels: bool) -> None:
    centers = np.arange(len(source.TARGETS), dtype=float)
    half_gap = 0.14
    ax.set_xlim(-0.46, len(source.TARGETS) - 1 + 0.46)
    ax.set_title(title, fontsize=8.8, fontweight="semibold", pad=5)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.80)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5F6570")
    ax.spines["bottom"].set_color("#5F6570")
    ax.tick_params(axis="y", labelsize=7.1, length=2.5, width=0.6)
    for boundary in (0.5, 1.5):
        ax.axvline(boundary, color=GRID_COLOR, linewidth=0.65, zorder=1)

    positions = []
    labels = []
    for center in centers:
        positions.extend([center - half_gap, center + half_gap])
        labels.extend(["C0", "C1"])
    ax.set_xticks(positions)
    if show_x_labels:
        ax.set_xticklabels(labels, fontsize=6.9)
        ax.tick_params(axis="x", length=0, pad=2)
        for center, target in zip(centers, source.TARGETS):
            ax.text(
                center,
                -0.20,
                target,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.3,
                fontweight="normal",
                clip_on=False,
            )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)


def plot_standard_metric(
    ax,
    data,
    means,
    metric: str,
    title: str,
    show_x_labels: bool,
) -> None:
    ton_delta = None
    for index, target in enumerate(source.TARGETS):
        mean0, mean1 = draw_pairs(ax, data, means, metric, target, float(index))
        if target == "ToN-IoT":
            ton_delta = mean1 - mean0

    lower, upper = source.metric_limits(data, metric)
    ton_values = data.loc[data["target"] == "ToN-IoT", metric].to_numpy(dtype=float)
    ton_top = float(ton_values.max())
    base_span = max(upper - lower, 1e-6)
    upper = max(upper, ton_top + 0.16 * base_span)
    ax.set_ylim(lower, upper)
    style_axis(ax, title, show_x_labels)

    if metric == "MCC":
        ax.axhline(0.0, color="#747A84", linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    if ton_delta is not None:
        annotation_y = ton_top + 0.045 * (upper - lower)
        ax.text(
            len(source.TARGETS) - 1,
            annotation_y,
            delta_text(metric, ton_delta),
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=TEXT_COLOR,
            zorder=6,
        )


def plot_domain_metric(ax, data, means, show_x_labels: bool) -> None:
    metric = "domain_discriminability"
    ton_mean0 = ton_mean1 = None
    for index, target in enumerate(source.TARGETS):
        mean0, mean1 = draw_pairs(ax, data, means, metric, target, float(index))
        if target == "ToN-IoT":
            ton_mean0, ton_mean1 = mean0, mean1

    # Full theoretical range: |AUROC - 0.5| is 0 at chance and 0.5 at maximum
    # separability.  The inset below is the only zoomed view.
    ax.set_ylim(0.0, 0.525)
    ax.set_yticks(np.arange(0.0, 0.51, 0.10))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.axhline(0.5, color="#747A84", linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    style_axis(
        ax,
        r"(d) $|\mathrm{AUROC}_{domain}-0.5|$ (lower is better)",
        show_x_labels,
    )
    ax.text(
        0.54,
        0.045,
        "0 = chance-level domain probe",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#5F6570",
    )

    inset = ax.inset_axes([0.075, 0.105, 0.43, 0.48])
    x0, x1 = 0.0, 1.0
    jitter = {42: -0.045, 123: 0.0, 2024: 0.045}
    for seed in source.SEEDS:
        y0 = source.value_by_key(data, "ToN-IoT", seed, "H3-C0", metric)
        y1 = source.value_by_key(data, "ToN-IoT", seed, "H3-C1", metric)
        xs = [x0 + jitter[seed], x1 + jitter[seed]]
        inset.plot(xs, [y0, y1], color=PAIR_COLOR, linewidth=0.75, zorder=2)
        inset.scatter(xs[0], y0, s=13, color=C0_COLOR, edgecolor="white", linewidth=0.35, zorder=3)
        inset.scatter(xs[1], y1, s=13, color=C1_COLOR, edgecolor="white", linewidth=0.35, zorder=3)
    inset.plot([x0, x1], [ton_mean0, ton_mean1], color=MEAN_COLOR, linewidth=1.35, zorder=4)
    inset.scatter(x0, ton_mean0, s=32, marker="D", color=C0_COLOR, edgecolor=MEAN_COLOR, linewidth=0.6, zorder=5)
    inset.scatter(x1, ton_mean1, s=32, marker="D", color=C1_COLOR, edgecolor=MEAN_COLOR, linewidth=0.6, zorder=5)
    inset.set_xlim(-0.25, 1.25)
    inset.set_ylim(0.4940, 0.5001)
    inset.set_xticks([0, 1], ["C0", "C1"])
    inset.set_yticks([0.495, 0.4975, 0.500])
    inset.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    inset.tick_params(axis="both", labelsize=5.8, length=1.8, width=0.45, pad=1.5)
    inset.grid(axis="y", color=GRID_COLOR, linewidth=0.45, alpha=0.75)
    inset.set_title(
        "ToN-IoT zoom  " + delta_text(metric, ton_mean1 - ton_mean0),
        fontsize=6.2,
        fontweight="normal",
        pad=2,
    )
    for spine in inset.spines.values():
        spine.set_color("#8A9099")
        spine.set_linewidth(0.55)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=C0_COLOR,
            markeredgecolor="white", markersize=5.2,
            label=r"C0: target-exposed $\lambda=0$ control",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=C1_COLOR,
            markeredgecolor="white", markersize=5.2,
            label="C1: frozen nonzero domain-loss schedule",
        ),
        Line2D(
            [0, 1], [0, 0], marker="o", color=PAIR_COLOR,
            markerfacecolor="#9CA1A9", markeredgecolor="white",
            linewidth=0.9, markersize=4.2, label="Individual seed pair",
        ),
        Line2D(
            [0, 1], [0, 0], marker="D", color=MEAN_COLOR,
            markerfacecolor="white", markeredgecolor=MEAN_COLOR,
            linewidth=1.5, markersize=5.1, label="3-seed mean",
        ),
    ]


def create_figure(data, means) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.45, 5.35),
        sharex="col",
        gridspec_kw={"wspace": 0.24, "hspace": 0.30},
    )
    plot_standard_metric(axes[0, 0], data, means, "MCC", r"(a) MCC ($\uparrow$)", False)
    plot_standard_metric(
        axes[1, 0], data, means, "Binary_F1", r"(b) Binary-F1 ($\uparrow$)", True,
    )
    plot_standard_metric(axes[0, 1], data, means, "MMD", r"(c) MMD ($\downarrow$)", False)
    plot_domain_metric(axes[1, 1], data, means, True)

    figure.text(
        0.27, 0.969, "Detection utility",
        ha="center", va="top", fontsize=9.5, fontweight="semibold",
    )
    figure.text(
        0.75, 0.969, "Representation discrepancy",
        ha="center", va="top", fontsize=9.5, fontweight="semibold",
    )
    figure.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handletextpad=0.50,
        columnspacing=1.35,
        labelspacing=0.65,
    )
    figure.subplots_adjust(left=0.074, right=0.985, top=0.895, bottom=0.245)
    return figure


def main() -> None:
    args = parse_args()
    csv_path = args.input.resolve()
    audit_path = args.audit.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, means = source.load_and_validate(csv_path, audit_path)
    expected = np.abs(data["domain_AUROC_mean"].to_numpy(dtype=float) - 0.5)
    observed = data["domain_discriminability"].to_numpy(dtype=float)
    source.require(
        np.allclose(expected, observed, atol=1e-12, rtol=0.0),
        "Domain discriminability is not |mean domain AUROC - 0.5|",
    )
    figure = create_figure(data, means)

    pdf_path = output_dir / "figure3_h3_dissociation_journal_clean.pdf"
    png_path = output_dir / "figure3_h3_dissociation_journal_clean.png"
    metadata = {
        "Title": "Matched H3 performance-representation dissociation",
        "Author": "Anonymous",
        "Subject": "Paired zero-loss control and nonzero domain-loss conditions",
        "Creator": Path(__file__).name,
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(
        pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.035,
        metadata=metadata,
    )
    figure.savefig(
        png_path, format="png", dpi=args.png_dpi,
        bbox_inches="tight", pad_inches=0.035,
        metadata={"Software": Path(__file__).name},
    )
    plt.close(figure)

    print(f"Validated data: {len(data)} rows (3 targets x 3 training seeds x 2 variants)")
    print("C0: target-exposed matched control with zero domain-loss weight")
    print("C1: matched condition with the frozen nonzero domain-loss schedule")
    print("Verified domain discriminability = |mean domain AUROC - 0.5|")
    print(f"Input SHA-256: {source.sha256(csv_path)}")
    for metric in source.METRICS:
        c0 = source.mean_by_key(means, "ToN-IoT", "H3-C0", metric)
        c1 = source.mean_by_key(means, "ToN-IoT", "H3-C1", metric)
        print(f"ToN-IoT {metric}: C0={c0:.8f}, C1={c1:.8f}, delta={c1-c0:+.8f}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
