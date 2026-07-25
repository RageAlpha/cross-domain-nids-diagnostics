#!/usr/bin/env python3
"""Render a compact journal version of the audited H3 dissociation figure.

Panel (d) uses chance proximity, 0.5 - |domain AUROC - 0.5|, because the
requested 0--0.006 display range is not the range of distance from chance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import _h3_plot_style as journal


source = journal.source
C0_COLOR = journal.C0_COLOR
C1_COLOR = journal.C1_COLOR
PAIR_COLOR = "#AEB4BD"
MEAN_COLOR = "#4D4D4D"
TEXT_COLOR = journal.TEXT_COLOR


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
    digits = 5 if metric == "chance_proximity" else 3
    sign = "−" if delta < 0 else "+"
    return rf"$\Delta_{{\mathrm{{mean}}}}$ = {sign}{abs(delta):.{digits}f}"


def draw_pairs(ax, data, means, metric: str, target: str, center: float) -> tuple[float, float]:
    half_gap = 0.14
    jitter = {42: -0.026, 123: 0.0, 2024: 0.026}
    x0 = center - half_gap
    x1 = center + half_gap

    for seed in source.SEEDS:
        y0 = source.value_by_key(data, target, seed, "H3-C0", metric)
        y1 = source.value_by_key(data, target, seed, "H3-C1", metric)
        xs = [x0 + jitter[seed], x1 + jitter[seed]]
        ax.plot(xs, [y0, y1], color=PAIR_COLOR, linewidth=0.95, alpha=0.95, zorder=2)
        ax.scatter(
            xs[0], y0, s=20, marker="o", color=C0_COLOR,
            edgecolor="white", linewidth=0.4, zorder=3,
        )
        ax.scatter(
            xs[1], y1, s=20, marker="o", color=C1_COLOR,
            edgecolor="white", linewidth=0.4, zorder=3,
        )

    mean0 = source.mean_by_key(means, target, "H3-C0", metric)
    mean1 = source.mean_by_key(means, target, "H3-C1", metric)
    ax.plot([x0, x1], [mean0, mean1], color=MEAN_COLOR, linewidth=1.35, zorder=4)
    ax.scatter(
        x0, mean0, s=42, marker="D", color=C0_COLOR,
        edgecolor=MEAN_COLOR, linewidth=0.65, zorder=5,
    )
    ax.scatter(
        x1, mean1, s=42, marker="D", color=C1_COLOR,
        edgecolor=MEAN_COLOR, linewidth=0.65, zorder=5,
    )
    return mean0, mean1


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
    upper = max(upper, ton_top + 0.15 * base_span)
    ax.set_ylim(lower, upper)
    journal.style_axis(ax, title, show_x_labels)

    if metric == "MCC":
        ax.axhline(0.0, color="#747A84", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    annotation_y = ton_top + 0.042 * (upper - lower)
    ax.text(
        len(source.TARGETS) - 1,
        annotation_y,
        delta_text(metric, ton_delta),
        ha="center",
        va="bottom",
        fontsize=6.9,
        color=TEXT_COLOR,
        zorder=6,
    )


def plot_chance_proximity(ax, data, means, show_x_labels: bool) -> None:
    metric = "chance_proximity"
    ton_delta = None
    for index, target in enumerate(source.TARGETS):
        mean0, mean1 = draw_pairs(ax, data, means, metric, target, float(index))
        if target == "ToN-IoT":
            ton_delta = mean1 - mean0

    maximum = float(data[metric].max())
    ton_top = float(data.loc[data["target"] == "ToN-IoT", metric].max())
    upper = max(0.0060, maximum * 1.20)
    ax.set_ylim(0.0, upper)
    ax.set_yticks([0.000, 0.002, 0.004, 0.006])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    journal.style_axis(
        ax,
        r"(d) Domain chance proximity ($\uparrow$)",
        show_x_labels,
    )
    annotation_y = ton_top + 0.055 * upper
    ax.text(
        len(source.TARGETS) - 1,
        annotation_y,
        delta_text(metric, ton_delta),
        ha="center",
        va="bottom",
        fontsize=6.9,
        color=TEXT_COLOR,
        zorder=6,
    )


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=C0_COLOR,
            markeredgecolor="white", markersize=5.0, label="C0",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=C1_COLOR,
            markeredgecolor="white", markersize=5.0, label="C1",
        ),
        Line2D(
            [0, 1], [0, 0], marker="o", color=PAIR_COLOR,
            markerfacecolor="#9CA1A9", markeredgecolor="white",
            linewidth=0.95, markersize=4.0, label="Individual seed pair",
        ),
        Line2D(
            [0, 1], [0, 0], marker="D", color=MEAN_COLOR,
            markerfacecolor="white", markeredgecolor=MEAN_COLOR,
            linewidth=1.35, markersize=4.7, label="3-seed mean",
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
        figsize=(7.45, 5.0),
        sharex="col",
        gridspec_kw={"wspace": 0.24, "hspace": 0.30},
    )
    plot_standard_metric(axes[0, 0], data, means, "MCC", r"(a) MCC ($\uparrow$)", False)
    plot_standard_metric(
        axes[1, 0], data, means, "Binary_F1", r"(b) Binary-F1 ($\uparrow$)", True,
    )
    plot_standard_metric(axes[0, 1], data, means, "MMD", r"(c) MMD ($\downarrow$)", False)
    plot_chance_proximity(axes[1, 1], data, means, True)

    figure.text(
        0.27, 0.969, "Detection utility",
        ha="center", va="top", fontsize=9.1, fontweight="semibold",
    )
    figure.text(
        0.75, 0.969, "Representation discrepancy",
        ha="center", va="top", fontsize=9.1, fontweight="semibold",
    )
    figure.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        frameon=False,
        fontsize=7.0,
        handletextpad=0.45,
        columnspacing=1.25,
    )
    figure.subplots_adjust(left=0.074, right=0.985, top=0.892, bottom=0.215)
    return figure


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    src_root = repository_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from cross_domain_nids.release_paths import require_generated_output

    csv_path = args.input.resolve()
    audit_path = args.audit.resolve()
    output_dir = require_generated_output(
        args.output_dir, "figures", repository_root=repository_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data, means = source.load_and_validate(csv_path, audit_path)
    data = data.copy()
    means = means.copy()
    expected_separability = np.abs(data["domain_AUROC_mean"].to_numpy(dtype=float) - 0.5)
    source.require(
        np.allclose(
            expected_separability,
            data["domain_discriminability"].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        ),
        "Domain discriminability is not |mean domain AUROC - 0.5|",
    )
    data["chance_proximity"] = 0.5 - data["domain_discriminability"]
    means["chance_proximity"] = 0.5 - means["domain_discriminability"]
    source.require(
        data["chance_proximity"].between(0.0, 0.5).all(),
        "Chance-proximity transform lies outside [0, 0.5]",
    )

    figure = create_figure(data, means)
    pdf_path = output_dir / "h3_paired_domain_loss.pdf"
    png_path = output_dir / "h3_paired_domain_loss.png"
    metadata = {
        "Title": "Matched H3 performance-representation dissociation",
        "Author": "Anonymous",
        "Subject": "Compact paired comparison with domain chance proximity",
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
    print(
        "Distance from chance range: "
        f"{data['domain_discriminability'].min():.8f} to "
        f"{data['domain_discriminability'].max():.8f}"
    )
    print(
        "Chance proximity range: "
        f"{data['chance_proximity'].min():.8f} to "
        f"{data['chance_proximity'].max():.8f}"
    )
    for metric in ("MCC", "Binary_F1", "MMD", "chance_proximity"):
        c0 = source.mean_by_key(means, "ToN-IoT", "H3-C0", metric)
        c1 = source.mean_by_key(means, "ToN-IoT", "H3-C1", metric)
        print(f"ToN-IoT {metric}: C0={c0:.8f}, C1={c1:.8f}, delta={c1-c0:+.8f}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
