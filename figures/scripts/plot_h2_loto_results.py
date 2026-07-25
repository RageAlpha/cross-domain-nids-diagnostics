"""Generate the publication-ready Figure 2 feature-family JSD heatmap.

This script follows the local scientific-figure-making skill conventions:
headless Matplotlib, explicit publication style, validated matrix inputs, and
stable vector/high-DPI outputs. It does not modify the manuscript LaTeX.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


TARGET_ORDER = ["BoT-IoT", "CICIDS2018", "ToN-IoT"]
FEATURE_FAMILY_ORDER = [
    "Byte volume",
    "Duration / timing",
    "ICMP",
    "Other",
    "Packet count",
    "Port",
    "Protocol / L7",
    "TCP flag",
]
DISPLAY_FAMILY_LABELS = {
    "Duration / timing": "Duration",
    "Packet count": "Packets",
    "Protocol / L7": "Protocol/L7",
}
VMIN = 0.0
VMAX = 0.8


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 1.8
    use_tex: bool = False
    font_family: tuple[str, ...] = (
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "sans-serif",
    )


def apply_publication_style(style: FigureStyle | None = None) -> None:
    """Apply the project publication style before creating any axes."""
    style = style or FigureStyle()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(style.font_family),
            "font.size": style.font_size,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.unicode_minus": False,
            "legend.frameon": False,
            "text.usetex": style.use_tex,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_heatmap_data(csv_path: Path) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame]:
    """Read and validate the 8-by-3 feature-family mean-JSD matrix."""
    data = pd.read_csv(csv_path)
    required = {"target_dataset", "feature_family", "num_features", "mean_jsd"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data.duplicated(["target_dataset", "feature_family"]).any():
        raise ValueError("Duplicate target/feature-family rows found.")
    if not data["mean_jsd"].between(0.0, 1.0).all():
        raise ValueError("All mean_jsd values must lie in [0, 1].")

    actual_targets = set(data["target_dataset"])
    actual_families = set(data["feature_family"])
    if actual_targets != set(TARGET_ORDER):
        raise ValueError(f"Unexpected targets: {sorted(actual_targets)}")
    if actual_families != set(FEATURE_FAMILY_ORDER):
        raise ValueError(f"Unexpected feature families: {sorted(actual_families)}")

    count_variation = data.groupby("feature_family")["num_features"].nunique()
    if not count_variation.eq(1).all():
        raise ValueError("Feature counts are inconsistent across target domains.")
    family_counts = (
        data.groupby("feature_family", sort=False)["num_features"]
        .first()
        .reindex(FEATURE_FAMILY_ORDER)
        .astype(int)
    )

    pivot = data.pivot(
        index="feature_family",
        columns="target_dataset",
        values="mean_jsd",
    ).reindex(index=FEATURE_FAMILY_ORDER, columns=TARGET_ORDER)
    if pivot.isna().any().any() or pivot.shape != (8, 3):
        raise ValueError(f"Expected a complete 8x3 matrix, got {pivot.shape}.")

    y_labels = [
        f"{DISPLAY_FAMILY_LABELS.get(family, family)} (n={family_counts[family]})"
        for family in FEATURE_FAMILY_ORDER
    ]
    return pivot.to_numpy(dtype=float), TARGET_ORDER, y_labels, data


def make_heatmap(
    ax: Axes,
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    *,
    cmap: str = "magma",
    cbar_label: str = "Mean JSD",
    annotate: bool = True,
) -> None:
    """Render an annotated heatmap with a shared, explicit JSD scale."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Heatmap matrix must be two-dimensional.")
    if values.shape != (len(y_labels), len(x_labels)):
        raise ValueError("Heatmap dimensions do not match the provided labels.")

    norm = Normalize(vmin=VMIN, vmax=VMAX, clip=True)
    colormap = matplotlib.colormaps.get_cmap(cmap)
    image = ax.imshow(
        values,
        cmap=colormap,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.tick_params(axis="x", which="major", length=0, pad=6, labelsize=16.5)
    ax.tick_params(axis="y", which="major", length=0, pad=6, labelsize=16.5)

    # Thin cell boundaries improve print readability without adding a heavy grid.
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                red, green, blue, _ = colormap(norm(value))
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                text_color = "white" if luminance < 0.5 else "#161616"
                ax.text(
                    col,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=17,
                    fontweight="semibold",
                )

    colorbar = ax.figure.colorbar(
        image,
        ax=ax,
        fraction=0.047,
        pad=0.025,
        aspect=26,
        ticks=np.linspace(VMIN, VMAX, 5),
    )
    colorbar.set_label(cbar_label, rotation=90, labelpad=9, fontsize=16.5)
    colorbar.ax.tick_params(labelsize=15, width=1.2, length=4)
    colorbar.outline.set_linewidth(1.2)


def finalize_figure(
    fig: Figure,
    out_path: Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    close: bool = True,
    pad: float = 0.015,
) -> list[Path]:
    """Write stable vector and raster outputs and return their paths."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        destination = out_path.with_suffix(f".{fmt}")
        fig.savefig(
            destination,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=pad,
            facecolor="white",
        )
        saved.append(destination)
    if close:
        plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir.parent / "data" / "h2_loto_figure_data.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir.parent / "generated" / "h2_loto_results",
        help="Output basename without extension.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib sequential colormap (default: magma).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    src_root = repository_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from cross_domain_nids.release_paths import require_generated_output

    args.output = require_generated_output(
        args.output, "figures", repository_root=repository_root
    )
    matrix, x_labels, y_labels, source = load_heatmap_data(args.data)

    apply_publication_style(FigureStyle())
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    make_heatmap(
        ax,
        matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        cmap=args.cmap,
        cbar_label="Mean JSD",
        annotate=True,
    )
    fig.tight_layout(pad=0.35)
    outputs = finalize_figure(fig, args.output, formats=("pdf", "png"), dpi=600)

    weighted_means = source.assign(
        weighted_jsd=source["num_features"] * source["mean_jsd"]
    ).groupby("target_dataset", sort=False).apply(
        lambda group: group["weighted_jsd"].sum() / group["num_features"].sum(),
        include_groups=False,
    )
    print(f"Validated matrix: {matrix.shape[0]} feature families x {matrix.shape[1]} targets")
    for target in TARGET_ORDER:
        print(f"Weighted mean JSD [{target}]: {weighted_means[target]:.4f}")
    for output in outputs:
        print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
