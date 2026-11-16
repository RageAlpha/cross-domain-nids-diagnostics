"""Regenerate current H1, matched H3 and random-deletion figures from released evidence."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from cross_domain_nids.release_paths import require_generated_output

def plot_h1(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    
    # ── Data (mean Binary-F1 over seeds 42 / 123 / 2024) ──────────────────────
    import pandas as pd
    frame = pd.read_csv(ROOT / 'results/reference/h1/seed_metrics.csv')
    data = {}
    for target in ['CICIDS2018', 'BoT-IoT', 'ToN-IoT']:
        data[target] = {}
        for name, model in [('MLP','MLP'),('RF','RandomForest'),('XGB','XGBoost')]:
            group = frame[(frame.target_dataset == target) & (frame.model == model)]
            arms = {arm: group[group.experiment == arm].binary_f1 for arm in ['E1','E2']}
            if any(len(values) != 3 for values in arms.values()):
                raise ValueError('H1 figure requires three seeds per target/model/arm')
            data[target][name] = dict(E1=arms['E1'].mean(), E2=arms['E2'].mean(),
                                      sd1=arms['E1'].std(ddof=1), sd2=arms['E2'].std(ddof=1))
    
    targets = ["CICIDS2018", "BoT-IoT", "ToN-IoT"]
    models  = ["MLP", "RF", "XGB"]
    
    colors  = {"MLP": "#4C72B0", "RF": "#55A868", "XGB": "#C44E52"}
    markers = {"MLP": "o",        "RF": "s",        "XGB": "^"}
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), sharey=False)
    
    for ax, target in zip(axes, targets):
        ax.set_title(target, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["E1\n(multiclass)", "E2\n(binary)"], fontsize=8)
        ax.set_ylabel("Binary-F1" if target == "CICIDS2018" else "", fontsize=8)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.5, color="lightgrey", linewidth=0.7, linestyle="--", zorder=0)
    
        # x jitter so error bars of different models don't overlap
        jitter = {"MLP": -0.12, "RF": 0.0, "XGB": 0.12}
    
        for model in models:
            v  = data[target][model]
            x1 = 0 + jitter[model]
            x2 = 1 + jitter[model]
            col = colors[model]
            mk  = markers[model]
    
            # connecting line
            ax.plot([x1, x2], [v["E1"], v["E2"]],
                    color=col, linewidth=1.4, zorder=2,
                    linestyle="-" if v["E2"] >= v["E1"] else "--")
    
            # error bars (±1 SD)
            ax.errorbar(x1, v["E1"], yerr=v["sd1"],
                        fmt=mk, color=col, markersize=5,
                        capsize=3, capthick=1, linewidth=1, zorder=3)
            ax.errorbar(x2, v["E2"], yerr=v["sd2"],
                        fmt=mk, color=col, markersize=5,
                        capsize=3, capthick=1, linewidth=1, zorder=3)
    
    # Legend
    handles = [mpatches.Patch(color=colors[m], label=m) for m in models]
    # solid = increase, dashed = decrease
    line_inc = plt.Line2D([0], [0], color="grey", linewidth=1.4, linestyle="-",  label="E2 ≥ E1")
    line_dec = plt.Line2D([0], [0], color="grey", linewidth=1.4, linestyle="--", label="E2 < E1")
    fig.legend(handles=handles + [line_inc, line_dec],
               loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    
    fig.suptitle(
        "H1: E1 (multiclass source) vs E2 (binary source) Binary-F1 by target and model family\n"
        "Solid lines = E2 improves; dashed lines = E2 degrades. Error bars: ±1 SD over seeds.",
        fontsize=8, y=1.01
    )
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output / "figure_4_h1.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output / "figure_4_h1.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved figure4.pdf")

def plot_h3(output):
    from pathlib import Path
    import hashlib
    import json
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np
    import pandas as pd
    
    
    OUTPUT = output
    SOURCE = ROOT / 'results/reference/revision/core/phase2_key_metrics.csv'
    GATES = ROOT / 'results/reference/revision/core/recomputed_h3_gates.csv'
    TARGETS = ["BoT-IoT", "CICIDS2018", "ToN-IoT"]
    SEEDS = [42, 123, 2024]
    METRICS = ["MCC", "Binary_F1", "FNR", "FPR", "MMD", "domain_discriminability"]
    COLORS = {"H3-C0": "#6b7280", "H3-C1": "#0072b2"}
    MARKERS = {42: "o", 123: "^", 2024: "s"}
    WIDTH, HEIGHT = 521.77875, 353.0084
    
    
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def reviewed_data():
        source = pd.read_csv(SOURCE)
        h3 = source[(source.analysis_version == "corrected_10_masks") & (source.experiment == "H3")]
        absolute = h3[(h3.value_type == "absolute") & (h3.training_seed != "mean")].copy()
        absolute["seed"] = absolute.training_seed.astype(int)
        expected = pd.MultiIndex.from_product([TARGETS, SEEDS, ["H3-C0", "H3-C1"]], names=["target", "seed", "condition"])
        actual = absolute.set_index(["target", "seed", "condition"]).sort_index()
        if not actual.index.equals(expected.sort_values()) or not np.isfinite(actual[METRICS]).all().all():
            raise ValueError("Expected 18 unique corrected H3 model evaluations")
        gates = pd.read_csv(GATES).set_index("target")
        means, deltas = {}, {}
        for target in TARGETS:
            frame = actual.loc[target]
            mean = frame.groupby("condition")[METRICS].mean()
            delta = mean.loc["H3-C1"] - mean.loc["H3-C0"]
            for condition in ["H3-C0", "H3-C1"]:
                saved = h3[(h3.target == target) & (h3.condition == condition) & (h3.training_seed == "mean")]
                np.testing.assert_allclose(mean.loc[condition], saved[METRICS].iloc[0], atol=1e-10)
            saved_delta = h3[(h3.target == target) & (h3.condition == "C1_minus_C0") & (h3.training_seed == "mean")]
            np.testing.assert_allclose(delta, saved_delta[METRICS].iloc[0], atol=1e-10)
            rep = (delta[["MMD", "domain_discriminability"]] <= 0).all()
            if rep != gates.loc[target, "representation_gate"]:
                raise ValueError(f"Representation gate differs for {target}")
            means[target], deltas[target] = mean, delta
        return actual, means, deltas
    
    
    def add_panel(fig, bounds, title, metric, limits, ticks, data, means, deltas, lower=False):
        left, top, right, bottom = bounds
        ax = fig.add_axes([left / WIDTH, (HEIGHT - bottom) / HEIGHT, (right - left) / WIDTH, (bottom - top) / HEIGHT])
        ax.set_xlim(-.46, 2.46)
        ax.set_ylim(*limits)
        ax.set_title(title, fontsize=8.8, fontweight="bold", pad=5)
        ax.axvspan(1.6, 2.4, color="#fff1bf", alpha=.55, zorder=0)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f" if metric == "domain_discriminability" else "%.2f"))
        ax.grid(axis="y", color="#d8dce2", linewidth=.55)
        for boundary in [.5, 1.5]:
            ax.axvline(boundary, color="#d8dce2", linewidth=.55, zorder=0)
        if metric in ["MCC", "domain_discriminability"]:
            ax.axhline(0 if metric == "MCC" else .5, color="#747a84", linewidth=.6, linestyle=(0, (4, 2)))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#5f6570")
            ax.spines[spine].set_linewidth(.65)
        ax.tick_params(axis="y", labelsize=7.1, width=.55, length=3)
        ax.tick_params(axis="x", length=0, labelsize=7.1, pad=2)
        x_values = [x for center in range(3) for x in [center - .14, center + .14]]
        ax.set_xticks(x_values)
        ax.set_xticklabels(["C0", "C1"] * 3 if lower else [""] * 6)
        for center, target in enumerate(TARGETS):
            for seed, jitter in zip(SEEDS, [-.028, 0, .028]):
                values = [data.loc[(target, seed, condition), metric] for condition in COLORS]
                if not all(limits[0] <= value <= limits[1] for value in values):
                    raise ValueError(f"Clipped values: {target}, {seed}, {metric}")
                x = np.array([center - .14, center + .14]) + jitter
                ax.plot(x, values, color="#8a8f98", alpha=.82, linewidth=.85, zorder=2)
                for position, value, condition in zip(x, values, COLORS):
                    ax.scatter(position, value, s=23, marker=MARKERS[seed], color=COLORS[condition], edgecolor="white", linewidth=.45, zorder=3)
            x = [center - .14, center + .14]
            values = [means[target].loc[condition, metric] for condition in COLORS]
            ax.plot(x, values, color="#111827", linewidth=1.65, zorder=4)
            for position, value, condition in zip(x, values, COLORS):
                ax.scatter(position, value, s=52, marker="D", color=COLORS[condition], edgecolor="#111827", linewidth=.85, zorder=5)
            if lower:
                ax.text(center, -.20, target, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.2, fontweight="bold" if target == "ToN-IoT" else "normal")
        precision = 5 if metric == "domain_discriminability" else 3
        delta_text = f"ToN mean Δ={deltas['ToN-IoT'][metric]:+.{precision}f}"
        # Keep the two detection annotations clear of the highest ToN-IoT markers.
        y = .94 if metric in ("MCC", "Binary_F1") else .84
        ax.text(.84, y, delta_text, transform=ax.transAxes, ha="center", va="center", fontsize=7,
                color="#111827", bbox=dict(boxstyle="round,pad=.14", facecolor="white", edgecolor="#c8b676", linewidth=.55), zorder=6)
        return ax
    
    
    def main():
        data, means, deltas = reviewed_data()
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.1, "axes.unicode_minus": False,
                             "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})
        fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor="white")
        panels = [
            ((26.879375, 27.3636, 238.941875, 133.767078), "MCC (higher is better)", "MCC", (-.22, .84), [-.2, 0, .2, .4, .6, .8], False),
            ((289.836875, 27.3636, 501.899375, 133.767078), "MMD (lower is better)", "MMD", (0, .76), [0, .15, .30, .45, .60, .75], False),
            ((26.879375, 165.688122, 238.941875, 272.0916), "Binary-F1 (higher is better)", "Binary_F1", (0, .96), [0, .2, .4, .6, .8], True),
            ((289.836875, 165.688122, 501.899375, 272.0916), "Domain discriminability (lower is better; zoomed)", "domain_discriminability", (.494, .5002), [.495, .496, .497, .498, .499, .500], True),
        ]
        for bounds, title, metric, limits, ticks, lower in panels:
            add_panel(fig, bounds, title, metric, limits, ticks, data, means, deltas, lower)
        for x, title in [(132.910625, "(a) Target detection performance"), (395.868125, "(b) Representation discrepancy")]:
            fig.text(x / WIDTH, .992, title, ha="center", va="top", fontsize=10.2, fontweight="bold")
        fig.text(.5, .052, "Paired lines connect the same training seed; diamonds show means. Shading highlights the ToN-IoT example.",
                 ha="center", va="bottom", fontsize=7, color="#454d5a")
        handles = [Line2D([], [], marker="o", linestyle="none", color=color, markeredgecolor="white", markersize=4.6, label=condition[-2:]) for condition, color in COLORS.items()]
        handles += [Line2D([], [], marker="D", color="#111827", markerfacecolor="white", markersize=5, linewidth=1.65, label="3-seed mean")]
        handles += [Line2D([], [], marker=MARKERS[seed], linestyle="none", markerfacecolor="#9ca3af", markeredgecolor="black", markersize=4.6, label=f"seed {seed}") for seed in SEEDS]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .004), ncol=6, frameon=False, fontsize=7, handletextpad=.55, columnspacing=1.05, borderaxespad=0)
        stem = OUTPUT / "figure_3_h3"
        fig.savefig(stem.with_suffix(".pdf"), metadata={"Title": "Matched H3 performance-representation dissociation", "Creator": "plot_h3_dissociation.py"})
        fig.savefig(stem.with_suffix(".png"), dpi=200)
        fig.savefig(stem.with_suffix(".svg"))
        plt.close(fig)
        print("PASS current H3 values and representation gates checked.")
    
    main()

def plot_random(output):
    from pathlib import Path
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd
    
    
    out = output
    data = ROOT / 'results/reference/revision/sensitivity'
    masks = pd.read_csv(data / "random_per_mask.csv").sort_values("replicate")
    gates = pd.read_csv(data / "gate_deltas.csv")
    shift = gates.loc[
        (gates.analysis_version == "corrected_10_masks")
        & (gates.method == "H2")
        & (gates.target == "ToN-IoT")
        & (gates.model == "MLP")
        & (gates.selection == "remove_shift_top10")
        & (gates.comparator == "full")
    ].set_index("seed")
    seeds = [42, 123, 2024]
    sizes = [3, 5, 10, 15]
    metrics = [
        ("mcc", "MCC", "MCC", [-0.10, 0.75], [0, 0.2, 0.4, 0.6]),
        ("binary_f1", "Binary_F1", "Binary-F1", [0.38, 0.83], [0.4, 0.5, 0.6, 0.7, 0.8]),
        ("fnr", "FNR", "FNR", [0.22, 0.59], [0.25, 0.35, 0.45, 0.55]),
    ]
    
    plt.rcParams.update({"font.size": 8.5, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 3, figsize=(7.15, 4.85), sharex=True, sharey="row")
    fig.subplots_adjust(left=.1, right=.985, bottom=.115, top=.845, wspace=.14, hspace=.22)
    for col, seed in enumerate(seeds):
        within_seed = masks[masks.seed == seed]
        axes[0, col].set_title(f"Seed {seed}", fontsize=10)
        for row, (metric, delta, label, limits, ticks) in enumerate(metrics):
            ax = axes[row, col]
            baseline = within_seed[f"{metric}_full"].unique().item()
            selected = baseline + shift.loc[seed, delta]
            ax.axhline(baseline, color="#52525b", linestyle="--", linewidth=.9)
            for x, size in enumerate(sizes):
                values = within_seed.loc[within_seed.k == size, metric]
                offsets = np.linspace(-.24, .24, len(values))
                ax.scatter(x + offsets, values, s=13, color="#0072b2", alpha=.74,
                           edgecolors="none", zorder=3)
                ax.hlines(values.median(), x - .27, x + .27, color="#172432",
                          linewidth=1.45, zorder=4)
            ax.scatter(sizes.index(10), selected, marker="D", s=32, color="#c34c10",
                       edgecolors="white", linewidth=.45, zorder=5)
            ax.set(xlim=(-.55, 3.55), ylim=limits, yticks=ticks, xticks=range(4),
                   xticklabels=sizes)
            if col == 0:
                ax.set_ylabel(label)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#b2b6bb")
            ax.tick_params(length=2.5, color="#8d939b")
            ax.grid(axis="y", color="#e1e4e8", linewidth=.5)
            ax.set_axisbelow(True)
    fig.supxlabel("Number of features removed", y=.022, fontsize=9.5)
    handles = [
        Line2D([], [], linestyle="none", marker="o", color="#0072b2", markersize=4, label="Random mask"),
        Line2D([], [], color="#172432", linewidth=1.45, label="Random median"),
        Line2D([], [], color="#52525b", linestyle="--", linewidth=.9, label="Full-feature E2"),
        Line2D([], [], linestyle="none", marker="D", color="#c34c10", markersize=5, label="Selected shift mask"),
    ]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(.53, .993),
               frameon=False, fontsize=8.5, columnspacing=1.5, handlelength=1.7)
    fig.savefig(out / "figure_5_random_deletion.png", dpi=200)
    fig.savefig(out / "figure_5_random_deletion.pdf")
    fig.savefig(out / "figure_5_random_deletion.svg")
    plt.close(fig)

def plot_context(output):
    """Regenerate historical appendix context from compact archived counts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    shift = pd.read_csv(ROOT / 'figures/data/historical_feature_shift.csv')
    targets = ['CICIDS2018', 'BoT-IoT', 'ToN-IoT']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, target in zip(axes, targets):
        top = shift[shift.target_dataset == target].nsmallest(10, 'rank_by_jsd').iloc[::-1]
        ax.barh(top.feature, top.jsd_mean, color='#0072b2')
        ax.set(title=target, xlabel='JSD (archived source partitions)', xlim=(0, 1))
        ax.tick_params(axis='y', labelsize=6)
    fig.tight_layout()
    fig.savefig(output / 'appendix_feature_shift.pdf')
    plt.close(fig)

    counts = pd.read_csv(ROOT / 'figures/data/historical_class_confusion.csv')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, target in zip(axes, targets):
        selected = counts[(counts.target_dataset == target) & (counts.model == 'RandomForest')]
        matrix = selected.groupby(['target_attack_type', 'predicted_source_class'])['count'].sum().unstack(fill_value=0)
        fractions = matrix.div(matrix.sum(axis=1), axis=0)
        if not np.allclose(fractions.sum(axis=1), 1):
            raise ValueError('Historical confusion rows do not normalize to one')
        image = ax.imshow(fractions, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set(title=target, xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
               yticks=range(len(matrix)), yticklabels=matrix.index)
        ax.tick_params(axis='x', labelrotation=90, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
    fig.suptitle('Historical RF source-class mapping; counts pooled over seeds (context only)')
    fig.colorbar(image, ax=list(axes), shrink=.6, label='Fraction within target attack type')
    fig.subplots_adjust(left=.08, right=.8, bottom=.25, wspace=.65)
    fig.savefig(output / 'appendix_historical_class_confusion.pdf')
    plt.close(fig)

    summary = pd.read_csv(ROOT / 'results/reference/calibration/summary.csv')
    chosen = summary[(summary.experiment == 'E2') & (summary.model == 'XGBoost')
                     & (summary.filtering_strategy == 'none')].set_index('target_dataset').loc[targets]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, metric, label in zip(axes, ['ece', 'brier_score'], ['ECE', 'Brier score']):
        ax.bar(targets, chosen[metric + '_mean'], yerr=chosen[metric + '_std'],
               color='#0072b2', capsize=3)
        ax.set(ylabel=label, title='Fixed E2 / XGBoost')
    fig.tight_layout()
    fig.savefig(output / 'appendix_calibration_summary.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures/generated/revision')
    args = parser.parse_args()
    output = require_generated_output(args.output_dir, 'figures')
    output.mkdir(parents=True, exist_ok=True)
    plot_h1(output)
    plot_h3(output)
    plot_random(output)
    plot_context(output)
    print(f'Wrote current manuscript figures to {output}')

if __name__ == '__main__':
    main()
