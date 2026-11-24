"""Step 5. Three charts and the headline numbers.

    python 05_report.py            a few seconds

Writes results/charts/methods.png, agreement.png and probes.png, and results/summary.json.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, read_json, write_json

CHARTS = RESULTS / "charts"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, PALE = "#2a78d6", "#eb6834", "#86b6ef"      # one color per thing; every mark also carries a written label
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": AXIS, "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2})


def clean(ax, grid_axis="x"):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def titled(fig, title, subtitle, legend_rows=0):
    height = fig.get_figheight()
    fig.text(0.012, 1 - 0.10 / height, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 1 - 0.40 / height, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)
    return 1 - (0.72 + 0.30 * legend_rows) / height


def chart_methods(models):
    kinds = models["models"][::-1]
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
    for i, kind in enumerate(kinds):
        m = models["test"][kind]
        ax.barh(i, m["mae_points"], height=0.5, color=BLUE if kind == models["chosen"] else PALE, zorder=3)
        ax.text(m["mae_points"] + 0.25, i, f"{m['mae_points']:.1f}" + ("   chosen" if kind == models["chosen"] else ""), va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(kinds)), [k.capitalize() for k in kinds], fontsize=9.5, color=INK)
    ax.set_xlim(0, 21)
    ax.set_xlabel("Mean error on the test responses, in points of 100. Lower is better.", loc="left")
    clean(ax)
    top = titled(fig, "How far off is each method?", f"{models['parts']['test']:,} test responses. The score runs from 0 to 100 within each prompt.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(CHARTS / "methods.png")
    plt.close(fig)


def chart_agreement(checks):
    rows = checks["label"]["by_prompt"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=200)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        a, b = r["kappa_rater_1_against_rater_2"], r["kappa_model_against_rater_2"]
        ax.plot([a, b], [yi, yi], color=AXIS, linewidth=2, zorder=2)
        ax.scatter([a], [yi], s=60, color=ORANGE, edgecolors=SURFACE, linewidths=2, zorder=4)
        ax.scatter([b], [yi], s=60, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=4)
        ax.text(max(a, b) + 0.012, yi, f"rater {a:.2f}, model {b:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [f"Prompt {r['prompt']}" for r in rows], fontsize=9.5, color=INK)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Agreement with rater 2 (quadratic weighted kappa). 1 is perfect.", loc="left")
    clean(ax)
    handles = [plt.Line2D([], [], color=ORANGE, marker="o", linestyle="", markersize=7, label="Rater 1 against rater 2"),
               plt.Line2D([], [], color=BLUE, marker="o", linestyle="", markersize=7, label="A model that learned from rater 1, against rater 2")]
    top = titled(fig, "Is the model as good as a second human rater?", "Test responses. Both are compared with rater 2, whose scores the model never saw.", legend_rows=1)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, top + 0.34 / fig.get_figheight()), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(CHARTS / "agreement.png")
    plt.close(fig)


def chart_probes(checks):
    probes = list(checks["probes"].items())[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 3.2), dpi=200)
    for i, (name, p) in enumerate(probes):
        ax.barh(i + 0.17, p["mean_score_before"], height=0.3, color=PALE, zorder=3)
        ax.barh(i - 0.17, p["mean_score_after"], height=0.3, color=BLUE, zorder=3)
        ax.text(p["mean_score_before"] + 1, i + 0.17, f"{p['mean_score_before']:.0f} before", va="center", fontsize=8.5, color=INK)
        ax.text(p["mean_score_after"] + 1, i - 0.17, f"{p['mean_score_after']:.0f} after", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(probes)), [name.capitalize() for name, _ in probes], fontsize=9.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean predicted score, in points of 100", loc="left")
    clean(ax)
    top = titled(fig, "Three changes a human rater would punish", "What the model predicts before and after each change to the test responses.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(CHARTS / "probes.png")
    plt.close(fig)


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    prepare, models, checks = read_json(RESULTS / "prepare.json"), read_json(RESULTS / "models.json"), read_json(RESULTS / "checks.json")
    chart_methods(models)
    chart_agreement(checks)
    chart_probes(checks)
    chosen = models["chosen"]
    keep = ("mae_points", "r2", "exact", "within_one", "kappa_mean")
    summary = {
        "step_2": {k: prepare[k] for k in ("raw_rows", "duplicate_groups", "duplicate_rows_removed", "duplicate_groups_with_two_different_scores", "kept")},
        "step_3_parts": models["parts"], "step_3_chosen": chosen,
        "step_3_cross_validation": {kind: {k: models["cross_validation"][kind][k] for k in keep} for kind in models["models"]},
        "step_3_tune": {kind: {k: models["tune"][kind][k] for k in keep} for kind in models["models"]},
        "step_3_steps_on_tune": models["steps_on_tune"],
        "step_3_test": {kind: {k: models["test"][kind][k] for k in keep} for kind in models["models"]},
        "step_3_improvement_on_test": models["improvement_on_test"],
        "step_4_label": {k: v for k, v in checks["label"].items() if k != "by_prompt"},
        "step_4_errors_by_score_band": checks["errors"]["by_score_band"], "step_4_errors_by_rater_agreement": checks["errors"]["by_rater_agreement"],
        "step_4_probes": checks["probes"],
        "step_4_new_prompt": {k: v for k, v in checks["new_prompt"].items() if k != "by_prompt"},
    }
    write_json(RESULTS / "summary.json", summary)
    print("done. See results/charts/ and results/summary.json")


if __name__ == "__main__":
    main()
