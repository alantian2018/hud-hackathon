"""Generate polished result charts for FleetForge (Greedy vs Agentic Fleet).

Outputs PNG + SVG into assets/. Numbers mirror the README results table and the
HUD reward validation. Labels are kept truthful: the dollar metric is gross
revenue (not net profit), and the learned policy is shown as an RL-trained model
that currently matches the value-aware heuristic.

    python3 make_results_charts.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

# --- palette -----------------------------------------------------------------
BG = "#0b1220"
PANEL = "#0f1a2e"
GRID = "#22324d"
TEXT = "#e6edf6"
MUTED = "#8aa0bd"
SCENARIO_COLORS = ["#38bdf8", "#a78bfa", "#fb7185", "#fbbf24"]  # sky, violet, rose, amber

# --- data (README results table + reward validation) -------------------------
SCENARIOS = ["Base", "Chase Center\nExit", "Market St\nSurge", "FiDi\nConference"]
ADDITIONAL_TRIPS = [51, 127, 185, 171]
REVENUE_LIFT = [2076.90, 6903.50, 8640.02, 8165.09]          # gross fare, USD
DEMAND_SERVED_LIFT = [15.36, 12.50, 22.48, 20.09]            # percentage points
WAIT_REDUCTION = [0.96, 3.00, 7.77, 5.76]                    # minutes saved

REWARD_LABELS = ["Nearest-car\nbaseline", "Value-aware\nheuristic", "RL-trained\nmodel"]
REWARD_VALUES = [0.268809, 0.320795, 0.321]
REWARD_ERR = [0.0, 0.0, 0.059]
REWARD_COLORS = ["#64748b", "#38bdf8", "#a78bfa"]

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)


def _base_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "figure.dpi": 160,
    })


def _style_axes(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_facecolor(PANEL)


def _label_bars(ax, bars, fmt, dy=0.02):
    top = max(b.get_height() for b in bars)
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2, h + top * dy, fmt(h),
            ha="center", va="bottom", color=TEXT, fontsize=10.5, fontweight="bold",
            path_effects=[withStroke(linewidth=2.5, foreground=BG)],
        )


def _scenario_panel(ax, title, values, fmt, ylabel):
    bars = ax.bar(range(len(SCENARIOS)), values, width=0.66,
                  color=SCENARIO_COLORS, edgecolor="none", zorder=3)
    for b in bars:
        b.set_alpha(0.92)
    _style_axes(ax)
    ax.set_title(title, color=TEXT, fontweight="bold", pad=10, loc="left")
    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels(SCENARIOS, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_ylim(0, max(values) * 1.22)
    _label_bars(ax, bars, fmt)


def scenarios_figure() -> None:
    _base_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.08, hspace=0.42, wspace=0.2)

    fig.text(0.07, 0.955, "Agentic Fleet vs Greedy", fontsize=22, fontweight="bold", color=TEXT)
    fig.text(0.07, 0.915,
             "Lift of the value-aware agentic orchestrator over the greedy baseline, same seeded world per scenario.",
             fontsize=11.5, color=MUTED)

    _scenario_panel(axes[0, 0], "Additional completed trips", ADDITIONAL_TRIPS,
                    lambda v: f"+{v:,.0f}", "trips")
    _scenario_panel(axes[0, 1], "Revenue lift  (gross fare)", REVENUE_LIFT,
                    lambda v: f"+${v:,.0f}", "USD")
    _scenario_panel(axes[1, 0], "Demand served lift", DEMAND_SERVED_LIFT,
                    lambda v: f"+{v:.1f} pp", "percentage points")
    _scenario_panel(axes[1, 1], "Average wait reduction", WAIT_REDUCTION,
                    lambda v: f"{v:.2f} min", "minutes saved")

    fig.text(0.07, 0.02,
             "Source: export_mobility_world.py + precompute_orchestrator_world.py --include-events.  "
             "Dollar figure is gross fare revenue, not net profit.",
             fontsize=8.5, color=MUTED)

    for ext in ("png", "svg"):
        fig.savefig(ASSETS / f"results_scenarios.{ext}", facecolor=BG)
    plt.close(fig)


def reward_figure() -> None:
    _base_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.8, bottom=0.14)

    fig.text(0.12, 0.93, "HUD absolute reward", fontsize=20, fontweight="bold", color=TEXT)
    fig.text(0.12, 0.86,
             "Mean episode reward across the 6-task mobility taskset (higher is better).",
             fontsize=11, color=MUTED)

    x = range(len(REWARD_LABELS))
    bars = ax.bar(x, REWARD_VALUES, width=0.6, color=REWARD_COLORS, zorder=3,
                  yerr=REWARD_ERR, capsize=6,
                  error_kw=dict(ecolor=TEXT, elinewidth=1.6, capthick=1.6, alpha=0.85))
    _style_axes(ax)
    ax.set_xticks(list(x))
    ax.set_xticklabels(REWARD_LABELS, fontsize=10.5)
    ax.set_ylabel("mean reward", fontsize=10)
    ax.set_ylim(0, max(REWARD_VALUES) * 1.25)

    for b, v, e in zip(bars, REWARD_VALUES, REWARD_ERR):
        txt = f"{v:.3f}" + (f"\n± {e:.3f}" if e else "")
        ax.text(b.get_x() + b.get_width() / 2, v + e + max(REWARD_VALUES) * 0.03, txt,
                ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold",
                path_effects=[withStroke(linewidth=2.5, foreground=BG)])

    # Reference line at baseline for quick "lift" read.
    ax.axhline(REWARD_VALUES[0], color=MUTED, linewidth=1, linestyle=(0, (4, 4)), alpha=0.5, zorder=2)

    fig.text(0.12, 0.025,
             "Baseline & heuristic from benchmark_nearest_baseline.py.  "
             "RL-trained model currently matches the heuristic (within noise).",
             fontsize=8.5, color=MUTED)

    for ext in ("png", "svg"):
        fig.savefig(ASSETS / f"results_reward.{ext}", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    scenarios_figure()
    reward_figure()
    print("wrote:", *(str(p) for p in sorted(ASSETS.glob("results_*"))), sep="\n  ")
