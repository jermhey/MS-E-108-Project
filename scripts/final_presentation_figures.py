#!/usr/bin/env python3
"""
Final Stanford MS&E presentation figures for Tauroi Technologies.

Generates four high-resolution transparent PNGs:
1. slide09_anatomy_of_the_shield.png
2. slide10_universal_heuristics_table.png
3. slide10_tuning_curve.png
4. slide11_relative_outperformance.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


BASE = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Global palette
BG = "#1a1a1a"
TEXT = "#cccccc"
SUBTLE = "#999999"
SPINE = "#444444"
GRID = "#3a3a3a"
NEON = "#39FF14"
GRAY = "#555555"
RED = "#AA5555"
CYAN = "#00E5FF"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "axes.titlecolor": TEXT,
    }
)


def style_axis(ax: plt.Axes, y_grid: bool = True) -> None:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors=TEXT, labelsize=12)
    ax.set_axisbelow(True)
    if y_grid:
        ax.grid(axis="y", color=GRID, linestyle=":", linewidth=1.0, alpha=0.45)
    else:
        ax.grid(False)


def glow_effect(color: str, linewidth: float = 8.0, alpha: float = 0.28):
    rgb = plt.matplotlib.colors.to_rgba(color, alpha=alpha)
    return [pe.withStroke(linewidth=linewidth, foreground=rgb)]


def add_bar_labels(ax: plt.Axes, bars, formatter, color: str = TEXT, offset_ratio: float = 0.02) -> None:
    ymax = max(bar.get_height() for bar in bars)
    offset = max(ymax * offset_ratio, 1.0)
    for bar in bars:
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(
            x,
            y + offset,
            formatter(y),
            ha="center",
            va="bottom",
            color=color,
            fontsize=12,
            fontweight="bold",
        )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"Saved: {path}")


def figure_1_anatomy_of_the_shield() -> None:
    labels = ["Opportunity Cost", "Toxic Loss Avoided"]
    values = [4618, 22657]
    colors = [GRAY, NEON]

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    fig.patch.set_facecolor(BG)
    style_axis(ax)

    bars = ax.bar(labels, values, color=colors, width=0.62, edgecolor="none", zorder=3)
    bars[1].set_path_effects(glow_effect(NEON, linewidth=12, alpha=0.22))

    ax.set_ylabel("Cents (¢)", fontsize=13)
    ax.set_ylim(0, 25000)
    ax.tick_params(axis="x", labelrotation=0)
    add_bar_labels(ax, bars, lambda v: f"{int(v):,}¢")
    ax.set_title("Single MLB Game: TOR vs LAD", fontsize=15, pad=14, color=TEXT)

    path = FIGURES_DIR / "slide09_anatomy_of_the_shield.png"
    save_figure(fig, path)


def figure_2_universal_heuristics_table() -> None:
    soft_neon = "#2FD80E"
    markets = ["MLB", "Weather", "Election", "Spotify"]
    metric_labels = ["Activity", "Move / Trade", "Toxicity Premium"]
    metric_values = np.array(
        [
            [289.87, 0.37, 1.63],
            [29.18, 1.00, 1.13],
            [0.76, 0.82, 1.45],
            [0.57, 1.71, 1.35],
        ],
        dtype=float,
    )
    metric_text = [
        ["289.9/hr", "0.37c", "1.63x"],
        ["29.2/hr", "1.00c", "1.13x"],
        ["0.8/hr", "0.82c", "1.45x"],
        ["0.6/hr", "1.71c", "1.35x"],
    ]

    heat = metric_values.copy()
    heat[:, 0] = np.log10(heat[:, 0] + 1.0)
    col_min = heat.min(axis=0)
    col_span = np.maximum(heat.max(axis=0) - col_min, 1e-9)
    heat = (heat - col_min) / col_span

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "tauroi_heat",
        ["#202020", "#29301F", soft_neon],
    )

    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.imshow(heat, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metric_labels)), metric_labels)
    ax.set_yticks(np.arange(len(markets)), markets)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelsize=12, colors=TEXT)

    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(1.0)

    ax.set_xticks(np.arange(-0.5, len(metric_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(markets), 1), minor=True)
    ax.grid(which="minor", color=SPINE, linestyle="-", linewidth=1.0, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(markets)):
        for j in range(len(metric_labels)):
            ax.text(
                j,
                i,
                metric_text[i][j],
                ha="center",
                va="center",
                color="#F5F5F5",
                fontsize=12,
                fontweight="bold",
            )

    ax.text(
        1.0,
        -1.25,
        "Market Microstructure Fingerprint",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
        ha="center",
    )

    ax.set_xlim(-0.5, 2.5)

    path = FIGURES_DIR / "slide10_universal_heuristics_table.png"
    save_figure(fig, path)


def figure_3_tuning_curve() -> None:
    x = np.array([0.1, 0.2, 0.3], dtype=float)
    y = np.array([1489, 1495, 1505], dtype=float)
    peak_x = 0.3
    peak_y = 1505.0

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.patch.set_facecolor(BG)
    style_axis(ax)

    ax.plot(x, y, color=CYAN, linewidth=3.0, marker="o", markersize=6, zorder=3)
    ax.lines[0].set_path_effects(glow_effect(CYAN, linewidth=10, alpha=0.22))

    ax.scatter([peak_x], [peak_y], s=90, color=NEON, edgecolor="none", zorder=4)
    ax.axvline(peak_x, color=NEON, linestyle=":", linewidth=1.5, alpha=0.8, zorder=2)
    ax.text(
        peak_x - 0.06,
        peak_y + 28,
        "Chosen tau = 0.3",
        color=NEON,
        fontsize=12,
        fontweight="bold",
        path_effects=glow_effect(NEON, linewidth=6, alpha=0.12),
    )

    ax.set_xlabel("Tau Threshold", fontsize=13)
    ax.set_ylabel("Validation Improvement vs. Naive (¢)", fontsize=13)
    ax.set_xlim(0.08, 0.32)
    ax.set_ylim(1450, 1525)
    ax.set_xticks(x)
    ax.set_title("Spotify Holdout Tuning Curve", fontsize=15, pad=14, color=TEXT)

    path = FIGURES_DIR / "slide10_tuning_curve.png"
    save_figure(fig, path)


def figure_4_relative_outperformance() -> None:
    markets = ["MLB", "Spotify", "Weather"]
    deltas = [2436, 3231, 18123]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    fig.patch.set_facecolor(BG)
    style_axis(ax)

    bars = ax.bar(markets, deltas, color=NEON, width=0.62, edgecolor="none", zorder=3)
    for bar in bars:
        bar.set_path_effects(glow_effect(NEON, linewidth=12, alpha=0.22))

    ax.set_ylabel("Improvement vs. Naive Quoting (¢)", fontsize=13)
    ax.set_ylim(0, 20000)
    add_bar_labels(ax, bars, lambda v: f"+{int(v):,}¢")

    path = FIGURES_DIR / "slide11_relative_outperformance.png"
    save_figure(fig, path)


def main() -> None:
    figure_1_anatomy_of_the_shield()
    figure_2_universal_heuristics_table()
    figure_3_tuning_curve()
    figure_4_relative_outperformance()


if __name__ == "__main__":
    main()
