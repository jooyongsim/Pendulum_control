"""Figures for the Furuta pendulum swing-up + LQR deck.

    furuta_arch.jpg     the loop: encoder -> estimator -> two controllers -> motor
    furuta_switch.jpg   the hysteresis band that decides which controller runs

Drawn, not measured: these illustrate the structure the deck describes. The
labels are English because matplotlib has no Korean face in this environment.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

NAVY, ORANGE, GREEN, GREY = "#203864", "#E97820", "#2e7d4f", "#5a6070"
CREAM, ICE = "#f6efe4", "#eef3fa"


def box(ax, x, y, w, h, text, fc=ICE, ec=NAVY, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                fc=fc, ec=ec, lw=1.5, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, fontweight="bold" if bold else "normal", color=NAVY)


def arrow(ax, x0, y0, x1, y1, color=NAVY, text=None, dy=1.8, fs=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.6, color=color, zorder=2))
    if text:
        ax.text((x0 + x1) / 2, max(y0, y1) + dy, text, ha="center", fontsize=fs,
                color=color, zorder=3)


# ------------------------------------------------------------ 1. architecture
fig, ax = plt.subplots(figsize=(9.6, 3.9))
ax.set_axis_off()
ax.set(xlim=(0, 100), ylim=(0, 42))

box(ax, 2, 18, 13, 8, "Encoder\n" + r"$\theta,\ \alpha$")
box(ax, 20, 18, 15, 8, "Velocity\nestimator", fc=CREAM)
box(ax, 42, 28, 20, 9, "Energy swing-up", fc="#fdf0e2", ec=ORANGE)
box(ax, 42, 8, 20, 9, "LQR  " + r"$u=-Kx$", fc="#e9f3ec", ec=GREEN)
box(ax, 69, 18, 12, 8, "Saturation", fc=CREAM)
box(ax, 86, 18, 12, 8, "Motor", bold=True)

arrow(ax, 15, 22, 20, 22)
arrow(ax, 35, 22, 41, 31)
arrow(ax, 35, 22, 41, 13)
arrow(ax, 62, 32, 68, 24)
arrow(ax, 62, 12, 68, 21)
arrow(ax, 81, 22, 86, 22)

ax.annotate("", xy=(50, 27), xytext=(50, 18), zorder=1,
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2, ls=(0, (3, 2))))
ax.text(52.5, 22.5, "switch", ha="left", va="center", fontsize=9.5,
        color=GREY, style="italic")

# feedback path
ax.add_patch(FancyArrowPatch((92, 18), (92, 4), arrowstyle="-", lw=1.4, color=GREY))
ax.add_patch(FancyArrowPatch((92, 4), (10, 4), arrowstyle="-", lw=1.4, color=GREY))
ax.add_patch(FancyArrowPatch((10, 4), (10, 17), arrowstyle="-|>", mutation_scale=14,
                             lw=1.4, color=GREY))
ax.text(51, 1.2, "the plant: arm angle and pendulum angle come back every sample",
        ha="center", fontsize=9, color=GREY)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "furuta_arch.jpg"), dpi=150, facecolor="white")
plt.close(fig)


# -------------------------------------------------------------- 2. hysteresis
fig, ax = plt.subplots(figsize=(8.4, 3.4))
ax.set_axis_off()
ax.set(xlim=(0, 100), ylim=(0, 34))

ax.add_patch(plt.Rectangle((10, 6), 22, 12, fc="#fdf0e2", ec=ORANGE, lw=1.8, zorder=2))
ax.text(21, 12, "SWING-UP", ha="center", va="center", fontsize=12,
        fontweight="bold", color=ORANGE, zorder=3)
ax.add_patch(plt.Rectangle((66, 6), 22, 12, fc="#e9f3ec", ec=GREEN, lw=1.8, zorder=2))
ax.text(77, 12, "LQR", ha="center", va="center", fontsize=12,
        fontweight="bold", color=GREEN, zorder=3)

ax.add_patch(FancyArrowPatch((32, 15), (66, 15), arrowstyle="-|>", mutation_scale=16,
                             lw=1.8, color=GREEN, connectionstyle="arc3,rad=-0.28"))
ax.text(49, 29.5, r"enter:  $|\alpha| < 15°$  and  $|\dot\alpha| < 2$ rad/s",
        ha="center", fontsize=11, color=GREEN)

ax.add_patch(FancyArrowPatch((66, 9), (32, 9), arrowstyle="-|>", mutation_scale=16,
                             lw=1.8, color=ORANGE, connectionstyle="arc3,rad=-0.28"))
ax.text(49, 0.8, r"exit:  $|\alpha| > 25°$", ha="center", fontsize=11, color=ORANGE)

ax.text(49, 12, "10° of\nhysteresis", ha="center", va="center", fontsize=9.5,
        color=GREY, style="italic")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "furuta_switch.jpg"), dpi=150, facecolor="white")
plt.close(fig)

from PIL import Image  # noqa: E402
for f in ("furuta_arch.jpg", "furuta_switch.jpg"):
    with Image.open(os.path.join(FIGS, f)) as im:
        print("  %-20s %dx%d  aspect %.2f" % (f, *im.size, im.size[0] / im.size[1]))
