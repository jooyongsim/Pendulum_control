"""Side-by-side animation of several controllers in the same arena episode.

Every panel is the same plant, the same start and the same model error -- only the controller
differs, so anything you see between panels is the controller. Labels are ASCII: Colab's
matplotlib has no Korean font and silently draws boxes instead of glyphs.

    from animate import compare
    compare([("expert", arena.expert()), ("dagger", c)], task="swingup", offset_deg=2.0)

The rod is drawn in the vertical plane of the arm tip and the pivot slides with phi * R_ARM,
which is the cart-pole picture the linearised model is built on. It is a reading aid, not a
rendering of the real geometry -- the arm actually sweeps a circle.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import arena
import rotary_model as rm

ROD_L = rm.L                     # 0.262 m
R_ARM = rm.R_ARM                 # 0.127 m
UP_BAND = 10.0                   # the +-10 deg guide drawn behind the rod


def compare(entries, task="swingup", theta0_deg=None, seconds=None, stride=5, fps=20,
            title=None, **plant):
    """entries = [(label, ctrl), ...]. Returns (HTML animation, {label: score dict})."""
    logs, scores = [], {}
    for label, ctrl in entries:
        lg = arena.run(ctrl, task, theta0_deg=theta0_deg, seconds=seconds, **plant)
        logs.append((label, lg))
        scores[label] = arena.score(lg)

    n = len(logs)
    frames = range(0, len(logs[0][1]["t"]), stride)
    span = R_ARM * math.radians(arena.ARM_LIMIT_DEG) * 0.55

    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.9))
    axes = np.atleast_1d(axes)
    arts = []
    for ax, (label, lg) in zip(axes, logs):
        ax.set_xlim(-span, span)
        ax.set_ylim(-0.08, ROD_L * 1.25)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in (-1, 1):                                  # the +-10 deg cone
            ax.plot([0, s * ROD_L * math.sin(math.radians(UP_BAND))],
                    [0, ROD_L * math.cos(math.radians(UP_BAND))],
                    color="0.85", lw=1, zorder=0)
        ax.axhline(0.0, color="0.85", lw=1, zorder=0)
        ax.set_title(label, fontsize=10)
        rod, = ax.plot([], [], lw=3, color="#1f77b4", solid_capstyle="round")
        cart, = ax.plot([], [], lw=7, color="0.35", solid_capstyle="round")
        txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=8,
                      family="monospace")
        arts.append((lg, rod, cart, txt))

    head = title or (f"{task} | " + " ".join(f"{k}={v}" for k, v in plant.items()) or
                     f"{task} | clean")
    fig.suptitle(head, fontsize=10)
    fig.tight_layout()

    def draw(i):
        out = []
        for lg, rod, cart, txt in arts:
            x0 = np.clip(lg["phi"][i] * R_ARM, -span * 0.9, span * 0.9)
            th = lg["th"][i]
            rod.set_data([x0, x0 + ROD_L * math.sin(th)], [0.0, ROD_L * math.cos(th)])
            cart.set_data([x0 - 0.02, x0 + 0.02], [0.0, 0.0])
            stopped = bool(lg["stop"]) and i >= len(lg["t"]) - 1
            rod.set_color("#d62728" if stopped else "#1f77b4")
            txt.set_text(f"t {lg['t'][i]:5.2f}\nth {math.degrees(th):+6.1f}\n"
                         f"phi {math.degrees(lg['phi'][i]):+6.0f}"
                         + (f"\n{lg['stop']}" if stopped else ""))
            out += [rod, cart, txt]
        return out

    ani = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=True)
    plt.close(fig)
    return ani, scores


def traces(entries, task="swingup", theta0_deg=None, seconds=None, **plant):
    """Static theta(t) and phi(t) overlay -- cheaper than an animation and easier to read."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for label, ctrl in entries:
        lg = arena.run(ctrl, task, theta0_deg=theta0_deg, seconds=seconds, **plant)
        a1.plot(lg["t"], np.degrees(lg["th"]), lw=1.2, label=label)
        a2.plot(lg["t"], np.degrees(lg["phi"]), lw=1.2, label=label)
    for v in (-UP_BAND, UP_BAND):
        a1.axhline(v, color="0.8", lw=1, ls="--")
    for v in (-arena.ARM_LIMIT_DEG, arena.ARM_LIMIT_DEG):
        a2.axhline(v, color="#d62728", lw=1, ls="--")
    a1.set_ylabel("theta [deg]")
    a2.set_ylabel("phi [deg]")
    a2.set_xlabel("t [s]")
    a1.set_title(f"{task} | " + (" ".join(f"{k}={v}" for k, v in plant.items()) or "clean"),
                 fontsize=10)
    a1.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    return fig
