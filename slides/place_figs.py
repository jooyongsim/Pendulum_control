"""Figures for the pole-placement lecture deck.

    place_poles.jpg    where each design puts the closed-loop poles
    place_runaway.jpg  what the two missing gains cost: the arm never comes back

Both run the linear model the functions themselves are designed on, so the
plots and the gain table in the deck come from the same arithmetic.
Labels are English: matplotlib has no Korean face here.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simulator"))
import rotary_model as rm  # noqa: E402

FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

NAVY, ORANGE, GREEN, GREY = "#203864", "#E97820", "#2e7d4f", "#5a6070"
A, B = rm.state_matrices()

RIG_POLES = [-8 + 8j, -8 - 8j, -2 + 1j, -2 - 1j]


def rig_place4(b, poles):
    """The copy that runs on the rig (swing-up/balance_control.py)."""
    ctrb = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(4)])
    phi = np.zeros_like(A)
    for c in np.real(np.poly(poles)):
        phi = phi @ A + c * np.eye(4)
    return np.linalg.solve(ctrb.T, np.eye(4)[:, -1]) @ phi


DESIGNS = [
    ("place2()  rod only", rm.place2(), ORANGE, "o"),
    ("place4()  simulator", rm.place4(), NAVY, "s"),
    ("place4()  rig default", rig_place4(rm.B_U, RIG_POLES), GREEN, "^"),
]

# ------------------------------------------------------------------ 1. poles
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.axvspan(0, 12, color="#f7d9d9", zorder=0)
ax.axhline(0, color="0.55", lw=0.9, zorder=1)
ax.axvline(0, color="0.55", lw=0.9, zorder=1)

op = np.linalg.eigvals(A)
ax.plot(op.real, op.imag, "x", ms=14, mew=3, color="#c0392b", zorder=2,
        label="open loop (+7.41 unstable)")
for zi, (name, K, color, mk) in enumerate(DESIGNS):
    ev = np.linalg.eigvals(A - B @ np.asarray(K, float).reshape(1, 4))
    ax.plot(ev.real, ev.imag, mk, ms=9, color=color, zorder=6 - zi, label=name)

ax.annotate("two poles stay at the origin\n(the arm is not fed back)",
            xy=(0, 0), xytext=(2.1, 5.4), color=ORANGE, fontsize=9.5,
            ha="left", arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
ax.set(xlim=(-12, 12), ylim=(-10, 10), xlabel="Re s [1/s]", ylabel="Im s [1/s]",
       title="Where each design puts the closed-loop poles")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "place_poles.jpg"), dpi=150, facecolor="white")
plt.close(fig)

# --------------------------------------------------------------- 2. runaway
T, dt = 6.0, 0.001
t = np.arange(0, T, dt)
x0 = np.array([np.radians(5.0), 0.0, 0.0, 0.0])

fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.6), sharex=True)
for name, K, color, _ in DESIGNS[:2]:
    Acl = A - B @ np.asarray(K, float).reshape(1, 4)
    x = np.empty((len(t), 4))
    x[0] = x0
    for i in range(1, len(t)):                     # RK4 on the linear model
        xk = x[i - 1]
        k1 = Acl @ xk
        k2 = Acl @ (xk + 0.5 * dt * k1)
        k3 = Acl @ (xk + 0.5 * dt * k2)
        k4 = Acl @ (xk + dt * k3)
        x[i] = xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    axes[0].plot(t, np.degrees(x[:, 0]), color=color, lw=1.9, label=name)
    axes[1].plot(t, np.degrees(x[:, 2]), color=color, lw=1.9, label=name)

axes[0].set(ylabel="rod  θ [deg]", title="Released 5° from upright")
axes[0].axhline(0, color="0.6", lw=0.8)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)
axes[1].set(xlabel="t [s]", ylabel="arm  φ [deg]")
axes[1].axhline(0, color="0.6", lw=0.8)
axes[1].grid(alpha=0.3)
axes[1].annotate("the arm never comes back", xy=(4.5, -170), fontsize=9.5,
                 color=ORANGE, ha="center")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "place_runaway.jpg"), dpi=150, facecolor="white")
plt.close(fig)

from PIL import Image  # noqa: E402
for f in ("place_poles.jpg", "place_runaway.jpg"):
    with Image.open(os.path.join(FIGS, f)) as im:
        print("  %-20s %dx%d  aspect %.2f" % (f, *im.size, im.size[0] / im.size[1]))
